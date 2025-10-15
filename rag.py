import fitz  # PyMuPDF
import base64
import openai
from mistralai import Mistral
import os
from pinecone import Pinecone, ServerlessSpec
import re
import hashlib



def initialize_rag(openai_api_key, pinecone_api_key, mistral_api_key, index_name="rag-index", storage_folder="rag_storage"):
    """Initialize RAG system using Pinecone for vector storage"""
    os.makedirs(storage_folder, exist_ok=True)
    os.makedirs(os.path.join(storage_folder, "texts"), exist_ok=True)

    openai_client = openai.OpenAI(api_key=openai_api_key)
    mistral_client = Mistral(api_key=mistral_api_key)

    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,  # for text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    return {
        "client": openai_client,
        "mistral_client": mistral_client,
        "index": index,
        "documents": [],
        "storage_folder": storage_folder
    }


def get_pdf_page_count(pdf_path):
    """Get the number of pages in a PDF without extracting images."""
    pdf_document = fitz.open(pdf_path)
    page_count = len(pdf_document)
    pdf_document.close()
    return page_count



def extract_text_with_mistral_ocr(mistral_client, pdf_path, storage_folder):
    """Extract text from entire PDF using Mistral OCR and cache results per page."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Encode PDF to base64
    try:
        with open(pdf_path, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding PDF: {e}")
        return {}

    # Process with Mistral OCR
    try:
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}" 
            },
            include_image_base64=False
        )
        
        # Extract text per page from response
        page_texts = {}
        text_folder = os.path.join(storage_folder, "texts", pdf_name)
        os.makedirs(text_folder, exist_ok=True)
        
        # Parse pages from response
        for page_idx, page in enumerate(ocr_response.pages):
            page_number = page_idx + 1
            extracted_text = page.markdown if hasattr(page, 'markdown') else page.text
            
            # Save to file
            text_path = os.path.join(text_folder, f"page_{page_number:03d}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            page_texts[page_number] = extracted_text
        
        return page_texts
        
    except Exception as e:
        print(f"Error extracting text with Mistral OCR: {e}")
        return {}


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks for better context retrieval."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks



def process_pdf(rag_system, pdf_path):
    mistral_client = rag_system["mistral_client"]
    storage_folder = rag_system["storage_folder"]
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Extract all text from PDF at once using Mistral OCR
    page_texts = extract_text_with_mistral_ocr(mistral_client, pdf_path, storage_folder)
    
    documents = []
    # Process each page's text
    for page_number, text_content in page_texts.items():
        if text_content:
            chunks = chunk_text(text_content)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "chunk_id": i + 1,
                    "page_number": page_number,
                    "content": chunk,
                    "metadata": {
                        "pdf_name": pdf_name,
                        "source": pdf_path,
                        "page": page_number
                    }
                })
    return documents



def add_documents_to_pinecone(rag_system, file_path):
    client = rag_system["client"]
    index = rag_system["index"]

    documents = process_pdf(rag_system, file_path)
    if not documents:
        print("No text found to embed.")
        return

    texts = [doc["content"] for doc in documents]
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embeddings = [item.embedding for item in response.data]

    vectors = []
    for doc, emb in zip(documents, embeddings):
        # Make a Pinecone-safe vector ID (ASCII only)
        file_name = os.path.basename(file_path)
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', file_name)
        # Fallback: hashed ID if filename still contains weird characters
        if not safe_filename.strip("_"):
            safe_filename = hashlib.md5(file_name.encode('utf-8')).hexdigest()

        doc_id = f"{safe_filename}_{doc['chunk_id']}"

        vectors.append({
            "id": doc_id,
            "values": emb,
            "metadata": {
                "original_filename": file_name,
                "source": file_path,
                **doc["metadata"],
                "content": doc["content"]
            }
        })

    index.upsert(vectors=vectors)
    rag_system["documents"].extend(documents)
    print(f"Uploaded {len(vectors)} chunks to Pinecone from {file_path}.")



def query_rag(rag_system, query, top_k=3, temperature=0.2, max_tokens=500):
    client = rag_system["client"]
    index = rag_system["index"]

    # Embed query
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    # Query Pinecone
    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)

    context_chunks = [match["metadata"]["content"] for match in results["matches"]]
    context_text = "\n\n".join(context_chunks)

    # Generate crisp RAG answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a concise assistant. Use only the provided context to answer clearly and briefly. "
                "Avoid fluff. Answer in 1–3 sentences with key facts."
            )},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer briefly:"}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()



