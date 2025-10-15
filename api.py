from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag import initialize_rag, add_documents_to_pinecone, query_rag
from dotenv import load_dotenv
import os
import shutil
from pathlib import Path

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize RAG system once
rag_system = initialize_rag(
    openai_api_key=OPENAI_API_KEY,
    pinecone_api_key=PINECONE_API_KEY,
    mistral_api_key=MISTRAL_API_KEY,
    index_name="my-rag-index"
)


def stream_rag_response(rag_system, query, top_k=3, temperature=0.2, max_tokens=500):
    client = rag_system["client"]
    index = rag_system["index"]

    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    context_chunks = [match["metadata"]["content"] for match in results["matches"]]
    context_text = "\n\n".join(context_chunks)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a concise assistant. Use only the provided context to answer clearly and briefly. "
                "Avoid fluff. Answer in 1–3 sentences with key facts."
            )},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer briefly:"}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    def generate():
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return generate()


# Request schemas
class QuestionRequest(BaseModel):
    question: str
    temperature: float = 0.2
    max_tokens: int = 500


# Endpoint to query RAG
@app.post("/ask")
async def stream_answer(request: QuestionRequest):
    generator = stream_rag_response(
        rag_system,
        query=request.question,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    return StreamingResponse(generator, media_type="text/plain")


# Endpoint to upload and process PDF
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file to process with OCR and add to the RAG system.
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF and add to Pinecone
        add_documents_to_pinecone(rag_system, str(file_path))
        
        return {
            "status": "success",
            "message": f"PDF '{file.filename}' processed successfully",
            "filename": file.filename,
            "saved_path": str(file_path)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        await file.close()


# Optional: Endpoint to list uploaded files
@app.get("/files")
async def list_files():
    """
    List all uploaded PDF files.
    """
    files = list(UPLOAD_DIR.glob("*.pdf"))
    return {
        "count": len(files),
        "files": [f.name for f in files]
    }