# RAG-OCR

Smart document Q&A system using Mistral OCR and RAG. Upload PDFs, ask questions in natural language.

## 🚀 Quick Start

**1. Setup**
```bash
pip install -r requirements.txt
```

**2. Configure API Keys** (create `.env` file)
```bash
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
PINECONE_API_KEY=...
```

**3. Run**
```bash
# Local
uvicorn api:app --reload

# Docker
docker-compose up --build
```

Access API at `http://localhost:8000` | Docs at `http://localhost:8000/docs`

## 📡 API Endpoints

### Upload PDF
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

### Ask Questions
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'
```

### List Files
```bash
curl http://localhost:8000/files
```

## 💻 Python Client

```python
import requests

# Upload
files = {'file': open('document.pdf', 'rb')}
requests.post('http://localhost:8000/upload', files=files)

# Ask
response = requests.post(
    'http://localhost:8000/ask',
    json={"question": "What are the main points?"},
    stream=True
)

for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        print(chunk.decode("utf-8"), end="")
```

## 🏗️ Architecture

```
PDF → Mistral OCR → Text Chunks → OpenAI Embeddings → Pinecone (Vector DB)
                                                              ↓
User Question → OpenAI Embedding → Similarity Search → GPT-4o-mini → Answer
```

**Tech Stack:**
- 🔍 **Mistral OCR**: Superior accuracy, multilingual support (Arabic, English, etc.)
- 📊 **Pinecone**: Cloud vector database (no local storage needed)
- 🤖 **OpenAI**: Embeddings (text-embedding-3-small) + LLM (GPT-4o-mini)
- ⚡ **FastAPI**: Modern async API with streaming responses

## 📁 Project Structure

```
rag-ocr/
├── api.py          # FastAPI endpoints
├── rag.py          # Core RAG logic
├── app.py          # Python client examples
├── main.py         # Standalone script
└── uploads/        # Uploaded PDFs
```

## 🐳 Docker

```bash
docker-compose up --build    # Start
docker-compose logs -f       # View logs
docker-compose down          # Stop
```

## 🌟 Features

✅ Upload PDFs via API  
✅ Automatic OCR with Mistral  
✅ Vector search with Pinecone  
✅ Streaming responses  
✅ Multilingual (Arabic, English, etc.)  
✅ Cloud-native (stateless)  
✅ Docker ready



