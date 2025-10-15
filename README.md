# RAG-OCR

A FastAPI-based RAG (Retrieval-Augmented Generation) system that processes PDF documents using Mistral OCR for text extraction and provides intelligent question-answering capabilities.

## Features

- 📄 PDF text extraction using Mistral OCR (state-of-the-art OCR model)
- 🔍 Vector-based semantic search with Pinecone
- 💬 Streaming response API for real-time answers
- 🚀 FastAPI backend with automatic API documentation
- 🐳 Docker support for easy deployment
- 🌍 Excellent support for Arabic and multilingual documents

## Prerequisites

- Python 3.11+
- OpenAI API key (for embeddings and answer generation)
- Mistral API key (for OCR text extraction)
- Pinecone API key (for vector storage)
- Docker (optional, for containerized deployment)

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-vision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Running the Application

### Option 1: Run Locally

```bash
uvicorn api:app --reload --port 8000
```

### Option 2: Run with Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t rag-vision .
docker run -p 8000:8000 --env-file .env rag-vision
```

The API will be available at `http://localhost:8000`

## API Usage

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Query Endpoint

**POST** `/ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Your question here",
    "temperature": 0.2,
    "max_tokens": 500
  }'
```

**Request Body:**
```json
{
  "question": "Your question here",
  "temperature": 0.2,
  "max_tokens": 500
}
```

**Response:**
Streaming text response with the answer based on the document context.

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "What is this document about?",
        "temperature": 0.0,
        "max_tokens": 400
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        print(chunk.decode("utf-8"), end="")
```

## Project Structure

```
rag-ocr/
├── api.py                  # FastAPI application
├── rag.py                  # RAG core functionality
├── app.py                  # Client example
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
└── rag_storage/           # Storage for processed documents
    └── texts/             # Extracted text files per page
```

## How It Works

1. **Document Processing**: PDFs are processed using Mistral OCR to extract text with high accuracy
2. **Text Chunking**: Extracted text is split into overlapping chunks for better retrieval
3. **Vector Embedding**: Text chunks are embedded using OpenAI's text-embedding-3-small model
4. **Storage**: Embeddings are stored in Pinecone for fast similarity search
5. **Query Processing**: User questions are embedded and matched against stored chunks
6. **Answer Generation**: Relevant context is passed to GPT-4o-mini to generate concise answers

## Why Mistral OCR?

Mistral OCR offers several advantages:
- **Superior Accuracy**: State-of-the-art OCR model with excellent accuracy for complex documents
- **Multilingual Support**: Excellent support for Arabic, English, and many other languages
- **Structured Output**: Preserves document structure and formatting
- **Efficient Processing**: Processes entire PDFs in a single API call



