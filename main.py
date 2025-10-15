from rag import (
    initialize_rag,
    add_documents_to_pinecone,
    query_rag
)
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def simple_rag_example():
    """Simple example of using the RAG system"""
    
    # Initialize
    rag_system = initialize_rag(
        openai_api_key=OPENAI_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        mistral_api_key=MISTRAL_API_KEY,
        index_name="my-rag-index"
    )
    # # Add documents
    add_documents_to_pinecone(rag_system, "كراسة الشروط والموصفات.pdf")
    
    # Ask questions
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        answer = query_rag(rag_system, question)
        print(f"\nAnswer: {answer}")

# Run the simple example
simple_rag_example()