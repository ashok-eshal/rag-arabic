import requests

def upload_pdf(file_path):
    """Upload a PDF file to the RAG system"""
    print(f"\n Uploading PDF: {file_path}")
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post("http://localhost:8000/upload", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"{result['message']}")
        return result
    else:
        print(f"Error: {response.json()['detail']}")
        return None


def ask_question(question, temperature=0.0, max_tokens=400):
    """Ask a question to the RAG system"""
    print("Answer: ", end="")
    
    response = requests.post(
        "http://localhost:8000/ask",
        json={
            "question": question,
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        stream=True
    )
    
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            print(chunk.decode("utf-8"), end="")
    print("\n")


def list_files():
    """List all uploaded files"""
    print("\nUploaded files:")
    response = requests.get("http://localhost:8000/files")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total files: {result['count']}")
        for file in result['files']:
            print(f"  - {file}")
    return response.json()


if __name__ == "__main__":

    upload_pdf("كراسة الشروط والموصفات.pdf")
    
    list_files()
    
    ask_question(
        "Hi",
        temperature=0.0,
        max_tokens=400
    )
    