import io
import requests
import re

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text: str):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

app = FastAPI(title="COMP902 Project API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "COMP902 API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/llm-test")
def llm_test():
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": "Explain supervised learning in one sentence.",
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    return {
        "response": data.get("response", "").strip()
    }

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    contents = await file.read()

    try:
        reader = PdfReader(io.BytesIO(contents))

        page_chunks = []
        total_characters = 0

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()

            if not page_text:
                continue

            cleaned_page_text = clean_text(page_text)
            total_characters += len(cleaned_page_text)

            chunks = chunk_text(cleaned_page_text)

            for chunk_index, chunk in enumerate(chunks, start=1):
                page_chunks.append({
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk
                })

        if page_chunks:
            first_embedding = embedding_model.encode(
                page_chunks[0]["text"]
            ).tolist()
        else:
            first_embedding = []

        return {
            "filename": file.filename,
            "pages": len(reader.pages),
            "characters": total_characters,
            "chunk_count": len(page_chunks),
            "first_chunk": page_chunks[0] if page_chunks else None,
            "embedding_length": len(first_embedding),
            "embedding_preview": first_embedding[:5]
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to process this PDF."
        )