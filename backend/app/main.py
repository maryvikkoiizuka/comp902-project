import io
import re
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# -------------------------
# Models and Database
# -------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(path="qdrant_storage")

COLLECTION_NAME = "document_chunks"


# -------------------------
# FastAPI App
# -------------------------

app = FastAPI(title="COMP902 Project API")


# Create Qdrant collection if it does not exist
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )


# -------------------------
# CORS
# -------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Helper Functions
# -------------------------

def clean_text(text: str):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if (
            len(current_chunk)
            + len(sentence)
            + 1
            <= chunk_size
        ):
            if current_chunk:
                current_chunk += " "

            current_chunk += sentence

        else:
            if current_chunk:
                chunks.append(
                    current_chunk.strip()
                )

            overlap_text = (
                current_chunk[-overlap:]
                if current_chunk
                else ""
            )

            current_chunk = (
                overlap_text
                + " "
                + sentence
            ).strip()

    if current_chunk:
        chunks.append(
            current_chunk.strip()
        )

    return chunks


# -------------------------
# Basic Endpoints
# -------------------------

@app.get("/")
def root():
    return {
        "message": "COMP902 API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# -------------------------
# Ollama Test
# -------------------------

@app.get("/llm-test")
def llm_test():
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": (
                "Explain supervised learning "
                "in one sentence."
            ),
            "stream": False,
            "options": {
                "temperature": 0
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return {
        "response": data.get(
            "response",
            ""
        ).strip()
    }


# -------------------------
# PDF Upload
# -------------------------

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    contents = await file.read()

    try:
        reader = PdfReader(
            io.BytesIO(contents)
        )

        page_chunks = []
        total_characters = 0

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            page_text = page.extract_text()

            if not page_text:
                continue

            cleaned_page_text = clean_text(
                page_text
            )

            total_characters += len(
                cleaned_page_text
            )

            chunks = chunk_text(
                cleaned_page_text
            )

            for chunk_index, chunk in enumerate(
                chunks,
                start=1,
            ):
                page_chunks.append(
                    {
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                )

        if not page_chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No readable text was "
                    "found in the PDF."
                ),
            )

        texts = [
            chunk["text"]
            for chunk in page_chunks
        ]

        embeddings = embedding_model.encode(
            texts
        )

        points = []

        for index, chunk in enumerate(
            page_chunks
        ):
            points.append(
                PointStruct(
                    id=index,
                    vector=embeddings[
                        index
                    ].tolist(),
                    payload={
                        "filename": file.filename,
                        "page": chunk["page"],
                        "chunk_index": chunk[
                            "chunk_index"
                        ],
                        "text": chunk["text"],
                    },
                )
            )

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        return {
            "filename": file.filename,
            "pages": len(reader.pages),
            "characters": total_characters,
            "chunk_count": len(page_chunks),
            "stored_in_qdrant": len(points),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to process this PDF: "
                f"{str(e)}"
            ),
        )


# -------------------------
# Vector Search Test
# -------------------------

@app.get("/search")
def search_chunks(
    query: str,
    top_k: int = 3,
):
    query_embedding = embedding_model.encode(
        query
    ).tolist()

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    results = response.points

    return {
        "query": query,
        "top_k": top_k,
        "results": [
            {
                "score": result.score,
                "filename": result.payload.get(
                    "filename"
                ),
                "page": result.payload.get(
                    "page"
                ),
                "chunk_index": result.payload.get(
                    "chunk_index"
                ),
                "text": result.payload.get(
                    "text"
                ),
            }
            for result in results
        ],
    }


# -------------------------
# RAG Ask Endpoint
# -------------------------

@app.get("/ask")
def ask_question(
    query: str
):
    # Internal retrieval settings
    top_k = 3
    minimum_score = 0.30

    # 1. Convert question into embedding
    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # 2. Retrieve top candidate chunks
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    results = response.points

    # 3. Remove weak retrieval matches
    results = [
        result
        for result in results
        if result.score >= minimum_score
    ]

    if not results:
        return {
            "query": query,
            "answer": (
                "The available sources do not "
                "provide enough information to "
                "answer this question."
            ),
            "sources": [],
        }

    # 4. Build context from relevant chunks
    context_parts = []

    for result in results:
        payload = result.payload

        context_parts.append(
            f"""
Source: {payload.get("filename")}
Page: {payload.get("page")}
Content:
{payload.get("text")}
"""
        )

    context = "\n".join(
        context_parts
    )

    # 5. Build RAG prompt
    prompt = f"""
You are answering questions from a provided document.

Use the context below to answer the question.

If the context contains information that answers the question,
give a clear and concise answer based only on that information.

Only say:
"The available sources do not provide enough information to answer this question."

when the context genuinely does not contain the answer.

Do not use outside knowledge.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    # 6. Send retrieved context to Ollama
    try:
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                },
            },
            timeout=120,
        )

        ollama_response.raise_for_status()

    except requests.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to connect to Ollama: "
                f"{str(e)}"
            ),
        )

    data = ollama_response.json()

    answer = data.get(
        "response",
        ""
    ).strip()

    # 7. Return answer and supporting sources
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "filename": result.payload.get(
                    "filename"
                ),
                "page": result.payload.get(
                    "page"
                ),
                "chunk_index": result.payload.get(
                    "chunk_index"
                ),
                "score": result.score,
            }
            for result in results
        ],
    }