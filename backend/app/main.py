import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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