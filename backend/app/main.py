from fastapi import FastAPI

app = FastAPI(title="COMP902 Project API")


@app.get("/")
def root():
    return {"message": "COMP902 API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}