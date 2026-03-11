from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Wholesale Clustering API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/webapp")
def read_root():
    pass  # TO DO : return the webapp

@app.get("/model")
def get_model():
    pass
