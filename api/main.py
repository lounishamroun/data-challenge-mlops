from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

@app.get("/webapp")
def read_root():
    pass # TO DO : return the webapp

@app.get("/model")
def read_root():
    pass

