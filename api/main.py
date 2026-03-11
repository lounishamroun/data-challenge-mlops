from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import load_model, predict

load_dotenv()

_model = None
_scaler = None
_run_id = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler, _run_id
    _model, _scaler, _run_id = load_model()
    yield


app = FastAPI(title="Wholesale Clustering API", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    Fresh: float
    Milk: float
    Grocery: float
    Frozen: float
    Detergents_Paper: float
    Delicassen: float
    Channel: int  # 1 = Horeca, 2 = Retail


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model")
def get_model_info():
    return {"run_id": _run_id, "experiment": "wholesale_segmentation"}


@app.post("/predict")
def predict_cluster(customer: CustomerFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    cluster = predict(_model, _scaler, customer.model_dump())
    return {"cluster": cluster}
