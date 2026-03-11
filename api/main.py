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
    try:
        _model, _scaler, _run_id = load_model()
    except Exception as e:
        print(f"[WARNING] Could not load model at startup: {e}")
        print("[WARNING] API started without a model. Train a model first.")
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


@app.get("/model-info")
def get_model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return {
        "run_id": _run_id,
        "experiment": "wholesale_segmentation",
        "n_clusters": int(_model.n_clusters),
    }


@app.post("/predict")
def predict_cluster(customer: CustomerFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Train a model first.")
    cluster = predict(_model, _scaler, customer.model_dump())
    return {"cluster": int(cluster)}
