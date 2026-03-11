"""API FastAPI pour l'inférence de segmentation client."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import load_model, predict

load_dotenv()

# Variables globales : modèle chargé au démarrage
_model = None
_scaler = None
_run_id = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage. Démarre sans modèle si aucun n'est disponible."""
    global _model, _scaler, _run_id
    try:
        _model, _scaler, _run_id = load_model()
    except Exception as e:
        print(f"[AVERTISSEMENT] Impossible de charger le modèle : {e}")
        print("[AVERTISSEMENT] L'API démarre sans modèle. Entraînez un modèle d'abord.")
    yield


app = FastAPI(title="Wholesale Clustering API", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    """Schéma des features client attendues par /predict."""
    Fresh: float
    Milk: float
    Grocery: float
    Frozen: float
    Detergents_Paper: float
    Delicassen: float
    Channel: int  # 1 = Hôtellerie/Restauration, 2 = Retail


@app.get("/health")
def health():
    """Vérification de l'état de l'API et du modèle."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info")
def get_model_info():
    """Informations sur le modèle chargé (run_id, nombre de clusters)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Aucun modèle chargé")
    return {
        "run_id": _run_id,
        "experiment": "wholesale_segmentation",
        "n_clusters": int(_model.n_clusters),
    }


@app.post("/predict")
def predict_cluster(customer: CustomerFeatures):
    """Prédit le segment client à partir des dépenses."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Aucun modèle chargé. Entraînez un modèle d'abord.")
    cluster = predict(_model, _scaler, customer.model_dump())
    return {"cluster": int(cluster)}
