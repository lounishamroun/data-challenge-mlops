"""
Ce script entraîne un modèle K-Means pour segmenter les clients d'un grossiste.
Il utilise MLflow pour suivre les expériences, les paramètres et les métriques.
"""

import argparse
from datetime import datetime
import joblib
import mlflow
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlflow.models import infer_signature

load_dotenv()

from src.data import load_data, preprocess

# Valeurs par défaut
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
DEFAULT_K = 3
MLFLOW_EXPERIMENT = "wholesale_segmentation"


def train(n_clusters: int = DEFAULT_K) -> dict:
    """
    Entraîne un modèle K-Means sur le dataset. Les résultats sont loggés dans MLflow. 
    
    Args:
        n_clusters: Nombre de clusters K pour K-Means.

    Returns:
        Dict avec run_id, silhouette_score et inertia.
    """

    # Chargement des données
    df_raw = load_data(DATA_URL)
    df_scaled, scaler = preprocess(df_raw)

    # Modèle K-Means
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = model.fit_predict(df_scaled)

    # Utilisation des métriques d'inertie et silhouette score pour évaluer les modèles
    inertia = model.inertia_
    silhouette = silhouette_score(df_scaled, labels)

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_name = f"kmeans_k{n_clusters}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Paramètres
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("n_samples", len(df_scaled))
        mlflow.log_param("features", list(df_scaled.columns))

        # Métriques
        mlflow.log_metric("inertia", inertia)
        mlflow.log_metric("silhouette_score", silhouette)

        # Artefacts : modèle + scaler
        signature = infer_signature(df_scaled, labels)
        mlflow.sklearn.log_model(model, artifact_path="kmeans_model", signature=signature, input_example=df_scaled.head(1))
        joblib.dump(scaler, "scaler.joblib")
        mlflow.log_artifact("scaler.joblib")

    return {
        "run_id": run.info.run_id,
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "inertia": inertia,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train K-Means segmentation")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Nombre de clusters")
    args = parser.parse_args()
    train(n_clusters=args.k)