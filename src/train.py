"""Entraînement K-Means avec suivi MLflow."""

import argparse
from datetime import datetime
import joblib
import mlflow
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlflow.models import infer_signature

load_dotenv()

from src.data import load_data, preprocess  # noqa: E402

# Configuration par défaut
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
DEFAULT_K = 3
MLFLOW_EXPERIMENT = "wholesale_segmentation"


def train(n_clusters: int = DEFAULT_K) -> dict:
    """Entraîne un K-Means et logue les résultats dans MLflow.

    Retourne un dict avec run_id, n_clusters, silhouette_score, inertia.
    """
    # Chargement et prétraitement
    df_raw = load_data(DATA_URL)
    df_scaled, scaler = preprocess(df_raw)

    # Entraînement K-Means
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = model.fit_predict(df_scaled)

    # Métriques d'évaluation
    inertia = model.inertia_
    silhouette = silhouette_score(df_scaled, labels)

    # Logging MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_name = f"kmeans_k{n_clusters}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("n_samples", len(df_scaled))
        mlflow.log_param("features", list(df_scaled.columns))

        mlflow.log_metric("inertia", inertia)
        mlflow.log_metric("silhouette_score", silhouette)

        # Sauvegarde du modèle et du scaler
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
    parser = argparse.ArgumentParser(description="Entraînement K-Means pour la segmentation client")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Nombre de clusters")
    args = parser.parse_args()
    train(n_clusters=args.k)