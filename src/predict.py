"""
Chargement du modèle K-Means et du scaler depuis MLflow et prédiction de cluster.
"""

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.data import prepare_features
from src.train import MLFLOW_EXPERIMENT


def load_model(run_id: str | None = None, experiment_name: str = MLFLOW_EXPERIMENT):
    """
    Charge le modèle K-Means et le scaler depuis MLflow.

    Priorité :
      1. run_id explicite
      2. Dernière version du Model Registry (wholesale_kmeans_best)
      3. Dernier run de l'expérience

    Returns:
        Tuple (model, scaler, run_id).
    """
    if run_id is None:
        # Tente de charger depuis le Model Registry (meilleur modèle sélectionné par le DAG)
        try:
            model = mlflow.sklearn.load_model("models:/wholesale_kmeans_best/latest")
            # Récupérer le run_id depuis le registry
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions("wholesale_kmeans_best")
            run_id = versions[0].run_id
            scaler_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/scaler.joblib")
            scaler = joblib.load(scaler_path)
            return model, scaler, run_id
        except Exception:
            pass

        # Fallback : dernier run de l'expérience
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Expérience '{experiment_name}' introuvable. Lancez d'abord train.py.")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise ValueError("Aucun run trouvé dans l'expérience.")
        run_id = runs.iloc[0].run_id

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/kmeans_model")
    scaler_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/scaler.joblib")
    scaler = joblib.load(scaler_path)
    return model, scaler, run_id


def predict(model, scaler, features: dict[str, float]) -> int:
    """
    Prédit le cluster pour un client donné.

    Args:
        model: Modèle K-Means chargé depuis MLflow.
        scaler: StandardScaler fité lors de l'entraînement.
        features: Dictionnaire avec les colonnes numériques + catégorielles.

    Returns:
        Cluster prédit (int).
    """
    df = pd.DataFrame([features])
<<<<<<< HEAD
    df_features = prepare_features(df)
    df_scaled = pd.DataFrame(scaler.transform(df_features), columns=df_features.columns)
=======
    df_encoded = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True).astype(float)
    df_final = pd.concat([df[NUMERIC_COLS], df_encoded], axis=1)
    df_scaled = pd.DataFrame(scaler.transform(df_final), columns=df_final.columns)
>>>>>>> ea808f8 (fix: remove predict.py from feat/kube (managed by colleague))
    return int(model.predict(df_scaled)[0])
