"""
DAG Airflow : Pipeline d'entraînement K-Means et sélection du meilleur modèle.

Étapes :
  1. fetch_data        — Vérifie que le dataset UCI est accessible.
  2. train_k{2..10}    — Entraîne un modèle K-Means pour chaque valeur de k (en parallèle).
  3. select_best_model — Compare les silhouette scores et tague le meilleur run MLflow.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Plage de k à explorer ──────────────────────────────────
K_MIN = 2
K_MAX = 10


# ── Fonctions des tâches ───────────────────────────────────

def fetch_data(**kwargs):
    """Télécharge le dataset et vérifie qu'il est exploitable."""
    from src.data import load_data
    from src.train import DATA_URL

    df = load_data(DATA_URL)
    n_rows, n_cols = df.shape
    print(f"Dataset chargé : {n_rows} lignes, {n_cols} colonnes")
    if n_rows == 0:
        raise ValueError("Le dataset est vide.")
    return {"n_rows": n_rows, "n_cols": n_cols}


def train_model(k: int, **kwargs):
    """Entraîne un modèle K-Means pour k clusters et pousse le résultat en XCom."""
    from src.train import train

    result = train(n_clusters=k)
    print(f"k={k}  silhouette={result['silhouette_score']:.4f}  run_id={result['run_id']}")
    return result


def select_best_model(**kwargs):
    """Récupère les résultats de tous les entraînements et tague le meilleur run."""
    import mlflow

    ti = kwargs["ti"]

    # Collecte des résultats via XCom
    results = []
    for k in range(K_MIN, K_MAX + 1):
        result = ti.xcom_pull(task_ids=f"train_k{k}")
        if result is not None:
            results.append(result)

    if not results:
        raise ValueError("Aucun résultat d'entraînement trouvé dans XCom.")

    # Sélection du meilleur modèle (silhouette score le plus élevé)
    best = max(results, key=lambda r: r["silhouette_score"])
    print(
        f"Meilleur modèle : k={best['n_clusters']}  "
        f"silhouette={best['silhouette_score']:.4f}  "
        f"run_id={best['run_id']}"
    )

    # Tag du meilleur run dans MLflow
    client = mlflow.MlflowClient()
    client.set_tag(best["run_id"], "best_model", "true")
    client.set_tag(best["run_id"], "selected_by", "airflow_dag")

    # Enregistrement du modèle dans le Model Registry
    model_uri = f"runs:/{best['run_id']}/kmeans_model"
    mlflow.register_model(model_uri, "wholesale_kmeans_best")
    print(f"Modèle enregistré dans le registry : wholesale_kmeans_best")

    return best


# ── Définition du DAG ──────────────────────────────────────

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="Entraînement K-Means (k=2..10) et sélection du meilleur modèle",
    schedule=None,                       # déclenché manuellement
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "training", "kmeans"],
) as dag:

    # 1. Récupération des données
    task_fetch = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data,
    )

    # 2. Entraînement parallèle pour chaque k
    train_tasks = []
    for k in range(K_MIN, K_MAX + 1):
        task = PythonOperator(
            task_id=f"train_k{k}",
            python_callable=train_model,
            op_kwargs={"k": k},
        )
        train_tasks.append(task)

    # 3. Sélection du meilleur modèle
    task_select = PythonOperator(
        task_id="select_best_model",
        python_callable=select_best_model,
    )

    # Dépendances : fetch → tous les train en parallèle → select
    task_fetch >> train_tasks >> task_select
