# Architecture — Wholesale Customer Segmentation MLOps

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                        DÉVELOPPEMENT                            │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────────┐  │
│  │ JupyterLab│    │  Airflow │    │  FastAPI │   │  Gradio  │  │
│  │  :8888   │    │  :8081   │    │  :8000   │   │  :7860   │  │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘   └────┬─────┘  │
│       │               │               │               │         │
│       └───────────────┴───────────────┴───────┬───────┘         │
│                                               │                 │
│                              ┌────────────────▼──────────────┐  │
│                              │           MLflow              │  │
│                              │   https://mlflow.becaert.com  │  │
│                              │  (tracking + model registry)  │  │
│                              └────────────────┬──────────────┘  │
│                                               │                 │
│                              ┌────────────────▼──────────────┐  │
│                              │            MinIO              │  │
│                              │   S3-compatible  :9000/:9001  │  │
│                              │   bucket: mlflow-artifacts    │  │
│                              └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Composants

### Données
- **Dataset** : Wholesale Customers UCI #292 — 440 clients, 8 features (dépenses annuelles par catégorie)
- **Chargement** : `src/data.py` — téléchargement direct depuis l'URL UCI via `pd.read_csv`
- **Preprocessing** : encodage de `Channel` (one-hot, drop_first), standardisation via `StandardScaler`

### Modèle
- **Algorithme** : K-Means (`sklearn.cluster.KMeans`)
- **Paramètres** : `n_clusters=3` (défaut), `n_init=20`, `random_state=42`
- **Métriques** : inertie, silhouette score
- **Fichier** : `src/train.py`

### MLflow (serveur du prof)
- **URL** : `https://mlflow.becaert.com`
- **Expérience** : `wholesale_segmentation`
- **Artefacts loggés** : modèle K-Means (`kmeans_model/`), scaler (`scaler.joblib`), signature, input_example
- **Config** : `MLFLOW_TRACKING_URI` dans `.env`, lu via `load_dotenv()`

### MinIO (stockage artéfacts)
- **Rôle** : backend S3-compatible pour les artefacts MLflow
- **Ports** : `9000` (API S3), `9001` (console web)
- **Bucket** : `mlflow-artifacts`
- **Credentials** : `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` dans `.env`

### API — FastAPI (`api/main.py`)
- **Port** : `8000`
- **Endpoints** :
  - `GET /health` — statut + état du modèle chargé
  - `GET /model` — run_id MLflow du modèle en service
  - `POST /predict` — reçoit les features client, retourne le cluster
- **Démarrage** : le modèle est chargé **une fois** au démarrage via `lifespan` (depuis MLflow)
- **Validation** : Pydantic `CustomerFeatures` valide automatiquement les entrées

### WebApp — Gradio (`webapp/app.py`)
- **Port** : `7860`
- **Rôle** : interface utilisateur qui appelle `POST /predict` de l'API
- **Config** : `API_URL` dans `.env`

### Airflow (`airflow_dev/dags/`)
- **Mode** : `airflow standalone` (dev)
- **Port** : `8081`
- **DAG** : `training_pipeline.py` — orchestre le téléchargement des données + entraînement + logging MLflow

## Flux de données

```
UCI Dataset
    │
    ▼
src/data.py (load_data + preprocess)
    │
    ▼
src/train.py ──────────────────────────► MLflow (métriques + modèle + scaler)
    │                                         │
    │                                         ▼
    │                                      MinIO (artefacts S3)
    │
    ▼ (au démarrage API)
src/predict.py (load_model depuis MLflow)
    │
    ▼
api/main.py POST /predict
    │
    ▼
webapp/app.py (Gradio → appel HTTP → cluster)
```

## Stack technique

| Couche | Technologie |
|---|---|
| Langage | Python 3.12 |
| Package manager | uv + hatchling |
| ML | scikit-learn (K-Means) |
| Tracking | MLflow ≥ 2.10, < 3.0 |
| Stockage | MinIO (S3-compatible) |
| API | FastAPI + Uvicorn |
| WebApp | Gradio |
| Orchestration | Apache Airflow 2.10 |
| Conteneurs (dev) | Docker Compose |
| Conteneurs (prod) | Kubernetes |
| CI/CD | GitHub Actions |
| Tests | pytest |

## Environnements

### Dev — Docker Compose
Tous les services dans `docker-compose.yml` avec variables injectées depuis `.env`.

### Prod — Kubernetes
Manifestes dans `k8s/` :
- `api-deployment.yaml` + `api-service.yaml`
- `webapp-deployment.yaml` + `webapp-service.yaml`
