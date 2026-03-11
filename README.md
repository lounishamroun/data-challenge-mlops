# Wholesale Customer Segmentation — MLOps POC

[![CI](https://github.com/lounishamroun/data-challenge-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/lounishamroun/data-challenge-mlops/actions/workflows/ci.yml)

Pipeline MLOps complète pour la segmentation de clients grossistes par K-Means sur le [UCI Wholesale Customers dataset](https://archive.ics.uci.edu/dataset/292/wholesale+customers) (440 échantillons, 8 features).

**Stack :** scikit-learn · MLflow · FastAPI · Gradio · Airflow · Docker Compose · Kubernetes · GitHub Actions

---

## Architecture

```
                    ┌───────────────────────────────┐
                    │  MLflow (mlflow.becaert.com)   │
                    │  tracking + model registry     │
                    └────────────┬──────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
  ┌───────▼────────┐    ┌───────▼────────┐    ┌────────▼───────┐
  │    Airflow      │    │   FastAPI       │    │    Gradio      │
  │  (orchestration │    │   (API REST)    │    │   (WebApp)     │
  │   DAG k=2..10)  │    │  /predict       │    │  2 onglets :   │
  └───────┬────────┘    └───────┬────────┘    │  - Prédiction  │
          │                      │             │  - Analyse     │
          │                      │             └────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
  ┌──────────────┐      ┌──────────────┐       API_URL/predict
  │   MinIO (S3)  │      │  src/predict  │
  │   artifacts   │      │  load_model() │
  └──────────────┘      └──────────────┘
```

| Environnement | Composants |
|---|---|
| **Dev** (Docker Compose) | MinIO · Airflow · FastAPI · Gradio · JupyterLab |
| **Prod** (Kubernetes) | FastAPI (ClusterIP) · Gradio (NodePort :30860) |

---

## Prérequis

- Docker & Docker Compose
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -Ls https://astral.sh/uv/install.sh | sh`
- kubectl + Minikube *(optionnel, pour le déploiement Kubernetes)*

---

## Démarrage rapide

```bash
# 1. Cloner le repo
git clone https://github.com/lounishamroun/data-challenge-mlops.git
cd data-challenge-mlops

# 2. Configurer l'environnement
cp .env.exemple .env   # remplir les credentials MinIO et l'URI MLflow

# 3. Installer les dépendances (dev local hors Docker)
uv pip install -e ".[dev]"

# 4. Lancer tous les services
docker compose up --build
```

| Service | URL |
|---|---|
| MinIO console | http://localhost:9001 |
| Airflow | http://localhost:8081 |
| FastAPI (Swagger) | http://localhost:8000/docs |
| WebApp (Gradio) | http://localhost:7860 |

---

## Entraînement

### En local

```bash
python -m src.train --k 3
```

### Via Airflow (recommandé)

Le DAG `wholesale_training_pipeline` entraîne K-Means pour k=2 à 10 en parallèle, sélectionne le meilleur modèle (silhouette score max) et l'enregistre dans le MLflow Model Registry (`wholesale_kmeans_best`).

```
fetch_data → [train_k2, train_k3, ..., train_k10] → select_best_model
```

Chaque run MLflow enregistre :
- **Paramètres :** `n_clusters`, `n_samples`, `features`
- **Métriques :** `inertia`, `silhouette_score`
- **Artefacts :** modèle K-Means + StandardScaler

---

## API REST (FastAPI)

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + statut du modèle |
| `GET` | `/model-info` | Run ID, expérience, nombre de clusters |
| `POST` | `/predict` | Retourne le cluster d'un client |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Fresh":12669,"Milk":9656,"Grocery":7561,"Frozen":214,"Detergents_Paper":2674,"Delicassen":1338,"Channel":1}'
```

Le modèle est chargé au démarrage avec 3 niveaux de priorité :
1. Run ID explicite (variable d'environnement)
2. Model Registry (`wholesale_kmeans_best/latest`)
3. Dernier run de l'expérience par date

---

## WebApp (Gradio)

Deux onglets :

- **Prédiction** — 6 sliders (dépenses par catégorie) + canal → appel API → cluster prédit
- **Interprétation** — Choisir k (2–10) puis visualiser :
  - Méthode du coude + Silhouette Score
  - Projection PCA 2D (clusters K-Means vs Region réelle)
  - Heatmap des centres de clusters (valeurs scalées)
  - Répartition clusters vs Region
  - Matrice de contingence Region × Cluster

---

## Tests

```bash
pytest tests/ -v --disable-warnings
```

34 tests couvrant :

| Module | Tests |
|---|---|
| `test_data.py` | Chargement, preprocessing, scaling, colonnes |
| `test_train.py` | Logging MLflow, différents k, valeur par défaut |
| `test_predict.py` | Chargement modèle, prédiction, déterminisme |
| `test_api.py` | Endpoints health, model-info, predict, validation |
| `test_model.py` | K-Means fit, silhouette, inertie, encodage Channel |

---

## Déploiement Kubernetes

```bash
# 1. Pointer Docker vers Minikube
eval $(minikube docker-env)

# 2. Builder les images
docker build -f Dockerfile.api -t wholesale-clustering-api:latest .
docker build -f Dockerfile.webapp -t wholesale-clustering-webapp:latest .

# 3. Déployer
kubectl apply -f k8s/namespace.yaml
cp k8s/secrets.example.yaml k8s/secrets.yaml   # remplir les valeurs base64
kubectl apply -f k8s/
```

| Ressource | Type | Accès |
|---|---|---|
| `api-service` | ClusterIP | Interne au cluster (:8000) |
| `webapp-service` | NodePort | Externe (:30860) |

```bash
# Accéder à la webapp via Minikube
minikube service webapp-service -n wholesale-mlops
```

---

## CI/CD (GitHub Actions)

Quatre jobs parallèles sur chaque push (`main`, `feat/**`) :

| Job | Description |
|---|---|
| **Lint** | `ruff check` sur src/, api/, webapp/, tests/ |
| **Test** | `pytest` avec couverture (--cov=src --cov=api) |
| **K8s Validate** | `kubectl apply --dry-run=client` sur tous les manifestes |
| **Docker Build** | Build des images API et WebApp |

---

## Structure du projet

```
├── src/
│   ├── data.py              # Chargement données + preprocessing + prepare_features
│   ├── train.py             # Entraînement K-Means + logging MLflow
│   ├── predict.py           # Chargement modèle (registry/run) + prédiction
│   └── utils.py
├── api/
│   └── main.py              # FastAPI — /health, /model-info, /predict
├── webapp/
│   └── app.py               # Gradio — onglets Prédiction + Interprétation
├── airflow_dev/
│   └── dags/
│       └── training_pipeline.py  # DAG : fetch → train k=2..10 → select best
├── notebooks/
│   └── visualisation_clustering.ipynb
├── tests/                   # 34 tests (pytest)
├── k8s/                     # Manifestes Kubernetes (namespace, deployments, services, secrets)
├── .github/workflows/
│   └── ci.yml               # CI : lint, test, k8s validate, docker build
├── docker-compose.yml       # Stack dev : MinIO, Airflow, API, WebApp
├── Dockerfile.airflow       # Image Airflow avec src/ et dags/
├── Dockerfile.api           # Image FastAPI
├── Dockerfile.webapp        # Image Gradio
├── pyproject.toml           # Dépendances (hatchling + uv)
└── ARCHITECTURE.md          # Description détaillée de l'architecture
```

---

## Variables d'environnement

```env
# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_BUCKET=mlflow-artifacts

# S3 / MLflow (partagées par tous les services)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
MLFLOW_TRACKING_URI=https://mlflow.becaert.com
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts/

# WebApp
API_URL=http://api:8000
```

---

## Équipe

Alexandre Mathias Donnat · Lounis Hamroun · Leo Ivars · Omar Fekih-Hassen · Anne Faury
