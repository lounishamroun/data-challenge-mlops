# Wholesale Customer Segmentation — MLOps POC

[![CI](https://github.com/lounishamroun/data-challenge-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/lounishamroun/data-challenge-mlops/actions/workflows/ci.yml)

End-to-end MLOps pipeline for wholesale customer segmentation using K-Means clustering on the [UCI Wholesale Customers dataset](https://archive.ics.uci.edu/dataset/292/wholesale+customers).

**Stack:** scikit-learn · MLflow · FastAPI · Gradio · Airflow · Docker Compose · Kubernetes · GitHub Actions

---

## Prerequisites

- Docker & Docker Compose
- [uv](https://docs.astral.sh/uv/) — `curl -Ls https://astral.sh/uv/install.sh | sh`
- Python 3.12+
- kubectl + Minikube or Docker Desktop Kubernetes (for production deployment)

---

## Local development (Docker Compose)

```bash
# 1. Clone
git clone https://github.com/lounishamroun/data-challenge-mlops.git
cd data-challenge-mlops

# 2. Set up environment variables
cp .env.example .env   # fill in MinIO credentials and MLflow URI

# 3. Install dependencies (for local dev outside Docker)
uv pip install -e ".[dev]"

# 4. Start all services
docker compose up --build
```

| Service | URL |
|---|---|
| MinIO console | http://localhost:9001 |
| JupyterLab | http://localhost:8888 |
| Airflow | http://localhost:8081 |
| FastAPI docs | http://localhost:8000/docs |
| WebApp (Gradio) | http://localhost:7860 |

---

## Training the model

```bash
# Locally
python -m src.train --k 3

# Via Docker Compose
docker compose run --rm api python -m src.train --k 3
```

Each run is logged to MLflow (experiment `wholesale_segmentation`) with:
- parameters: `n_clusters`, `n_samples`, `features`
- metrics: `inertia`, `silhouette_score`
- artifacts: fitted K-Means model + StandardScaler

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + model status |
| `GET` | `/model-info` | Active run ID, experiment, n_clusters |
| `POST` | `/predict` | Returns cluster for a customer |

Example predict request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Fresh":12669,"Milk":9656,"Grocery":7561,"Frozen":214,"Detergents_Paper":2674,"Delicassen":1338,"Channel":1}'
```

---

## Running tests

```bash
pytest tests/ -v --disable-warnings
```

---

## Kubernetes deployment (local production)

```bash
# 1. Point Docker to Minikube (skip for Docker Desktop)
eval $(minikube docker-env)

# 2. Build images
docker build -f Dockerfile.api -t wholesale-clustering-api:latest .
docker build -f Dockerfile.webapp -t wholesale-clustering-webapp:latest .

# 3. Apply manifests
kubectl apply -f k8s/namespace.yaml
cp k8s/secrets.example.yaml k8s/secrets.yaml  # fill in base64 values
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/api-deployment.yaml -f k8s/api-service.yaml
kubectl apply -f k8s/webapp-deployment.yaml -f k8s/webapp-service.yaml

# 4. Access
minikube service webapp-service -n wholesale-mlops   # WebApp
kubectl port-forward svc/api-service 8000:8000 -n wholesale-mlops  # API
```

> **Docker Desktop:** skip `eval $(minikube docker-env)` and access WebApp at http://localhost:30860 directly.

---

## CI/CD (GitHub Actions)

Four jobs run on every push to `main` and `feat/**` branches:

| Job | What it does |
|---|---|
| `lint` | `ruff check` on all source code |
| `test` | `pytest` full test suite |
| `k8s-validate` | `kubectl apply --dry-run=client` on all manifests |
| `docker-build` | Builds API and WebApp Docker images |

---

## Project structure

```
├── src/
│   ├── data.py           # Data loading & preprocessing
│   ├── train.py          # K-Means training + MLflow logging
│   └── predict.py        # Model loading & cluster prediction
├── api/
│   └── main.py           # FastAPI — /health, /model-info, /predict
├── webapp/
│   └── app.py            # Gradio inference UI
├── airflow_dev/
│   └── dags/             # Retraining pipeline DAG
├── k8s/                  # Kubernetes manifests
│   ├── namespace.yaml
│   ├── secrets.example.yaml
│   ├── api-deployment.yaml / api-service.yaml
│   └── webapp-deployment.yaml / webapp-service.yaml
├── tests/                # Unit & integration tests
├── .github/workflows/    # GitHub Actions CI
├── docker-compose.yml    # Dev stack
├── pyproject.toml        # Dependencies (uv + hatchling)
└── ARCHITECTURE.md       # Full architecture description
```

---

## Team

Alexandre Mathias Donnat · Lounis Hamroun · Leo Ivars · Omar Fekih-Hassen · Anne Faury
