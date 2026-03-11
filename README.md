# Wholesale Customer Segmentation — MLOps POC

POC MLOps de segmentation de clients grossistes (dataset UCI #292) avec K-Means, MLflow, Airflow, FastAPI et Gradio.

## Prérequis

- Docker & Docker Compose
- [uv](https://docs.astral.sh/uv/) (`curl -Ls https://astral.sh/uv/install.sh | sh`)
- Python 3.12+

## Installation locale

```bash
# Cloner le repo
git clone https://github.com/lounishamroun/data-challenge-mlops.git
cd data-challenge-mlops

# Créer le .env à partir du template
cp .env .env.local  # puis éditer les valeurs

# Installer les dépendances
uv pip install -e ".[dev]"
```

Le `.env` contient les credentials MinIO, l'URI MLflow et la config API.  
Récupère les valeurs auprès de l'équipe et remplis les champs `à_modifier`.

## Lancer l'environnement dev (Docker Compose)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| MinIO console | http://localhost:9001 |
| JupyterLab | http://localhost:8888 |
| Airflow | http://localhost:8081 |
| API (FastAPI) | http://localhost:8000/docs |
| WebApp (Gradio) | http://localhost:7860 |

## Entraîner le modèle

```bash
# En local
python -m src.train --k 3

# Via Docker
docker compose run --rm api python -m src.train --k 3
```

Le run est automatiquement loggé sur `https://mlflow.becaert.com` (expérience `wholesale_segmentation`).

## Lancer les tests

```bash
python -m pytest tests/ -v
```

## Structure du projet

```
├── src/                  # Code ML
│   ├── data.py           # Chargement & preprocessing
│   ├── train.py          # Entraînement K-Means + MLflow
│   └── predict.py        # Chargement modèle & prédiction
├── api/
│   └── main.py           # FastAPI : /health /model /predict
├── webapp/
│   └── app.py            # Interface Gradio
├── airflow_dev/
│   └── dags/             # DAG de la pipeline d'entraînement
├── k8s/                  # Manifestes Kubernetes
├── tests/                # Tests unitaires
├── docker-compose.yml    # Stack de développement
├── pyproject.toml        # Dépendances (uv + hatchling)
└── ARCHITECTURE.md       # Détail de l'architecture
```

## Équipe

Alexandre Mathias Donnat · Lounis Hamroun · Leo Ivars · Omar Fekih-Hassen · Anne Faury
