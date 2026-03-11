"""Tests unitaires — chargement du modèle et prédiction (src/predict.py)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.predict import load_model, predict


def _make_fake_model_and_scaler():
    """Crée un modèle K-Means et un scaler sur des données synthétiques."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 7)
    model = KMeans(n_clusters=3, n_init=5, random_state=42)
    model.fit(X)
    scaler = StandardScaler()
    scaler.fit(X)
    return model, scaler


class TestPredict:
    def test_returns_int(self):
        model, scaler = _make_fake_model_and_scaler()
        features = {
            "Fresh": 12669, "Milk": 9656, "Grocery": 7561,
            "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338,
            "Channel": 1,
        }
        result = predict(model, scaler, features)
        assert isinstance(result, int)

    def test_cluster_in_range(self):
        model, scaler = _make_fake_model_and_scaler()
        features = {
            "Fresh": 500, "Milk": 1000, "Grocery": 2000,
            "Frozen": 100, "Detergents_Paper": 300, "Delicassen": 50,
            "Channel": 2,
        }
        result = predict(model, scaler, features)
        assert 0 <= result < model.n_clusters

    def test_deterministic(self):
        model, scaler = _make_fake_model_and_scaler()
        features = {
            "Fresh": 12669, "Milk": 9656, "Grocery": 7561,
            "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338,
            "Channel": 1,
        }
        r1 = predict(model, scaler, features)
        r2 = predict(model, scaler, features)
        assert r1 == r2


class TestLoadModel:
    @patch("src.predict.joblib")
    @patch("src.predict.mlflow")
    def test_load_model_with_run_id(self, mock_mlflow, mock_joblib):
        model, scaler = _make_fake_model_and_scaler()
        mock_mlflow.sklearn.load_model.return_value = model
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/scaler.joblib"
        mock_joblib.load.return_value = scaler

        loaded_model, loaded_scaler, run_id = load_model(run_id="abc123")

        mock_mlflow.sklearn.load_model.assert_called_once_with("runs:/abc123/kmeans_model")
        assert run_id == "abc123"
        assert loaded_model is model
        assert loaded_scaler is scaler

    @patch("src.predict.mlflow")
    def test_load_model_no_experiment_raises(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None

        with pytest.raises(ValueError, match="introuvable"):
            load_model()

    @patch("src.predict.mlflow")
    def test_load_model_no_runs_raises(self, mock_mlflow):
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "0"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_runs.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="Aucun run"):
            load_model()
