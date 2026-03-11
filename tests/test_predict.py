"""
Tests unitaires pour src/predict.py — chargement modèle et prédiction.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sklearn.cluster import KMeans

from src.predict import load_model, predict
from src.data import NUMERIC_COLS, CATEGORICAL_COLS


def _make_fake_model() -> KMeans:
    """Crée un modèle K-Means fité sur des données synthétiques."""
    # 7 features : 6 numériques + 1 catégorielle encodée (Channel_2)
    rng = np.random.RandomState(42)
    X = rng.randn(20, 7)
    model = KMeans(n_clusters=3, n_init=5, random_state=42)
    model.fit(X)
    return model


class TestPredict:
    def test_returns_int(self):
        model = _make_fake_model()
        features = {
            "Fresh": 12669, "Milk": 9656, "Grocery": 7561,
            "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338,
            "Channel": 1,
        }
        result = predict(model, features)
        assert isinstance(result, int)

    def test_cluster_in_range(self):
        model = _make_fake_model()
        features = {
            "Fresh": 500, "Milk": 1000, "Grocery": 2000,
            "Frozen": 100, "Detergents_Paper": 300, "Delicassen": 50,
            "Channel": 2,
        }
        result = predict(model, features)
        assert 0 <= result < model.n_clusters

    def test_deterministic(self):
        model = _make_fake_model()
        features = {
            "Fresh": 12669, "Milk": 9656, "Grocery": 7561,
            "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338,
            "Channel": 1,
        }
        r1 = predict(model, features)
        r2 = predict(model, features)
        assert r1 == r2


class TestLoadModel:
    @patch("src.predict.mlflow")
    def test_load_model_with_run_id(self, mock_mlflow):
        fake_model = _make_fake_model()
        mock_mlflow.sklearn.load_model.return_value = fake_model

        model, run_id = load_model(run_id="abc123")

        mock_mlflow.sklearn.load_model.assert_called_once_with("runs:/abc123/kmeans_model")
        assert run_id == "abc123"
        assert model is fake_model

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
