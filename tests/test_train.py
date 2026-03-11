"""
Tests unitaires pour src/train.py — entraînement K-Means avec MLflow.
"""

import mlflow
import pandas as pd
from unittest.mock import patch

from src.train import train, DEFAULT_K, MLFLOW_EXPERIMENT


def _make_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Channel": [1, 2, 1, 2, 1, 2],
        "Region": [3, 3, 3, 1, 2, 2],
        "Fresh": [12669, 7057, 6353, 13265, 22615, 9413],
        "Milk": [9656, 9810, 8808, 1196, 5410, 8259],
        "Grocery": [7561, 9568, 7684, 4221, 7198, 5126],
        "Frozen": [214, 1762, 2405, 6404, 3915, 666],
        "Detergents_Paper": [2674, 3293, 3516, 507, 1777, 1795],
        "Delicassen": [1338, 1776, 7844, 1788, 5185, 1451],
    })


class TestTrain:
    @patch("src.train.load_data")
    def test_train_logs_to_mlflow(self, mock_load_data, tmp_path):
        """Vérifie que train() crée un run MLflow avec les bonnes métriques."""
        mock_load_data.return_value = _make_sample_df()

        # Utiliser un tracking URI temporaire pour ne pas polluer
        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")

        train(n_clusters=2)

        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        assert experiment is not None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) >= 1

        last_run = runs.iloc[0]
        assert "params.n_clusters" in last_run.index
        assert last_run["params.n_clusters"] == "2"
        assert "metrics.inertia" in last_run.index
        assert "metrics.silhouette_score" in last_run.index
        assert last_run["metrics.silhouette_score"] > -1  # silhouette ∈ [-1, 1]

    @patch("src.train.load_data")
    def test_train_different_k(self, mock_load_data, tmp_path):
        """Vérifie que le paramètre n_clusters est bien utilisé."""
        mock_load_data.return_value = _make_sample_df()
        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")

        train(n_clusters=3)

        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        last_run = runs.iloc[0]
        assert last_run["params.n_clusters"] == "3"

    def test_default_k_value(self):
        assert DEFAULT_K == 3
