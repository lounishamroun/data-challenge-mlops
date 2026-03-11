"""
Tests d'intégration pour api/main.py — endpoints FastAPI.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from unittest.mock import patch

import api.main as main_module
from api.main import app


def _make_fake_model_and_scaler():
    """K-Means + scaler fitted on 7-feature synthetic data (matches prepare_features output)."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 7)  # 6 numeric cols + Channel_2
    model = KMeans(n_clusters=3, n_init=5, random_state=42)
    model.fit(X)
    scaler = StandardScaler()
    scaler.fit(X)
    return model, scaler


_VALID_PAYLOAD = {
    "Fresh": 12669,
    "Milk": 9656,
    "Grocery": 7561,
    "Frozen": 214,
    "Detergents_Paper": 2674,
    "Delicassen": 1338,
    "Channel": 1,
}


# ─────────────────────────────────────────────
# Fixture: client WITHOUT a loaded model
# ─────────────────────────────────────────────
@pytest.fixture
def client_no_model():
    """TestClient with no model loaded (simulates cold start before training)."""
    # Patch load_model to raise so lifespan sets globals to None gracefully,
    # then directly reset the module-level globals to ensure no model is set.
    with patch("api.main.load_model", side_effect=Exception("no model")):
        with TestClient(app) as c:
            # Ensure module globals are None regardless of lifespan outcome
            main_module._model = None
            main_module._scaler = None
            main_module._run_id = None
            yield c


# ─────────────────────────────────────────────
# Fixture: client WITH a loaded model
# ─────────────────────────────────────────────
@pytest.fixture
def client_with_model():
    """TestClient with a fake K-Means model injected."""
    model, scaler = _make_fake_model_and_scaler()
    with patch("api.main.load_model", return_value=(model, scaler, "fake-run-id")):
        with TestClient(app) as c:
            yield c


# ─────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────
class TestHealth:
    def test_health_no_model(self, client_no_model):
        r = client_no_model.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["model_loaded"] is False

    def test_health_with_model(self, client_with_model):
        r = client_with_model.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["model_loaded"] is True


# ─────────────────────────────────────────────
# /model-info
# ─────────────────────────────────────────────
class TestModelInfo:
    def test_model_info_no_model_returns_503(self, client_no_model):
        r = client_no_model.get("/model-info")
        assert r.status_code == 503

    def test_model_info_with_model(self, client_with_model):
        r = client_with_model.get("/model-info")
        assert r.status_code == 200
        data = r.json()
        assert data["run_id"] == "fake-run-id"
        assert data["experiment"] == "wholesale_segmentation"
        assert data["n_clusters"] == 3


# ─────────────────────────────────────────────
# /predict
# ─────────────────────────────────────────────
class TestPredict:
    def test_predict_no_model_returns_503(self, client_no_model):
        r = client_no_model.post("/predict", json=_VALID_PAYLOAD)
        assert r.status_code == 503

    def test_predict_returns_cluster(self, client_with_model):
        r = client_with_model.post("/predict", json=_VALID_PAYLOAD)
        assert r.status_code == 200
        data = r.json()
        assert "cluster" in data
        assert isinstance(data["cluster"], int)
        assert 0 <= data["cluster"] < 3

    def test_predict_channel_2(self, client_with_model):
        payload = {**_VALID_PAYLOAD, "Channel": 2}
        r = client_with_model.post("/predict", json=payload)
        assert r.status_code == 200
        assert isinstance(r.json()["cluster"], int)

    def test_predict_missing_field_returns_422(self, client_with_model):
        incomplete = {k: v for k, v in _VALID_PAYLOAD.items() if k != "Fresh"}
        r = client_with_model.post("/predict", json=incomplete)
        assert r.status_code == 422

    def test_predict_invalid_type_returns_422(self, client_with_model):
        bad = {**_VALID_PAYLOAD, "Fresh": "not-a-number"}
        r = client_with_model.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_deterministic(self, client_with_model):
        r1 = client_with_model.post("/predict", json=_VALID_PAYLOAD)
        r2 = client_with_model.post("/predict", json=_VALID_PAYLOAD)
        assert r1.json()["cluster"] == r2.json()["cluster"]
