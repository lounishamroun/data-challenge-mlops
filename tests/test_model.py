"""
Tests unitaires sur le modèle K-Means — comportement, métriques, cohérence.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data import preprocess, prepare_features


def _make_sample_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "Channel": rng.choice([1, 2], size=n),
        "Region": rng.choice([1, 2, 3], size=n),
        "Fresh": rng.randint(1000, 50000, n),
        "Milk": rng.randint(500, 20000, n),
        "Grocery": rng.randint(500, 20000, n),
        "Frozen": rng.randint(100, 10000, n),
        "Detergents_Paper": rng.randint(100, 10000, n),
        "Delicassen": rng.randint(100, 5000, n),
    })


class TestKMeansModel:
    def test_fit_produces_labels(self):
        df = _make_sample_df(30)
        df_scaled, _ = preprocess(df)
        model = KMeans(n_clusters=3, n_init=10, random_state=42)
        labels = model.fit_predict(df_scaled)
        assert len(labels) == 30
        assert set(labels).issubset({0, 1, 2})

    def test_n_clusters_respected(self):
        for k in [2, 3, 4]:
            df = _make_sample_df(30)
            df_scaled, _ = preprocess(df)
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            model.fit(df_scaled)
            assert model.n_clusters == k
            assert len(model.cluster_centers_) == k

    def test_inertia_is_positive(self):
        df = _make_sample_df(30)
        df_scaled, _ = preprocess(df)
        model = KMeans(n_clusters=3, n_init=10, random_state=42)
        model.fit(df_scaled)
        assert model.inertia_ > 0

    def test_silhouette_score_in_range(self):
        df = _make_sample_df(30)
        df_scaled, _ = preprocess(df)
        model = KMeans(n_clusters=3, n_init=10, random_state=42)
        labels = model.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, labels)
        assert -1 <= score <= 1

    def test_predict_output_shape(self):
        df = _make_sample_df(30)
        df_scaled, _ = preprocess(df)
        model = KMeans(n_clusters=3, n_init=10, random_state=42)
        model.fit(df_scaled)
        preds = model.predict(df_scaled)
        assert preds.shape == (30,)

    def test_deterministic_with_same_seed(self):
        df = _make_sample_df(30)
        df_scaled, _ = preprocess(df)
        m1 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(df_scaled)
        m2 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(df_scaled)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)

    def test_scaler_output_shape_matches_input(self):
        df = _make_sample_df(20)
        df_scaled, scaler = preprocess(df)
        assert df_scaled.shape[0] == len(df)
        # 6 numeric + 1 encoded channel
        assert df_scaled.shape[1] == 7

    def test_prepare_features_channel_encoding(self):
        df = pd.DataFrame([{
            "Channel": 1, "Region": 3,
            "Fresh": 1000, "Milk": 500, "Grocery": 800,
            "Frozen": 200, "Detergents_Paper": 300, "Delicassen": 100,
        }])
        feat = prepare_features(df)
        assert "Channel_2" in feat.columns
        assert feat["Channel_2"].iloc[0] == 0.0  # Channel=1 → Channel_2=0

        df2 = df.copy()
        df2["Channel"] = 2
        feat2 = prepare_features(df2)
        assert feat2["Channel_2"].iloc[0] == 1.0  # Channel=2 → Channel_2=1
