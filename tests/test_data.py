"""
Tests unitaires pour src/data.py — chargement et prétraitement des données.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data import load_data, preprocess, NUMERIC_COLS, CATEGORICAL_COLS


def _make_sample_df() -> pd.DataFrame:
    """Crée un DataFrame échantillon avec la même structure que le dataset réel."""
    return pd.DataFrame({
        "Channel": [1, 2, 1, 2],
        "Region": [3, 3, 3, 1],
        "Fresh": [12669, 7057, 6353, 13265],
        "Milk": [9656, 9810, 8808, 1196],
        "Grocery": [7561, 9568, 7684, 4221],
        "Frozen": [214, 1762, 2405, 6404],
        "Detergents_Paper": [2674, 3293, 3516, 507],
        "Delicassen": [1338, 1776, 7844, 1788],
    })


class TestLoadData:
    def test_returns_dataframe(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = _make_sample_df()
        df.to_csv(csv_path, index=False)

        result = load_data(str(csv_path))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_columns_present(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = _make_sample_df()
        df.to_csv(csv_path, index=False)

        result = load_data(str(csv_path))
        for col in NUMERIC_COLS + CATEGORICAL_COLS:
            assert col in result.columns


class TestPreprocess:
    def test_returns_scaled_df_and_scaler(self):
        df = _make_sample_df()
        df_scaled, scaler = preprocess(df)

        assert isinstance(df_scaled, pd.DataFrame)
        assert isinstance(scaler, StandardScaler)

    def test_numeric_cols_present(self):
        df = _make_sample_df()
        df_scaled, _ = preprocess(df)

        for col in NUMERIC_COLS:
            assert col in df_scaled.columns

    def test_categorical_encoded(self):
        df = _make_sample_df()
        df_scaled, _ = preprocess(df)

        # Channel (int) passe par get_dummies — doit être présent dans le résultat
        assert "Channel" in df_scaled.columns or "Channel_2" in df_scaled.columns

    def test_scaled_values_centered(self):
        df = _make_sample_df()
        df_scaled, _ = preprocess(df)

        # Les colonnes scalées doivent avoir une moyenne proche de 0
        for col in NUMERIC_COLS:
            assert abs(df_scaled[col].mean()) < 1e-10

    def test_row_count_preserved(self):
        df = _make_sample_df()
        df_scaled, _ = preprocess(df)
        assert len(df_scaled) == len(df)
