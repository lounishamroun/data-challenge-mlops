"""Chargement et prétraitement du dataset Wholesale Customers."""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Colonnes de dépenses utilisées pour le clustering
NUMERIC_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
# Variable catégorielle (1 = Hôtellerie/Restauration, 2 = Retail)
CATEGORICAL_COLS = ["Channel"]


def load_data(url: str) -> pd.DataFrame:
    """Charge le dataset CSV depuis une URL ou un chemin local."""
    return pd.read_csv(url)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les features : colonnes numériques + encodage binaire de Channel."""
    df_out = df[NUMERIC_COLS].copy()
    # Channel_2 = 1 si retail (Channel == 2), sinon 0
    df_out["Channel_2"] = (df["Channel"] == 2).astype(float)
    return df_out


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Prétraite les données : encodage + normalisation (StandardScaler).

    Retourne le DataFrame normalisé et le scaler fité.
    """
    df_features = prepare_features(df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled, columns=df_features.columns)
    return df_scaled, scaler