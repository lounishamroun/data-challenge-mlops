"""
Chargement et prétraitement du Wholesale Customers Dataset.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Colonnes numériques utilisées pour la segmentation
NUMERIC_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
CATEGORICAL_COLS = ["Channel"]

def load_data(url: str) -> pd.DataFrame:
    """
    Télécharge le dataset depuis l'URL UCI.

    Args:
        url: URL du dataset CSV.

    Returns:         
        DataFrame contenant les données brutes.
    """
    return pd.read_csv(url)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles et assemble les features.

    Args:
        df: DataFrame brut.

    Returns:
        DataFrame avec colonnes numériques + catégorielles encodées.
    """
    df_out = df[NUMERIC_COLS].copy()
    # Encodage binaire de Channel : Channel_2 = 1 si Channel == 2, sinon 0
    df_out["Channel_2"] = (df["Channel"] == 2).astype(float)
    return df_out


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Prétraite les données pour l'entraînement du modèle K-Means: encodage des variables catégorielles et normalisation des colonnes numériques.

    Args:
        df: DataFrame brut chargé depuis le CSV.

    Returns:
        (df_scaled, scaler) — DataFrame scalé + scaler fité (pour réutilisation).
    """
    df_features = prepare_features(df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled, columns=df_features.columns)
    return df_scaled, scaler