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


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Prétraite les données pour l'entraînement du modèle K-Means: encodage des variables catégorielles et normalisation des colonnes numériques.

    Args:
        df: DataFrame brut chargé depuis le CSV.

    Returns:
        (df_scaled, scaler) — DataFrame scalé + scaler fité (pour réutilisation).
    """

    # Encodage de Channel (variable catégorielle)
    df_encoded = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True).astype(float)  # drop_first pour éviter la colinéarité

    # Concaténation avec les colonnes numériques
    df_features = pd.concat([df[NUMERIC_COLS], df_encoded], axis=1)

    # Normalisation
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled, columns=df_features.columns)
    return df_scaled, scaler