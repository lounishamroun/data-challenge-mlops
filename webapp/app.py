import os
import io
import base64

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

API_URL = os.getenv("API_URL", "http://api:8000")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
NUMERIC_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

sns.set_theme(style="whitegrid")


# ── Helpers ────────────────────────────────────────────────

def _fig_to_image(fig):
    """Convertit une figure matplotlib en image PIL pour Gradio."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)


def _load_and_prepare():
    """Charge le dataset, prétraite, entraîne les modèles k=2..10."""
    df_raw = pd.read_csv(DATA_URL)
    df_features = df_raw[NUMERIC_COLS].copy()
    df_features["Channel_2"] = (df_raw["Channel"] == 2).astype(float)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)
    return df_raw, df_scaled


# ── Onglet Prédiction ─────────────────────────────────────

def predict_segment(fresh, milk, grocery, frozen, detergents, delicassen, channel):
    payload = {
        "Fresh": fresh, "Milk": milk, "Grocery": grocery,
        "Frozen": frozen, "Detergents_Paper": detergents,
        "Delicassen": delicassen, "Channel": channel,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        cluster = response.json()["cluster"]
        return f"Ce client appartient au Segment : {cluster}"
    except Exception as e:
        return f"Erreur lors de la prédiction : {e}"


# ── Onglet Interprétation ─────────────────────────────────

def generate_analysis(k_selected):
    """Génère toutes les visualisations d'interprétation."""
    k_selected = int(k_selected)
    df_raw, df_scaled = _load_and_prepare()

    # Modèle sélectionné
    model = KMeans(n_clusters=k_selected, n_init=20, random_state=42)
    labels = model.fit_predict(df_scaled)
    df_raw["Cluster"] = labels
    sil = silhouette_score(df_scaled, labels)

    # Elbow method (k=2..10)
    k_range = range(2, 11)
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        km.fit(df_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(df_scaled, km.labels_))

    # --- 1. Elbow + Silhouette ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(list(k_range), inertias, "bo-")
    ax1.axvline(x=k_selected, color="red", linestyle="--", label=f"k={k_selected}")
    ax1.set_xlabel("Nombre de clusters (k)")
    ax1.set_ylabel("Inertie")
    ax1.set_title("Méthode du coude")
    ax1.legend()

    ax2.plot(list(k_range), silhouettes, "ro-")
    ax2.axvline(x=k_selected, color="red", linestyle="--", label=f"k={k_selected}")
    ax2.set_xlabel("Nombre de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score par k")
    ax2.legend()
    fig1.suptitle("Sélection du nombre optimal de clusters", fontsize=14)
    plt.tight_layout()
    img_elbow = _fig_to_image(fig1)

    # --- 2. PCA : K-Means vs Region ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(df_scaled)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6, s=40)
    centers_pca = pca.transform(model.cluster_centers_)
    ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", marker="X", s=200, edgecolors="black", label="Centroïdes")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("Clusters K-Means")
    ax1.legend()
    plt.colorbar(sc1, ax=ax1, label="Cluster")

    sc2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=df_raw["Region"], cmap="Set1", alpha=0.6, s=40)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title("Vraies étiquettes (Region)")
    plt.colorbar(sc2, ax=ax2, label="Region")
    fig2.suptitle("Projection PCA 2D", fontsize=14)
    plt.tight_layout()
    img_pca = _fig_to_image(fig2)

    # --- 3. Heatmap des centres ---
    centers_df = pd.DataFrame(model.cluster_centers_, columns=df_scaled.columns)
    fig3, ax = plt.subplots(figsize=(12, max(3, k_selected)))
    sns.heatmap(centers_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Centres des clusters (valeurs scalées)")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    img_heatmap = _fig_to_image(fig3)

    # --- 4. Répartition clusters vs Region ---
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    cluster_counts.plot(kind="bar", color=sns.color_palette("viridis", len(cluster_counts)), ax=ax1)
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Nombre de clients")
    ax1.set_title("Répartition par cluster (K-Means)")
    for i, v in enumerate(cluster_counts):
        ax1.text(i, v + 2, str(v), ha="center", fontweight="bold")

    region_counts = df_raw["Region"].value_counts().sort_index()
    region_counts.plot(kind="bar", color=sns.color_palette("Set1", len(region_counts)), ax=ax2)
    ax2.set_xlabel("Region")
    ax2.set_ylabel("Nombre de clients")
    ax2.set_title("Répartition réelle (Region)")
    for i, v in enumerate(region_counts):
        ax2.text(i, v + 2, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    img_distrib = _fig_to_image(fig4)

    # --- 5. Contingence Region × Cluster ---
    contingency = pd.crosstab(df_raw["Region"], df_raw["Cluster"],
                              rownames=["Region"], colnames=["Cluster"])
    fig5, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("Contingence Region × Cluster (effectifs)")
    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
    sns.heatmap(contingency_pct, annot=True, fmt=".1f", cmap="Blues", ax=ax2)
    ax2.set_title("Contingence Region × Cluster (% par Region)")
    plt.tight_layout()
    img_contingency = _fig_to_image(fig5)

    summary = (
        f"**Modèle K-Means avec k={k_selected}**\n\n"
        f"- Silhouette Score : **{sil:.4f}**\n"
        f"- Inertie : **{model.inertia_:.2f}**\n"
        f"- Nombre d'échantillons : **{len(df_raw)}**\n\n"
        f"Meilleur k par silhouette : **{list(k_range)[np.argmax(silhouettes)]}** "
        f"(score = {max(silhouettes):.4f})"
    )

    return summary, img_elbow, img_pca, img_heatmap, img_distrib, img_contingency


# ── Construction de l'interface ────────────────────────────

with gr.Blocks(title="Segmentation de Clients de Gros") as app:
    gr.Markdown("# Segmentation de Clients de Gros")

    with gr.Tab("Prédiction"):
        gr.Markdown("Ajustez les dépenses annuelles pour prédire le cluster du client.")
        with gr.Row():
            with gr.Column():
                fresh = gr.Slider(0, 100000, label="Dépenses Produits Frais")
                milk = gr.Slider(0, 100000, label="Dépenses Lait")
                grocery = gr.Slider(0, 100000, label="Dépenses Épicerie")
                frozen = gr.Slider(0, 100000, label="Dépenses Surgelés")
                detergents = gr.Slider(0, 100000, label="Dépenses Détergents/Papier")
                delicassen = gr.Slider(0, 100000, label="Dépenses Épicerie fine")
                channel = gr.Radio([1, 2], label="Canal (1=Horeca, 2=Retail)", value=1)
                btn_predict = gr.Button("Prédire", variant="primary")
            with gr.Column():
                result = gr.Text(label="Résultat de la segmentation")

        btn_predict.click(
            predict_segment,
            inputs=[fresh, milk, grocery, frozen, detergents, delicassen, channel],
            outputs=result,
        )

    with gr.Tab("Interprétation"):
        gr.Markdown("Analyse des clusters : choisissez un k puis cliquez sur **Analyser**.")
        with gr.Row():
            k_slider = gr.Slider(2, 10, value=3, step=1, label="Nombre de clusters (k)")
            btn_analyze = gr.Button("Analyser", variant="primary")

        summary_md = gr.Markdown()
        gr.Markdown("### Méthode du coude & Silhouette")
        img_elbow = gr.Image(label="Coude & Silhouette")
        gr.Markdown("### Projection PCA 2D")
        img_pca = gr.Image(label="PCA : K-Means vs Region")
        gr.Markdown("### Heatmap des centres de clusters")
        img_heatmap = gr.Image(label="Centres des clusters")
        gr.Markdown("### Répartition des clusters vs Region")
        img_distrib = gr.Image(label="Distribution")
        gr.Markdown("### Contingence Region × Cluster")
        img_contingency = gr.Image(label="Contingence")

        btn_analyze.click(
            generate_analysis,
            inputs=k_slider,
            outputs=[summary_md, img_elbow, img_pca, img_heatmap, img_distrib, img_contingency],
        )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)