import os
import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://api:8000") + "/predict"

def predict_segment(fresh, milk, grocery, frozen, detergents, delicassen, channel):
    payload = {
        "Fresh": fresh,
        "Milk": milk,
        "Grocery": grocery,
        "Frozen": frozen,
        "Detergents_Paper": detergents,
        "Delicassen": delicassen,
        "Channel": channel
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        cluster = response.json()["cluster"]
        return f"Ce client appartient au Segment : {cluster}"
    except Exception as e:
        return f"Erreur lors de la prédiction : {e}"

interface = gr.Interface(
    fn=predict_segment,
    inputs=[
        gr.Slider(0, 100000, label="Dépenses Produits Frais"),
        gr.Slider(0, 100000, label="Dépenses Lait"),
        gr.Slider(0, 100000, label="Dépenses Épicerie"),
        gr.Slider(0, 100000, label="Dépenses Surgelés"),
        gr.Slider(0, 100000, label="Dépenses Détergents/Papier"),
        gr.Slider(0, 100000, label="Dépenses Épicerie fine"),
        gr.Radio([1, 2], label="Canal (1=Horeca, 2=Retail)", value=1)
    ],
    outputs=gr.Text(label="Résultat de la segmentation"),
    title="Segmentation de Clients de Gros",
    description="Ajustez les dépenses annuelles pour prédire le cluster du client."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)