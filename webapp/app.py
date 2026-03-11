import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# Wholesale Customer Segmentation")
    gr.Markdown("WebApp en cours de développement...")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
