import gradio as gr
from src.skeletal import FaceMesh

facemesh = FaceMesh("config.yaml")

with gr.Blocks() as demo:
    gr.Markdown("Estimate smilar person using this demo.")
    with gr.Row():
        with gr.Column(scale=1):
            input = gr.Image(type="filepath", label="Input image")
            dropdown = gr.Dropdown(
                [5, 10, 20, 30, 40, 50], value="20", label="Top K"
            )
            button = gr.Button("Estimate")
        with gr.Column(scale=2):
            output = gr.Dataframe()

    button.click(
        facemesh.estimate_similar_person, inputs=[input, dropdown], outputs=output
    )

demo.launch()
