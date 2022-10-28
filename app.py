import gradio as gr

from convert import convert


def run(token: str, model_id: str) -> str:
    if token == "" or model_id == "":
        return """
        ### Invalid input 🐞
        
        Please fill a token and model_id.
        """
    try:
        pr_url = convert(token=token, model_id=model_id)
        return f"""
        ### Success 🔥

        Yay! This model was successfully converted and a PR was open using your token, here:

        {pr_url}
        """
    except Exception as e:
        return f"""
        ### Error 😢😢😢
        
        {e}
        """


DESCRIPTION = """
The steps are the following:

- Paste a read-access token from hf.co/settings/tokens. Read access is enough given that we will open a PR against the source repo.
- Input a model id from the Hub
- Click "Submit"
- That's it! You'll get feedback if it works or not, and if it worked, you'll get the URL of the opened PR 🔥

⚠️ For now only `pytorch_model.bin` files are supported but we'll extend in the future.
"""

demo = gr.Interface(
    title="Convert any model to Safetensors and open a PR",
    description=DESCRIPTION,
    allow_flagging="never",
    article="Check out the [Safetensors repo on GitHub](https://github.com/huggingface/safetensors)",
    inputs=[
        gr.Text(max_lines=1, label="your_hf_token"),
        gr.Text(max_lines=1, label="model_id"),
    ],
    outputs=[gr.Markdown(label="output")],
    fn=run,
)

demo.launch()
