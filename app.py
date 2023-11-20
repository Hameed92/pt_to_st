import csv
from datetime import datetime
import os
from typing import Optional
import gradio as gr

from convert import convert
from huggingface_hub import HfApi, Repository


DATASET_REPO_URL = "https://huggingface.co/datasets/safetensors/conversions"
DATA_FILENAME = "data.csv"
DATA_FILE = os.path.join("data", DATA_FILENAME)

HF_TOKEN = os.environ.get("HF_TOKEN")

repo: Optional[Repository] = None
# TODO
if False and HF_TOKEN:
    repo = Repository(local_dir="data", clone_from=DATASET_REPO_URL, token=HF_TOKEN)


def run(model_id: str, is_private: bool, token: Optional[str] = None) -> str:
    if model_id == "":
        return """
        ### Invalid input 🐞
        
        Please fill a token and model_id.
        """
    try:
        if is_private:
            api = HfApi(token=token)
        else:
            api = HfApi(token=HF_TOKEN)
        hf_is_private = api.model_info(repo_id=model_id).private
        if is_private and not hf_is_private:
            # This model is NOT private
            # Change the token so we make the PR on behalf of the bot.
            api = HfApi(token=HF_TOKEN)

        print("is_private", is_private)

        commit_info, errors = convert(api=api, model_id=model_id)
        print("[commit_info]", commit_info)


        string =  f"""
        ### Success 🔥

        Yay! This model was successfully converted and a PR was open using your token, here:

        [{commit_info.pr_url}]({commit_info.pr_url})
        """
        if errors:
            string += "\nErrors during conversion:\n"
            string += "\n".join(f"Error while converting {filename}: {e}, skipped conversion" for filename, e in errors)
        return string
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

title="Convert any model to Safetensors and open a PR"
allow_flagging="never"

def token_text(visible=False):
    return gr.Text(max_lines=1, label="your_hf_token", visible=visible)

with gr.Blocks(title=title) as demo:
    description = gr.Markdown(f"""# {title}""")
    description = gr.Markdown(DESCRIPTION)

    with gr.Row() as r:
        with gr.Column() as c:
            model_id = gr.Text(max_lines=1, label="model_id")
            is_private = gr.Checkbox(label="Private model")
            token = token_text()
            with gr.Row() as c:
                clean = gr.ClearButton()
                submit = gr.Button("Submit", variant="primary")

        with gr.Column() as d:
            output = gr.Markdown()

    is_private.change(lambda s: token_text(s), inputs=is_private, outputs=token)
    submit.click(run, inputs=[model_id, is_private, token], outputs=output, concurrency_limit=1)

demo.queue(max_size=10).launch(show_api=True)
