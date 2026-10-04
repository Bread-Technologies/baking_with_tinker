"""Serve a HuggingFace model with an OpenAI-compatible vLLM server on Modal.

Used for models Tinker doesn't host (e.g. Qwen2.5-Coder-1.5B-Instruct).

  modal deploy tpch_eval/modal_serve.py
  python tpch_eval/eval.py --backend openai --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --base-url https://<workspace>--tpch-qwen-coder-serve.modal.run/v1

Override the model with MODEL_NAME=... at deploy time.
"""

import os
import subprocess

import modal

MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "MODEL_NAME": MODEL})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("tpch-qwen-coder")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve():
    subprocess.Popen(
        ["vllm", "serve", MODEL, "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "16384"]
    )
