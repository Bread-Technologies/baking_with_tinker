"""Serve a base model plus LoRA adapters from the training volumes with vLLM (OpenAI-compatible).

  BASE=/data/models/Qwen3.5-2B-untied LORAS="r001-s100=/outputs/r001_2b_proxy/loras/step_100" \\
      modal deploy tpch_eval/modal_serve_lora.py
  python tpch_eval/eval.py --backend openai --model r001-s100 --base-url https://<ws>--tpch-lora-serve.modal.run/v1 ...

LORAS is a space-separated list of name=path; each name becomes a servable model id.
"""

import os
import subprocess

import modal

BASE = os.environ.get("BASE", "/data/models/Qwen3.5-2B-untied")
LORAS = os.environ.get("LORAS", "")

image = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python=None)
    .entrypoint([])
    .dockerfile_commands(["RUN ln -sf $(which python3) /usr/local/bin/python"])
    .env({"BASE": BASE, "LORAS": LORAS})
)
app = modal.App(os.environ.get("MODAL_APP_NAME", "tpch-lora"))
VOLS = {"/data": modal.Volume.from_name("tpch-opd-data"), "/outputs": modal.Volume.from_name("tpch-opd-outputs"),
        "/root/.cache/huggingface": modal.Volume.from_name("hf-cache", create_if_missing=True)}


@app.function(image=image, gpu="H100", volumes=VOLS, timeout=4 * 3600, scaledown_window=5 * 60)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve():
    cmd = ["vllm", "serve", BASE, "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "32768",
           "--served-model-name", "base"]
    if LORAS.strip():
        cmd += ["--enable-lora", "--max-lora-rank", "64", "--max-loras", "4", "--lora-modules", *LORAS.split()]
    subprocess.Popen(cmd)
