"""Evaluate prime-rl LoRA checkpoints entirely on Modal (survives the local sandbox restarting).

eval_lora: one H100 runs vLLM (base + one LoRA) on localhost and the eval scripts against it, then
appends summary lines to /outputs/evals/test_summary.txt and copies the per-question result files to
/outputs/evals/results/. No public endpoint.
queue: a cheap CPU function that waits for each milestone LoRA to appear and spawns eval_lora for it.

  modal deploy tpch_eval/modal_eval_lora.py
  python -c "import modal; modal.Function.from_name('tpch-eval-lora','queue').spawn(
      'b01_2b_mix_lr2e-4', '/data/models/Qwen3.5-2B-untied', [120, 140], '--samples 16 --retries 2', 0)"
  modal volume get tpch-opd-outputs /evals/test_summary.txt -    # read results
"""

import subprocess
import time
from pathlib import Path

import modal

REPO = "/home/user/baking_with_tinker"  # same absolute path as the sandbox, so dataset rows' paths resolve
IGNORE = ["**/__pycache__", "**/results", "**/runs", "**/data/spider*", "**/data/*.jsonl", "**/*.duckdb",
          "**/prime", "**/prime_env"]

image = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python=None)
    .entrypoint([])
    .dockerfile_commands(["RUN ln -sf $(which python3) /usr/local/bin/python"])
    .run_commands("python -m pip install duckdb python-dotenv openai tpchgen-cli sqlglot")
    .add_local_dir(Path(__file__).parent, f"{REPO}/tpch_eval", ignore=IGNORE)
    .add_local_dir(Path(__file__).parent.parent / "training", f"{REPO}/training", ignore=IGNORE + ["**/data"])
)
cpu_image = modal.Image.debian_slim()
app = modal.App("tpch-eval-lora")
outputs = modal.Volume.from_name("tpch-opd-outputs")
VOLS = {"/data": modal.Volume.from_name("tpch-opd-data"), "/outputs": outputs,
        "/root/.cache/huggingface": modal.Volume.from_name("hf-cache", create_if_missing=True)}


def _parse(log: str, kind: str) -> str:
    import re
    if kind == "spider":
        m = re.search(r"(\d+/\d+ = [0-9.]+ ± [0-9.]+)", log)
        return m.group(1) if m else "FAILED"
    m = re.search(r": ([0-9.]+/\d+ mean over .*)$", log, flags=re.M)
    v = re.search(r"majority-vote@\d+ \d+/\d+", log)
    return (m.group(1) if m else "FAILED") + (f" | {v.group(0)}" if v else "")


@app.function(image=image, gpu="H100", volumes=VOLS, timeout=4 * 3600)
def eval_lora(run: str, base: str, step: int, eval_args: str = "--samples 16 --retries 2", spider: int = 0,
              sets: tuple = ("tpch", "fresh", "probe")) -> list[str]:
    import os
    import shutil
    if not os.path.exists(f"{REPO}/training/data"):  # containers can be reused across calls
        os.symlink("/data", f"{REPO}/training/data")
    # step 0 = the untrained base model (baseline), served under the run name
    name = f"{run}-s{step}"
    cmd = ["vllm", "serve", base, "--port", "8000", "--max-model-len", "32768", "--served-model-name", name if step == 0 else "base"]
    if step:
        cmd += ["--enable-lora", "--max-lora-rank", "64", "--lora-modules", f"{name}=/outputs/{run}/loras/step_{step}"]
    subprocess.Popen(cmd, stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        if subprocess.run("curl -sf localhost:8000/v1/models", shell=True, capture_output=True).returncode == 0:
            break
        time.sleep(5)
    url = "http://localhost:8000/v1"
    common = f"--backend openai --base-url {url} --model {name} --thinking on --temperature 1.0 --top-p 0.95 --max-tokens 16384"
    procs = {st: subprocess.Popen(f"python tpch_eval/eval.py {common} {eval_args} --concurrency 96 --set {st}", shell=True,
                                  cwd=REPO, stdout=open(f"/tmp/{st}.log", "w"), stderr=subprocess.STDOUT) for st in sets}
    if spider:
        procs["spider"] = subprocess.Popen(
            f"python training/sql_eval.py --split spider_test_clean {common} --n {spider} --concurrency 96 --tag __{name}",
            shell=True, cwd=REPO, stdout=open("/tmp/spider.log", "w"), stderr=subprocess.STDOUT)
    for p in procs.values():
        p.wait()
    lines = []
    for k in procs:
        log = Path(f"/tmp/{k}.log").read_text()
        res = _parse(log, k)
        if "FAILED" in res:
            print(log[-3000:], flush=True)
        lines.append(f"{run} step{step} {k if k != 'spider' else 'spider_test'}: {res}")
    outputs.reload()
    dst = Path("/outputs/evals/results")
    dst.mkdir(parents=True, exist_ok=True)
    for d in (Path(REPO) / "tpch_eval" / "results", Path(REPO) / "training" / "results"):
        for f in d.glob(f"*{name}*") if d.exists() else []:
            shutil.copy(f, dst / f.name)
    with open("/outputs/evals/test_summary.txt", "a") as f:
        f.write("\n".join(lines) + "\n")
    outputs.commit()
    print("\n".join(lines), flush=True)
    return lines


@app.function(image=cpu_image, volumes={"/outputs": outputs}, timeout=24 * 3600)
def queue(run: str, base: str, steps: list[int], eval_args: str = "--samples 16 --retries 2", spider: int = 0):
    pending, calls = list(steps), []
    while pending:
        outputs.reload()
        for s in list(pending):
            if (Path(f"/outputs/{run}/loras/step_{s}") / "adapter_model.safetensors").exists():
                calls.append(eval_lora.spawn(run, base, s, eval_args, spider))
                pending.remove(s)
        if pending:
            time.sleep(120)
    return [line for c in calls for line in c.get()]
