"""Run prime-rl on Modal: on-policy distillation of a Qwen3.5 student with the 397B teacher on Tinker.

The prime-rl image runs vLLM (student sampling) + trainer on 2 GPUs. The teacher is
tinker_teacher_shim.py on localhost:8001, which answers prime-rl's prefill-scoring call with
Tinker compute_logprobs, so the 397B never runs on our GPUs.

  modal run training/prime/modal_prime.py::inspect
  modal run --detach training/prime/modal_prime.py::train --config training/prime/configs/opd_2b.toml --run-name r001
"""

import os
from pathlib import Path

import modal

HERE = Path(__file__).parent
TRAINING = HERE.parent

image = (
    modal.Image.from_registry("ghcr.io/primeintellect-ai/prime-rl:main", add_python=None)
    .entrypoint([])
    .add_local_dir(str(TRAINING / "prime_env" / "sql_opd"), "/pkgs/sql_opd", copy=True)
    .add_local_file(str(TRAINING / "tinker_teacher_shim.py"), "/pkgs/shim/tinker_teacher_shim.py", copy=True)
    .run_commands(
        "cd /app && uv pip install --python /app/.venv/bin/python -e /pkgs/sql_opd tinker fastapi uvicorn duckdb"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)
data = modal.Volume.from_name("tpch-opd-data")
outputs = modal.Volume.from_name("tpch-opd-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_dict({"TINKER_API_KEY": os.environ.get("TINKER_API_KEY", "")})]
app = modal.App("tpch-opd-prime")
VOLS = {"/data": data, "/outputs": outputs, "/root/.cache/huggingface": hf_cache}


def _sh(cmd: str, **kw):
    import subprocess
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


@app.function(image=image, volumes=VOLS, timeout=1800)
def inspect():
    for cmd in [
        "which python; python --version; ls /app | head -30",
        "cd /app && git log -1 --format='%h %cd' 2>/dev/null || true",
        "cd /app && .venv/bin/python -c 'import prime_rl, verifiers, vllm, torch; print(vllm.__version__, torch.__version__)'",
        "cd /app && .venv/bin/python -c 'import sql_opd, tinker, duckdb; print(\"taskset ok\")'",
        "cd /app && .venv/bin/rl --help 2>&1 | head -40",
        "ls /data | head",
    ]:
        r = _sh(cmd)
        print(r.stdout[-3000:], r.stderr[-2000:], flush=True)


@app.function(image=image, gpu="H100:2", volumes=VOLS, secrets=secrets, timeout=24 * 3600)
def train(config_toml: str, run_name: str, extra_args: str = ""):
    import subprocess
    import time

    run_dir = Path("/outputs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.toml").write_text(config_toml)

    shim = subprocess.Popen(
        ["/app/.venv/bin/python", "-m", "uvicorn", "tinker_teacher_shim:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd="/pkgs/shim", stdout=open(run_dir / "shim.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(5)

    cmd = f"cd /app && .venv/bin/rl @ {run_dir / 'config.toml'} --output-dir {run_dir} {extra_args}"
    print("$", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(run_dir / "rl.log", "w") as log:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
    # The launcher returns once components start; they log to files under run_dir. Keep the
    # container alive while they run, stream orchestrator progress, and commit the volume
    # periodically so checkpoints and logs are visible from outside while training runs.
    rc = proc.wait()
    seen = 0
    while True:
        alive = _sh("pgrep -f 'prime_rl.(orchestrator|trainer)' | wc -l").stdout.strip()
        orch = sorted(run_dir.glob("*/logs/attempt_*/orchestrator.log"))
        if orch:
            lines = orch[-1].read_text(errors="replace").splitlines()
            for line in lines[seen:]:
                if any(k in line for k in ("Step", "step", "Eval", "eval", "ERROR", "Error", "Traceback")):
                    print(line, flush=True)
            seen = len(lines)
        outputs.commit()
        if alive in ("", "0"):
            break
        time.sleep(60)
    shim.terminate()
    _sh(f"curl -s localhost:8001/stats > {run_dir / 'teacher_stats.json'} || true")
    outputs.commit()
    print("exit code", rc)
    return rc


@app.function(image=image, volumes=VOLS, timeout=1800)
def dry_run(config_toml: str):
    Path("/tmp/cfg.toml").write_text(config_toml)
    r = _sh("cd /app && .venv/bin/rl @ /tmp/cfg.toml --output-dir /tmp/dry --dry-run")
    print(r.stdout[-6000:], r.stderr[-6000:], flush=True)
    return r.returncode


@app.local_entrypoint()
def main(config: str = "", run_name: str = "", extra_args: str = "", dry: bool = False):
    if dry:
        print("dry-run exit", dry_run.remote(Path(config).read_text()))
        return
    rc = train.remote(Path(config).read_text(), run_name, extra_args)
    print("finished with", rc)
