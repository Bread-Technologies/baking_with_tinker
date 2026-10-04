"""Evaluate prime-rl LoRA milestones on the test sets as they appear (Modal vLLM, no Tinker cost).

For each milestone step whose adapter exists under /outputs/<run>/loras/step_N: deploy a vLLM
server with the base model + that LoRA, run TPC-H 22 / fresh / probe (t=1.0, 4 samples, thinking
on), append the scores to training/results/test_summary.txt, then stop the server.

  python training/auto_eval_lora.py --run b01_2b_mix_lr2e-4 --base /data/models/Qwen3.5-2B-untied \\
      --steps 80,100,120,140,160,180,200
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUMMARY = ROOT / "training" / "results" / "test_summary.txt"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def lora_exists(run: str, step: int) -> bool:
    out = sh(f"modal volume ls tpch-opd-outputs /{run}/loras/step_{step}").stdout
    return "adapter_model.safetensors" in out


def done(run: str, step: int) -> bool:
    return SUMMARY.exists() and f"{run} step{step} tpch:" in SUMMARY.read_text()


def evaluate(run: str, base: str, step: int, app: str, log_dir: Path, eval_args: str = "--samples 4"):
    name = f"{run}-s{step}"
    env = dict(os.environ, BASE=base, LORAS=f"{name}=/outputs/{run}/loras/step_{step}", MODAL_APP_NAME=app)
    r = subprocess.run("modal deploy tpch_eval/modal_serve_lora.py", shell=True, cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{name}] deploy failed: {r.stdout[-500:]} {r.stderr[-500:]}", flush=True)
        return
    url = sh(f"{sys.executable} -c \"import modal; print(modal.Function.from_name('{app}','serve').get_web_url())\"").stdout.strip()
    sh(f"curl -sSL -m 1200 {url}/v1/models")  # wait for the server to come up
    procs = {}
    for st in ("tpch", "fresh", "probe"):
        cmd = (f"{sys.executable} tpch_eval/eval.py --backend openai --base-url {url}/v1 --model {name} --thinking on "
               f"--temperature 1.0 --top-p 0.95 {eval_args} --max-tokens 16384 --concurrency 96 --set {st}")
        procs[st] = subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=open(log_dir / f"{name}_{st}.log", "w"),
                                     stderr=subprocess.STDOUT)
    for p in procs.values():
        p.wait()
    sh(f"modal app stop -y {app}")
    lines = []
    for st in ("tpch", "fresh", "probe"):
        m = re.search(r": ([0-9.]+/\d+ mean over .*)$", (log_dir / f"{name}_{st}.log").read_text(), flags=re.M)
        v = re.search(r"majority-vote@\d+ \d+/\d+", (log_dir / f"{name}_{st}.log").read_text())
        lines.append(f"{run} step{step} {st}: {m.group(1) if m else 'FAILED'}" + (f" | {v.group(0)}" if v else ""))
    with open(SUMMARY, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--app", default="tpch-lora-auto")
    ap.add_argument("--log-dir", default="/tmp")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--eval-args", default="--samples 4", help="extra tpch_eval/eval.py args, e.g. '--samples 16 --retries 2'")
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    log_dir = Path(args.log_dir)
    while steps:
        for s in list(steps):
            if done(args.run, s):
                steps.remove(s)
            elif lora_exists(args.run, s):
                evaluate(args.run, args.base, s, args.app, log_dir, args.eval_args)
                steps.remove(s)
        if steps:
            time.sleep(args.poll)


if __name__ == "__main__":
    main()
