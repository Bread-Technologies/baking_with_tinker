"""Watch Tinker sweep runs and score each new checkpoint on target_dev (in parallel, as they appear).

Appends one line per scored checkpoint to training/results/sweep_summary.tsv:
  run  step  acc  ci95  n  sampler_path

  python training/watch_eval.py "training/runs/s0*" --max-parallel 6
"""

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SUMMARY = HERE / "results" / "sweep_summary.tsv"


def scored() -> set[tuple[str, str]]:
    if not SUMMARY.exists():
        return set()
    return {tuple(line.split("\t")[:2]) for line in SUMMARY.read_text().splitlines()[1:] if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--idle-exit-min", type=float, default=30.0, help="exit after this long with nothing new")
    args = ap.parse_args()

    SUMMARY.parent.mkdir(exist_ok=True)
    if not SUMMARY.exists():
        SUMMARY.write_text("run\tstep\tacc\tci95\tn\tsampler_path\n")
    running: dict[tuple[str, str], subprocess.Popen] = {}
    launched: set[tuple[str, str]] = set(scored())
    last_activity = time.time()
    while True:
        for key, proc in list(running.items()):
            if proc.poll() is not None:
                out = proc.stdout.read()
                m = re.search(r"(\d+)/(\d+) = ([0-9.]+) ± ([0-9.]+)", out)
                run, step = key
                if m:
                    with open(SUMMARY, "a") as f:
                        f.write(f"{run}\t{step}\t{m.group(3)}\t{m.group(4)}\t{m.group(2)}\t{proc.args[proc.args.index('--model-path') + 1]}\n")
                    print(f"{run} step {step}: {m.group(3)} ± {m.group(4)}", flush=True)
                else:
                    print(f"{run} step {step}: eval failed\n{out[-500:]}", flush=True)
                del running[key]
                last_activity = time.time()
        for run_dir in sorted(glob.glob(args.pattern)):
            ck = Path(run_dir) / "checkpoints.jsonl"
            if not ck.exists():
                continue
            for line in ck.read_text().splitlines():
                c = json.loads(line)
                if not c.get("sampler_path") or c["name"] == "final":
                    continue
                key = (Path(run_dir).name, str(c["batch"]))
                if key in launched or len(running) >= args.max_parallel:
                    continue
                cmd = [sys.executable, str(HERE / "sql_eval.py"), "--split", "target_dev", "--backend", "tinker",
                       "--model", "Qwen/Qwen3.5-4B", "--temperature", "1.0", "--top-p", "0.95", "--max-tokens", "16384",
                       "--repeats", str(args.repeats), "--concurrency", "48", "--model-path", c["sampler_path"],
                       "--tag", f"__{key[0]}__{key[1]}"]
                running[key] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                launched.add(key)
                last_activity = time.time()
        if not running and time.time() - last_activity > args.idle_exit_min * 60:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
