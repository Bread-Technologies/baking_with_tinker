"""Score every saved sampler checkpoint of a Tinker OPD run on the dev set (and optionally TPC-H).

  python training/eval_tinker_ckpts.py training/runs/tinker4b_proxy_r1            # dev only
  python training/eval_tinker_ckpts.py training/runs/tinker4b_proxy_r1 --final-tpch  # + TPC-H on final only

Per the hill-climbing rules, TPC-H is only run on the final checkpoint (chosen in advance).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--final-tpch", action="store_true")
    ap.add_argument("--only-final", action="store_true")
    args = ap.parse_args()

    run = Path(args.run_dir)
    ckpts = [json.loads(line) for line in open(run / "checkpoints.jsonl")]
    ckpts = [c for c in ckpts if c.get("sampler_path")]
    by_batch = {}
    for c in ckpts:  # "final" repeats the last numbered checkpoint; keep one per batch (prefer final)
        if c["batch"] not in by_batch or c["name"] == "final":
            by_batch[c["batch"]] = c
    ckpts = [by_batch[b] for b in sorted(by_batch)]
    if args.only_final:
        ckpts = ckpts[-1:]
    common = ["--backend", "tinker", "--model", args.model, "--temperature", "1.0", "--top-p", "0.95"]
    for i, c in enumerate(ckpts):
        name = c.get("name", str(i))
        tag = f"__{run.name}__{name}"
        print(f"== {name}: {c['sampler_path']}", flush=True)
        subprocess.run([sys.executable, str(HERE / "sql_eval.py"), "--split", "spider_dev_clean", *common,
                        "--model-path", c["sampler_path"], "--max-tokens", "8192", "--n", str(args.n),
                        "--concurrency", "48", "--tag", tag], check=False)
        if args.final_tpch and i == len(ckpts) - 1:
            subprocess.run([sys.executable, str(ROOT / "tpch_eval" / "eval.py"), *common,
                            "--model-path", c["sampler_path"], "--samples", "4", "--max-tokens", "16384",
                            "--concurrency", "88", "--tag", tag], check=False)


if __name__ == "__main__":
    main()
