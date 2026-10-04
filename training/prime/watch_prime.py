"""Poll prime-rl runs on the outputs volume and print new eval / step-summary lines.

  python training/prime/watch_prime.py b01_2b_mix_lr2e-4 b02_... --every 300
"""

import re
import subprocess
import sys
import time

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def orchestrator_log(run: str) -> str:
    ls = subprocess.run(["modal", "volume", "ls", "tpch-opd-outputs", f"/{run}"], capture_output=True, text=True).stdout
    sub = [line.strip() for line in ls.splitlines() if line.strip().startswith(f"{run}/") and "--" in line]
    if not sub:
        return ""
    path = f"/{sub[0]}/logs/attempt_1/orchestrator.log"
    out = subprocess.run(["modal", "volume", "get", "tpch-opd-outputs", path, "-"], capture_output=True, text=True)
    return ANSI.sub("", out.stdout)


def main():
    args = sys.argv[1:]
    every = 300
    if "--every" in args:
        i = args.index("--every")
        every = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    seen = {r: set() for r in args}
    while True:
        for run in args:
            for line in orchestrator_log(run).splitlines():
                if ("Evaluated" in line or re.search(r"Step \d+0 \|", line) or "Traceback" in line) and line not in seen[run]:
                    seen[run].add(line)
                    print(f"[{run}] {line[:220]}", flush=True)
        time.sleep(every)


if __name__ == "__main__":
    main()
