"""Poll prime-rl runs on the outputs volume and print new eval / step-summary lines.

  python training/prime/watch_prime.py b01_2b_mix_lr2e-4 b02_... --every 300
"""

import re
import subprocess
import sys
import time

ANSI = re.compile(r"\x1b\[[0-9;]*m")


_where: dict[str, tuple[str, str]] = {}


def _containers() -> list[str]:
    out = subprocess.run(["modal", "container", "list"], capture_output=True, text=True).stdout
    return [line.split()[1] for line in out.splitlines() if "tpch-opd-prime" in line and "ta-" in line]


def orchestrator_log(run: str) -> str:
    """Read the live log from inside the running container (the volume only syncs at commit)."""
    if run not in _where:
        for c in _containers():
            found = subprocess.run(["modal", "container", "exec", c, "--", "find", f"/outputs/{run}", "-name",
                                    "orchestrator.log", "-printf", "%T@ %p\n"],
                                   capture_output=True, text=True, timeout=90).stdout.split("\n")
            found = sorted((float(t), p) for t, _, p in (line.partition(" ") for line in found if line.strip()))
            if found:
                _where[run] = (c, found[-1][1])  # newest log: earlier attempts leave old run dirs behind
                break
    if run not in _where:
        return ""
    c, path = _where[run]
    out = subprocess.run(["modal", "container", "exec", c, "--", "cat", path], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        _where.pop(run, None)
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
