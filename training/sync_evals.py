"""Pull Modal-side eval results (tpch_eval/modal_eval_lora.py) into the repo.

Merges /evals/test_summary.txt on the outputs volume into training/results/test_summary.txt (new lines
only) and copies per-question result files into tpch_eval/results/ and training/results/.

  python training/sync_evals.py
"""

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUMMARY = ROOT / "training" / "results" / "test_summary.txt"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["modal", "volume", "get", "--force", "tpch-opd-outputs", "/evals", tmp],
                           capture_output=True, text=True)
        src = Path(tmp) / "evals"
        if not src.exists():
            src = Path(tmp)
        remote = src / "test_summary.txt"
        if not remote.exists():
            print("no remote summary yet", r.stderr[-300:])
            return
        have = set(SUMMARY.read_text().splitlines()) if SUMMARY.exists() else set()
        new = [line for line in remote.read_text().splitlines() if line.strip() and line not in have]
        with open(SUMMARY, "a") as f:
            f.writelines(line + "\n" for line in new)
        for p in (src / "results").glob("*.json"):
            dst = ROOT / ("training" if p.name.startswith(("spider", "target_dev")) else "tpch_eval") / "results" / p.name
            dst.write_bytes(p.read_bytes())
        print("\n".join(new) if new else "no new lines")


if __name__ == "__main__":
    main()
