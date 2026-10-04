"""Decontaminate training prompts against the TPC-H test (the ONLY training-side code that reads the test).

A training prompt is dropped if any of these hold against any of the 22 test questions:
  text    : question wording too similar (character 5-gram Jaccard over normalized text)
  struct  : its teacher SQL has the same table set AND the same aggregate/subquery fingerprint
            as a reference query, and they share a distinctive literal or join 3+ tables

Everything dropped is logged with the matching test query, so the filter itself can be audited.

  python training/decontam.py training/data/target_train_raw.jsonl training/data/target_train.jsonl
"""

import json
import re
import sys
from pathlib import Path

import sqlglot
from sqlglot import exp

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tpch_eval"))
from questions import QUESTIONS  # noqa: E402  (test access is confined to this file)

TEXT_JACCARD = 0.35
COMMON_LITERALS = {"0", "1", "2", "100", "0.0", "1.0"}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9#%' ]", " ", t.lower())).strip()


def shingles(t: str, k: int = 5) -> set[str]:
    t = norm(t)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def fingerprint(sql: str):
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:  # noqa: BLE001
        return None
    tables = frozenset(t.name.lower() for t in tree.find_all(exp.Table))
    aggs = tuple(sorted(type(a).__name__ for a in tree.find_all(exp.AggFunc)))
    shape = (len(list(tree.find_all(exp.Subquery))), len(list(tree.find_all(exp.Exists))),
             tree.find(exp.Having) is not None, tree.find(exp.Window) is not None)
    lits = {str(l.this).lower() for l in tree.find_all(exp.Literal)} - COMMON_LITERALS
    return tables, aggs, shape, lits


def load_test():
    test = []
    for q, text in QUESTIONS.items():
        sql = (ROOT / "tpch_eval" / "reference" / f"q{q:02d}.sql").read_text()
        test.append({"q": q, "shingles": shingles(text.split("Output columns:")[0]), "fp": fingerprint(sql)})
    return test


def check(row, test):
    question = row.get("question") or row["prompt"].split("Question:\n", 1)[-1]
    sh = shingles(question.split("Output columns:")[0])
    fp = fingerprint(row.get("teacher_sql") or row.get("gold_sql") or "")
    for t in test:
        j = jaccard(sh, t["shingles"])
        if j >= TEXT_JACCARD:
            return f"text~Q{t['q']} (jaccard {j:.2f})"
        if fp and t["fp"]:
            tables, aggs, shape, lits = fp
            ttables, taggs, tshape, tlits = t["fp"]
            shared = lits & tlits
            same_shape = tables == ttables and aggs == taggs and shape == tshape
            if same_shape and (len(shared) >= 1 or len(tables) >= 3):
                return f"struct~Q{t['q']} (shared literals {sorted(shared)[:5]})"
    return None


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    test = load_test()
    rows = [json.loads(line) for line in open(src)]
    rows = [r for r in rows if r.get("kept", True)]
    kept, dropped = [], []
    for r in rows:
        why = check(r, test)
        (dropped if why else kept).append({**r, "decontam": why} if why else r)
    with open(dst, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    log = dst.with_suffix(".dropped.jsonl")
    with open(log, "w") as f:
        for r in dropped:
            f.write(json.dumps({"id": r.get("id"), "why": r["decontam"],
                                "question": (r.get("question") or r["prompt"][-400:])[:400]}) + "\n")
    print(f"{src.name}: kept {len(kept)}, dropped {len(dropped)} (log: {log.name})")
    for r in dropped[:10]:
        print("  -", r["decontam"], "|", (r.get("question") or "")[:120].replace("\n", " "))


if __name__ == "__main__":
    main()
