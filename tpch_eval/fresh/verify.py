"""Verify the fresh TPC-H test sets.

Checks every gold_sql in fresh_questions.json and probe_questions.json on the SF 0.01 data:
  - the query runs and returns a non-empty, non-trivial result (not a single all-NULL row)
  - the result has the number of columns listed on the question's "Output columns:" line
  - the result has at most MAX_ROWS rows
  - if the query ends in ORDER BY ... LIMIT k, the cutoff does not fall inside a tie
    (row k and row k+1 of the unlimited result differ on the ORDER BY keys)
  - fresh questions: ids/fields are well formed and the result differs from every standard query's result
  - probes: one per standard query, and the result differs from the original reference query

Usage: python tpch_eval/fresh/verify.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from eval import connect, results_match, run_sql, norm  # noqa: E402
from questions import QUESTIONS  # noqa: E402

DATA = ROOT / "data" / "sf0.01"
MAX_ROWS = 500

LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.I)


def output_columns(question: str) -> list[str]:
    m = re.search(r"Output columns:\s*(.+?)\s*$", question.strip())
    if not m:
        return []
    return [c.strip() for c in m.group(1).split(",")]


def order_by_items(sql: str) -> list[str]:
    """ORDER BY items (names only, ASC/DESC stripped) of the outermost/last ORDER BY clause."""
    idx = [m.start() for m in re.finditer(r"\bORDER\s+BY\b", sql, re.I)]
    if not idx:
        return []
    tail = sql[idx[-1]:]
    tail = re.sub(r"^ORDER\s+BY", "", tail, flags=re.I)
    tail = LIMIT_RE.sub("", tail)
    items = []
    for it in tail.split(","):
        it = re.sub(r"\s+(ASC|DESC)\b.*$", "", it.strip(), flags=re.I | re.S).strip()
        items.append(it)
    return items


def limit_tie_problem(con, sql: str) -> str | None:
    """Return a message if a trailing LIMIT cuts inside a tie, else None."""
    m = LIMIT_RE.search(sql)
    if not m:
        return None
    k = int(m.group(1))
    unlimited = LIMIT_RE.sub("", sql).rstrip().rstrip(";")
    cur = con.cursor()
    rows = cur.execute(unlimited).fetchall()
    cols = [d[0] for d in cur.description]
    cur.close()
    if len(rows) <= k:
        return None  # LIMIT does not cut anything
    keys = order_by_items(unlimited)
    pos = []
    for key in keys:
        name = key.split(".")[-1].strip()
        if name.isdigit():
            pos.append(int(name) - 1)
        elif name in cols:
            pos.append(cols.index(name))
        else:
            return f"cannot map ORDER BY item {key!r} to an output column {cols}"
    a = tuple(norm(rows[k - 1][i]) for i in pos)
    b = tuple(norm(rows[k][i]) for i in pos)
    if a == b:
        return f"tie at LIMIT {k}: row {k} and row {k + 1} share ORDER BY keys {a}"
    return None


def check_result(con, item, sql, question) -> tuple[list[str], list[tuple]]:
    errs = []
    try:
        rows = run_sql(con, sql)
    except Exception as e:  # noqa: BLE001
        return [f"error: {e}"], []
    if not rows:
        errs.append("empty result")
    elif len(rows) == 1 and all(v is None for v in rows[0]):
        errs.append("trivial result (single all-NULL row)")
    if len(rows) > MAX_ROWS:
        errs.append(f"{len(rows)} rows > {MAX_ROWS}")
    cols = output_columns(question)
    if not cols:
        errs.append("question does not end with an 'Output columns:' line")
    elif rows and len(rows[0]) != len(cols):
        errs.append(f"result has {len(rows[0])} columns, question lists {len(cols)}")
    tie = limit_tie_problem(con, sql)
    if tie:
        errs.append(tie)
    return errs, rows


def main() -> int:
    con = connect(DATA)
    refs = {n: run_sql(con, (ROOT / "reference" / f"q{n:02d}.sql").read_text()) for n in QUESTIONS}
    failures = 0

    fresh = json.loads((HERE / "fresh_questions.json").read_text())
    print(f"== fresh_questions.json: {len(fresh)} items")
    ids = [f["id"] for f in fresh]
    if len(set(ids)) != len(ids):
        print("  FAIL duplicate ids")
        failures += 1
    for it in fresh:
        missing = [k for k in ("id", "question", "gold_sql", "skills", "why_hard") if not it.get(k)]
        errs, rows = check_result(con, it, it.get("gold_sql", ""), it.get("question", ""))
        if missing:
            errs.append(f"missing fields {missing}")
        same = [n for n, r in refs.items() if rows and results_match(rows, r)[0]]
        if same:
            errs.append(f"result identical to standard query {same}")
        status = "ok  " if not errs else "FAIL"
        failures += bool(errs)
        print(f"  {status} {it['id']}: {len(rows):4d} rows" + ("" if not errs else "  <- " + "; ".join(errs)))

    probes = json.loads((HERE / "probe_questions.json").read_text())
    print(f"== probe_questions.json: {len(probes)} items")
    covered = sorted(p["base_q"] for p in probes)
    if covered != sorted(QUESTIONS):
        print(f"  FAIL probes cover {covered}, expected 1..22 once each")
        failures += 1
    for it in probes:
        errs, rows = check_result(con, it, it["gold_sql"], it["question"])
        if it["question"].strip() == QUESTIONS[it["base_q"]].strip():
            errs.append("question identical to the original")
        orig = refs[it["base_q"]]
        if rows and results_match(rows, orig)[0]:
            errs.append("result identical to the original reference query")
        status = "ok  " if not errs else "FAIL"
        failures += bool(errs)
        print(f"  {status} {it['id']} (q{it['base_q']:02d}): {len(rows):4d} rows vs original {len(orig):4d}"
              + ("" if not errs else "  <- " + "; ".join(errs)))

    total = len(fresh) + len(probes)
    print(f"\nSUMMARY: {total - failures}/{total} items pass" + (" -- ALL CHECKS PASSED" if not failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
