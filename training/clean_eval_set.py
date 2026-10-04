"""Keep only unambiguous, engine-portable questions in the held-out Spider sets.

A question is kept when its gold SQL, transpiled SQLite -> DuckDB with sqlglot, returns the
same rows in DuckDB as the original does in SQLite, the result is non-empty, and an
ORDER BY ... LIMIT does not cut through tied values. This removes items a correct answer
could fail for reasons outside the model's control (blog: ambiguous tasks / grader errors).
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sqlglot

sys.path.insert(0, str(Path(__file__).parent))
from sql_eval import DATA, run_duckdb, run_sqlite, tpch  # noqa: E402


def limit_ties(row, duck_sql):
    """True if ORDER BY ... LIMIT k cuts through ties (row k and k+1 share the sort key)."""
    m = re.search(r"\blimit\s+(\d+)\s*$", duck_sql, flags=re.I)
    if not m or not re.search(r"\border\s+by\b", duck_sql, flags=re.I):
        return False
    k = int(m.group(1))
    wider = re.sub(r"\blimit\s+\d+\s*$", f"LIMIT {k + 1}", duck_sql, flags=re.I)
    # Compare the ORDER BY key of rows k and k+1 by re-running with the key as the only output.
    order = re.split(r"\border\s+by\b", duck_sql, flags=re.I)[-1]
    order = re.sub(r"\blimit\s+\d+\s*$", "", order, flags=re.I)
    key = re.sub(r"\b(asc|desc)\b", "", order, flags=re.I)
    try:
        rows = run_duckdb(row["duckdb_path"], f"SELECT {key} FROM ({wider.replace(order, ' ' + order)}) _x")
    except Exception:  # noqa: BLE001
        try:
            rows = run_duckdb(row["duckdb_path"], wider)
            return len(rows) > k and rows[k - 1] == rows[k]
        except Exception:  # noqa: BLE001
            return True
    return len(rows) > k and rows[k - 1] == rows[k]


def check(row):
    try:
        duck_sql = sqlglot.transpile(row["gold_sql"], read="sqlite", write="duckdb")[0]
        gold = run_sqlite(row["sqlite_path"], row["gold_sql"])
        got = run_duckdb(row["duckdb_path"], duck_sql)
    except Exception as e:  # noqa: BLE001
        return row, False, f"error: {str(e)[:80]}"
    ok, why = tpch.results_match(got, gold)
    if not ok:
        return row, False, "engine mismatch"
    if not gold:
        return row, False, "empty result"
    if limit_ties(row, duck_sql):
        return row, False, "limit ties"
    return row, True, "ok"


def main():
    for split in ["spider_dev", "spider_test"]:
        rows = [json.loads(line) for line in open(DATA / f"{split}.jsonl")]
        with ThreadPoolExecutor(8) as pool:
            res = list(pool.map(check, rows))
        kept = [r for r, ok, _ in res if ok]
        from collections import Counter
        print(split, f"kept {len(kept)}/{len(rows)}", Counter(w.split(":")[0] for _, ok, w in res if not ok))
        with open(DATA / f"{split}_clean.jsonl", "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
