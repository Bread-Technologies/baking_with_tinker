"""Execution-scored text-to-SQL eval on held-out Spider databases (dev / far-transfer test).

Predictions run in DuckDB on the converted database; gold SQL runs in SQLite on the
original file. A prediction passes if both give the same multiset of rows (same
normalization and numeric tolerance as tpch_eval). This set drives every hill-climbing
decision; the TPC-H sets are only touched at pre-chosen checkpoints.

  python training/sql_eval.py --split spider_dev --backend gold          # grader sanity check
  python training/sql_eval.py --split spider_dev --backend openai --model Qwen/Qwen3.5-2B --base-url .../v1 --n 300
"""

import argparse
import json
import random
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "tpch_eval"))
import eval as tpch  # noqa: E402

DATA = HERE / "data"
TIMEOUT_S = 30


def run_sqlite(path: str, sql: str):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    timer = threading.Timer(TIMEOUT_S, con.interrupt)
    timer.start()
    try:
        return con.execute(sql).fetchall()
    finally:
        timer.cancel()
        con.close()


def run_duckdb(path: str, sql: str):
    con = duckdb.connect(path, read_only=True)
    try:
        return tpch.run_sql(con, sql)
    finally:
        con.close()


def score(row: dict, response: str) -> tuple[bool, str, str]:
    sql = tpch.extract_sql(response)
    try:
        gold = run_sqlite(row["sqlite_path"], row["gold_sql"])
    except Exception as e:  # noqa: BLE001
        return False, sql, f"gold error: {e}"
    try:
        got = run_duckdb(row["duckdb_path"], sql)
    except Exception as e:  # noqa: BLE001
        return False, sql, f"sql error: {str(e).splitlines()[0][:200]}"
    ok, why = tpch.results_match(got, gold)
    return ok, sql, why


def load(split: str, n: int | None, seed: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in open(DATA / f"{split}.jsonl")]
    if n and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="spider_dev", choices=["spider_dev", "spider_test"])
    ap.add_argument("--backend", required=True, choices=["claude", "tinker", "openai", "hf", "gold"])
    ap.add_argument("--model", default="gold")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--renderer")
    ap.add_argument("--thinking", choices=["on", "off"])
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--n", type=int, help="random subset size (fixed seed)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = load(args.split, args.n)
    if args.backend == "gold":
        def gen(row):
            return "```sql\n" + row["gold_sql"] + "\n```"
    else:
        g = tpch.make_generator(args)

        def gen(row):
            return g(row["prompt"])

    def work(row):
        try:
            response = gen(row)
        except Exception as e:  # noqa: BLE001
            return {"id": row["id"], "correct": False, "reason": f"generation error: {e}", "sql": "", "response": ""}
        ok, sql, why = score(row, response)
        return {"id": row["id"], "db_id": row["db_id"], "correct": ok, "reason": why, "sql": sql, "response": response}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(work, rows))

    k = sum(r["correct"] for r in results)
    n = len(results)
    p = k / n
    se = (p * (1 - p) / n) ** 0.5
    print(f"{args.model} on {args.split}: {k}/{n} = {p:.3f} ± {1.96 * se:.3f} (95% CI)")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{args.split}__{args.backend}__{args.model.replace('/', '_')}{args.tag}.json"
    path.write_text(json.dumps({"model": args.model, "split": args.split, "n": n, "score": k, "acc": p,
                                "thinking": args.thinking, "results": results}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
