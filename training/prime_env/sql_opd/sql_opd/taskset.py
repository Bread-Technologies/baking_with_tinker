"""sql-opd: single-turn text-to-SQL over JSONL prompts (from training/build_data.py etc.).

Each JSONL row has `prompt` (schema + question, the eval's exact format) and optionally a
reference to score against: `gold_sql` + `sqlite_path` (Spider; gold runs in SQLite),
or `teacher_sql` (target track; runs in DuckDB on the TPC-H file). The prediction runs in
DuckDB (`duckdb_path`, or the TPC-H file). Reward is 1 on an execution match, else 0.

For on-policy distillation the reward is not a training signal (the teacher's KL is);
it is logged so train/eval accuracy can be tracked. Rows with no reference score 0.
"""

import json
import math
import re
import sqlite3
import threading
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import verifiers.v1 as vf

SYSTEM_PROMPT = (
    "You are an expert SQL analyst. You write a single DuckDB SQL query that answers the "
    "user's question against the given schema. Return exactly the requested output columns, "
    "in the requested order. Reply with only the SQL query inside a ```sql code block."
)
TIMEOUT_S = 30


def _norm(v):
    if v is None:
        return None
    if isinstance(v, (bool, int, float, Decimal)):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()[:10]
    return str(v).strip()


def _key(row):
    return tuple((0, x, "") if isinstance(x, float) else (1, 0.0, "" if x is None else x) for x in row)


def _match(got, gold) -> bool:
    if len(got) != len(gold) or (gold and len(got[0]) != len(gold[0])):
        return False
    g = sorted([tuple(_norm(x) for x in r) for r in got], key=_key)
    e = sorted([tuple(_norm(x) for x in r) for r in gold], key=_key)
    for rg, re_ in zip(g, e):
        for a, b in zip(rg, re_):
            if isinstance(a, float) and isinstance(b, float):
                if not math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6):
                    return False
            elif a != b:
                return False
    return True


def _extract_sql(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    blocks = re.findall(r"```(?:sql|SQL)?\s*\n(.*?)```", text, flags=re.S)
    return (blocks[-1] if blocks else text).strip().rstrip(";").strip()


def _run_duckdb(path: str, sql: str):
    con = duckdb.connect(path, read_only=True)
    timer = threading.Timer(TIMEOUT_S, con.interrupt)
    timer.start()
    try:
        return con.execute(sql).fetchall()
    finally:
        timer.cancel()
        con.close()


def _run_sqlite(path: str, sql: str):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    timer = threading.Timer(TIMEOUT_S, con.interrupt)
    timer.start()
    try:
        return con.execute(sql).fetchall()
    finally:
        timer.cancel()
        con.close()


class SqlData(vf.TaskData):
    gold_sql: str | None = None
    teacher_sql: str | None = None
    sqlite_path: str | None = None
    duckdb_path: str | None = None


class SqlTask(vf.Task[SqlData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def execution_match(self, trace: vf.Trace) -> float:
        d = self.data
        if not d.duckdb_path or not (d.gold_sql or d.teacher_sql):
            return 0.0
        try:
            if d.gold_sql and d.sqlite_path:
                gold = _run_sqlite(d.sqlite_path, d.gold_sql)
            else:
                gold = _run_duckdb(d.duckdb_path, d.teacher_sql)
            got = _run_duckdb(d.duckdb_path, _extract_sql(trace.last_reply))
        except Exception:  # noqa: BLE001
            return 0.0
        return 1.0 if _match(got, gold) else 0.0


class SqlTasksetConfig(vf.TasksetConfig):
    data_path: str = "/data/proxy_train.jsonl"
    """JSONL file(s), comma-separated."""
    data_root: str = "/data"
    """Where training/data/ is mounted; absolute paths in the JSONL are re-rooted here."""
    tpch_duckdb: str = "/data/tpch_sf0.01.duckdb"
    """DuckDB file for rows on the TPC-H schema (target track)."""


class SqlTaskset(vf.Taskset[SqlTask, SqlTasksetConfig]):
    def _reroot(self, p: str | None) -> str | None:
        if not p:
            return None
        marker = "training/data/"
        return str(Path(self.config.data_root) / p.split(marker, 1)[1]) if marker in p else p

    def load(self) -> list[SqlTask]:
        tasks = []
        for path in self.config.data_path.split(","):
            for line in open(path):
                r = json.loads(line)
                duck = self._reroot(r.get("duckdb_path"))
                if r.get("teacher_sql") and not duck:
                    duck = self.config.tpch_duckdb
                tasks.append(SqlTask(SqlData(
                    prompt=r["prompt"],
                    system_prompt=SYSTEM_PROMPT,
                    gold_sql=r.get("gold_sql") if r.get("sqlite_path") else None,
                    teacher_sql=r.get("teacher_sql"),
                    sqlite_path=self._reroot(r.get("sqlite_path")),
                    duckdb_path=duck,
                ), self.config.task))
        return tasks
