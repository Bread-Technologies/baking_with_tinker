"""Convert Spider SQLite databases to DuckDB files and describe them as DDL.

Prompts use DDL generated from the converted DuckDB tables, so the column
types the model sees are the ones its SQL actually runs against.
"""

import sqlite3
from pathlib import Path

import duckdb
import pandas as pd

DUCK_DIR = Path(__file__).parent / "data" / "spider_duckdb"


def convert(sqlite_path: str) -> Path:
    src = Path(sqlite_path)
    dst = DUCK_DIR / f"{src.parent.parent.name}__{src.stem}.duckdb"
    if dst.exists():
        return dst
    DUCK_DIR.mkdir(parents=True, exist_ok=True)
    lite = sqlite3.connect(src)
    lite.text_factory = lambda b: b.decode("utf-8", errors="replace")
    tmp = dst.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    duck = duckdb.connect(str(tmp))
    tables = [r[0] for r in lite.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in tables:
        df = pd.read_sql_query(f'SELECT * FROM "{t}"', lite)
        df.columns = [str(c) for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                # SQLite columns can mix types ('' alongside numbers); make each column one type.
                col = df[c].replace("", None)
                num = pd.to_numeric(col, errors="coerce")
                if num.notna().sum() == col.notna().sum():
                    df[c] = num
                else:
                    # Text column: keep '' as '' so gold SQL (run in SQLite) and predictions see the same values.
                    df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        duck.register("df_view", df)
        duck.execute(f'CREATE TABLE "{t}" AS SELECT * FROM df_view')
        duck.unregister("df_view")
    duck.close()
    lite.close()
    tmp.rename(dst)
    return dst


def ddl(duck_path: Path, spider_tables: dict | None = None) -> str:
    """CREATE TABLE text from the DuckDB file, with Spider's PK/FK annotations when available."""
    con = duckdb.connect(str(duck_path), read_only=True)
    pks, fks = set(), {}
    if spider_tables:
        cols = spider_tables["column_names_original"]
        names = spider_tables["table_names_original"]
        for k in spider_tables["primary_keys"]:
            for ci in (k if isinstance(k, list) else [k]):
                pks.add((names[cols[ci][0]].lower(), cols[ci][1].lower()))
        for a, b in spider_tables["foreign_keys"]:
            fks[(names[cols[a][0]].lower(), cols[a][1].lower())] = (names[cols[b][0]], cols[b][1])
    out = []
    for (t,) in con.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall():
        lines = []
        for name, typ, *_ in con.execute(f'DESCRIBE "{t}"').fetchall():
            line = f'  "{name}" {typ}'
            if (t.lower(), name.lower()) in pks:
                line += " PRIMARY KEY"
            if (t.lower(), name.lower()) in fks:
                rt, rc = fks[(t.lower(), name.lower())]
                line += f' REFERENCES "{rt}"("{rc}")'
            lines.append(line)
        out.append(f'CREATE TABLE "{t}" (\n' + ",\n".join(lines) + "\n);")
    con.close()
    return "\n\n".join(out)
