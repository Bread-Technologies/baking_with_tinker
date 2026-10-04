"""Build training prompts and held-out dev/test sets for TPC-H distillation.

Nothing here reads tpch_eval/questions.py or tpch_eval/reference/; the only
step allowed to look at the TPC-H test is decontam.py.

Outputs (training/data/):
  proxy_train.jsonl   BIRD train + Spider train prompts (no TPC-H schema anywhere)
  spider_dev.jsonl    Spider dev: held-out databases, executable gold -> dev metric
  spider_test.jsonl   Spider test: databases unseen by train and dev -> far-transfer metric

Each row: {"id", "source", "db_id", "prompt", "gold_sql"?, "sqlite_path"?}
"""

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
DATA = HERE / "data"
SPIDER = DATA / "spider" / "spider_data"

sys.path.insert(0, str(HERE.parent / "tpch_eval"))
from prompting import build_prompt  # noqa: E402

sys.path.insert(0, str(HERE))
from spider_duckdb import convert, ddl  # noqa: E402


def bird_rows():
    p = hf_hub_download("xu3kev/BIRD-SQL-data-train", "data/train-00000-of-00001-fe8894d41b7815be.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    for i, r in df.iterrows():
        question = r["question"].strip()
        if r["evidence"] and r["evidence"].strip():
            question += f"\nHint: {r['evidence'].strip()}"
        yield {"id": f"bird-{i}", "source": "bird_train", "db_id": r["db_id"],
               "prompt": build_prompt(r["schema"].strip(), question), "gold_sql": r["SQL"]}


def spider_rows(split_files, source, db_dir):
    tables = {t["db_id"]: t for t in json.load(open(SPIDER / ("test_tables.json" if source == "spider_test" else "tables.json")))}
    rows = []
    for f in split_files:
        rows += json.load(open(SPIDER / f))
    schemas, duck_paths = {}, {}
    for i, r in enumerate(rows):
        db = r["db_id"]
        if db not in schemas:
            duck_paths[db] = convert(str(db_dir / db / f"{db}.sqlite"))
            schemas[db] = ddl(duck_paths[db], tables.get(db))
        yield {"id": f"{source}-{i}", "source": source, "db_id": db,
               "prompt": build_prompt(schemas[db], r["question"].strip()),
               "gold_sql": r["query"], "sqlite_path": str(db_dir / db / f"{db}.sqlite"),
               "duckdb_path": str(duck_paths[db])}


def write(name, rows):
    path = DATA / name
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} rows")


def main():
    random.seed(0)
    train = list(bird_rows()) + list(spider_rows(["train_spider.json", "train_others.json"], "spider_train",
                                                 SPIDER / "database"))
    random.shuffle(train)
    write("proxy_train.jsonl", train)

    dev = list(spider_rows(["dev.json"], "spider_dev", SPIDER / "database"))
    write("spider_dev.jsonl", dev)

    # Spider test also contains the dev databases; keep only databases unseen by both train and dev.
    seen = {r["db_id"] for r in train} | {r["db_id"] for r in dev}
    test = [r for r in spider_rows(["test.json"], "spider_test", SPIDER / "test_database") if r["db_id"] not in seen]
    write("spider_test.jsonl", test)
    print("dbs: train", len({r['db_id'] for r in train}), "dev", len({r['db_id'] for r in dev}),
          "test", len({r['db_id'] for r in test}))


if __name__ == "__main__":
    main()
