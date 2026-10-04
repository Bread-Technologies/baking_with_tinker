"""TPC-H text-to-SQL baseline: score a model out of 22.

The model gets the schema and a natural-language version of each TPC-H query
(questions.py) and must write one DuckDB SQL query. A query scores 1 if its
result matches the official TPC-H query (reference/qNN.sql) run on the same
data: same row count and the same multiset of rows, with numeric tolerance.
One attempt per question, greedy decoding.

Backends:
  claude   Anthropic API              (ANTHROPIC_API_KEY)
  tinker   Tinker sampling + renderer (TINKER_API_KEY)
  openai   OpenAI-compatible server, e.g. vLLM on Modal (--base-url)
  hf       local HuggingFace transformers (greedy; CPU is fine for ~1.5B)
  file     pre-generated responses from a JSON file (--responses), e.g. a Claude Code subagent
  gold     returns the reference SQL; sanity-checks the scorer (expect 22/22)

Examples:
  python tpch_eval/eval.py --backend tinker --model Qwen/Qwen3.5-397B-A17B
  python tpch_eval/eval.py --backend claude --model claude-opus-5-5
  python tpch_eval/eval.py --backend openai --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --base-url https://<workspace>--tpch-qwen-coder-serve.modal.run/v1
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import threading
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from questions import QUESTIONS  # noqa: E402

ROOT = Path(__file__).parent
REPO = ROOT.parent
load_dotenv(REPO / "care package" / ".env")

TABLES = ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]

SYSTEM_PROMPT = (
    "You are an expert SQL analyst. You write a single DuckDB SQL query that answers the "
    "user's question against the TPC-H schema. Return exactly the requested output columns, "
    "in the requested order. Reply with only the SQL query inside a ```sql code block."
)

QUERY_TIMEOUT_S = 60


# ---------------------------------------------------------------- data

def ensure_data(sf: float) -> Path:
    data_dir = ROOT / "data" / f"sf{sf:g}"
    if all((data_dir / f"{t}.parquet").exists() for t in TABLES):
        return data_dir
    if not shutil.which("tpchgen-cli"):
        sys.exit("tpchgen-cli not found: pip install tpchgen-cli")
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tpchgen-cli", "-s", str(sf), "--format", "parquet", "-o", str(data_dir)], check=True)
    return data_dir


def connect(data_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in TABLES:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{data_dir / (t + '.parquet')}')")
    return con


def schema_text(con: duckdb.DuckDBPyConnection) -> str:
    parts = []
    for t in TABLES:
        cols = con.execute(f"DESCRIBE {t}").fetchall()
        parts.append(f"CREATE TABLE {t} (\n" + ",\n".join(f"  {c[0]} {c[1]}" for c in cols) + "\n);")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- scoring

def run_sql(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    """Run sql on a cursor, interrupting it after QUERY_TIMEOUT_S."""
    cur = con.cursor()
    timer = threading.Timer(QUERY_TIMEOUT_S, cur.interrupt)
    timer.start()
    try:
        return cur.execute(sql).fetchall()
    finally:
        timer.cancel()
        cur.close()


def norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float, Decimal)):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()[:10]
    return str(v).strip()


def sort_key(row):
    return tuple((0, x, "") if isinstance(x, float) else (1, 0.0, "" if x is None else x) for x in row)


def values_match(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6)
    return a == b


def results_match(got: list[tuple], gold: list[tuple]) -> tuple[bool, str]:
    if len(got) != len(gold):
        return False, f"row count {len(got)} != {len(gold)}"
    if gold and len(got[0]) != len(gold[0]):
        return False, f"column count {len(got[0])} != {len(gold[0])}"
    g = sorted([tuple(norm(x) for x in r) for r in got], key=sort_key)
    e = sorted([tuple(norm(x) for x in r) for r in gold], key=sort_key)
    for i, (rg, re_) in enumerate(zip(g, e)):
        if not all(values_match(a, b) for a, b in zip(rg, re_)):
            return False, f"row mismatch: got {rg} expected {re_}"
    return True, "ok"


def extract_sql(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:sql|SQL)?\s*\n(.*?)```", text, flags=re.S)
    sql = blocks[-1] if blocks else text
    return sql.strip().rstrip(";").strip()


# ---------------------------------------------------------------- backends

def make_generator(args):
    if args.backend == "claude":
        import anthropic

        client = anthropic.Anthropic()

        def gen(prompt: str) -> str:
            resp = client.messages.create(
                model=args.model,
                max_tokens=args.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        return gen

    if args.backend == "openai":
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY")
        # vLLM: toggle thinking for hybrid models (small Qwen3.5 models default to non-thinking)
        extra = {} if args.thinking is None else {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": args.thinking == "on"}}}

        def gen(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=0.0,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                **extra,
            )
            msg = resp.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            return (f"<think>{reasoning}</think>\n" if reasoning else "") + (msg.content or "")

        return gen

    if args.backend == "tinker":
        import tinker
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.renderers.base import get_text_content
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        renderer_name = args.renderer or model_info.get_recommended_renderer_name(args.model)
        renderer = renderers.get_renderer(renderer_name, get_tokenizer(args.model))
        client = tinker.ServiceClient().create_sampling_client(base_model=args.model)
        params = tinker.types.SamplingParams(
            max_tokens=args.max_tokens, temperature=0.0, stop=renderer.get_stop_sequences()
        )
        print(f"tinker renderer: {renderer_name}")

        def gen(prompt: str) -> str:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            out = client.sample(renderer.build_generation_prompt(msgs), sampling_params=params, num_samples=1).result()
            msg, _ = renderer.parse_response(out.sequences[0].tokens)
            # Keep the reasoning trace in the saved response; extract_sql strips <think> blocks.
            parts = msg["content"] if isinstance(msg["content"], list) else []
            thinking = "".join(p["thinking"] for p in parts if p.get("type") == "thinking")
            text = get_text_content(msg)
            return f"<think>{thinking}</think>\n{text}" if thinking else text

        return gen

    if args.backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        model.eval()

        def gen(prompt: str) -> str:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_tokens, do_sample=False)
            return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

        return gen

    if args.backend == "file":
        # Pre-generated responses, e.g. from a Claude Code subagent: JSON {"1": "<response>", ...}
        responses = json.loads(Path(args.responses).read_text())
        by_prompt = {QUESTIONS[q]: responses[str(q)] for q in QUESTIONS if str(q) in responses}
        return lambda prompt: by_prompt[prompt.split("Question:\n", 1)[1]]

    if args.backend == "gold":
        # Sanity check: answer each question with its reference SQL (should score 22/22).
        refs = {QUESTIONS[q]: (ROOT / "reference" / f"q{q:02d}.sql").read_text() for q in QUESTIONS}
        return lambda prompt: "```sql\n" + refs[prompt.split("Question:\n", 1)[1]] + "```"

    raise ValueError(args.backend)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["claude", "tinker", "openai", "hf", "file", "gold"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", help="openai backend: server base URL ending in /v1")
    ap.add_argument("--api-key", help="openai backend: API key, if the server needs one")
    ap.add_argument("--responses", help="file backend: JSON mapping question number to response text")
    ap.add_argument("--dump-prompts", help="write {question number: full prompt} JSON here and exit")
    ap.add_argument("--renderer", help="tinker backend: override the recommended renderer")
    ap.add_argument("--sf", type=float, default=0.01, help="TPC-H scale factor")
    ap.add_argument("--thinking", choices=["on", "off"], help="openai backend: force thinking on/off (vLLM)")
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel generation requests")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--queries", default="1-22", help="e.g. 1-22 or 1,3,5")
    args = ap.parse_args()

    if "-" in args.queries:
        lo, hi = map(int, args.queries.split("-"))
        qids = list(range(lo, hi + 1))
    else:
        qids = [int(q) for q in args.queries.split(",")]

    con = connect(ensure_data(args.sf))
    schema = schema_text(con)
    if args.dump_prompts:
        prompts = {q: f"Schema:\n\n{schema}\n\nQuestion:\n{QUESTIONS[q]}" for q in qids}
        Path(args.dump_prompts).write_text(json.dumps({"system": SYSTEM_PROMPT, "prompts": prompts}, indent=2))
        print(f"wrote {len(prompts)} prompts to {args.dump_prompts}")
        return
    gen = make_generator(args)

    prompts = {q: f"Schema:\n\n{schema}\n\nQuestion:\n{QUESTIONS[q]}" for q in qids}

    def generate(q):
        try:
            return gen(prompts[q]), None
        except Exception as e:  # noqa: BLE001
            return "", f"generation error: {e}"

    # Generation can run in parallel (servers batch requests); scoring stays sequential on one connection.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        generated = dict(zip(qids, pool.map(generate, qids)))

    results = []
    for q in qids:
        gold = run_sql(con, (ROOT / "reference" / f"q{q:02d}.sql").read_text())
        response, err = generated[q]
        if err:
            sql, ok, why = "", False, err
        else:
            sql = extract_sql(response)
            try:
                ok, why = results_match(run_sql(con, sql), gold)
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"sql error: {str(e).splitlines()[0][:200]}"
        results.append({"q": q, "correct": ok, "reason": why, "sql": sql, "response": response})
        print(f"Q{q:02d} {'PASS' if ok else 'FAIL'}  {'' if ok else why}", flush=True)

    score = sum(r["correct"] for r in results)
    print(f"\n{args.model}: {score}/{len(results)}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.backend}__{args.model.replace('/', '_')}{args.tag}.json"
    out.write_text(json.dumps({"model": args.model, "backend": args.backend, "sf": args.sf,
                               "thinking": args.thinking, "max_tokens": args.max_tokens,
                               "score": score, "total": len(results), "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
