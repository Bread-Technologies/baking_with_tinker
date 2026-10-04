"""Target track: generate NEW analytical questions over the TPC-H schema.

Walled off from the test: this script never reads tpch_eval/questions.py or
tpch_eval/reference/. Inputs are the schema, a generic SQL skill list, a business
role, and constants sampled from the generated data. Leakage the generator brings
in from pre-training (it has seen TPC-H) is caught afterwards by decontam.py.

Pipeline:
  1. 397B writes a question (with an explicit "Output columns:" line, like the eval format).
  2. 397B answers it twice (thinking on). Keep if both SQLs run, agree, and the result
     is non-empty and not trivially small/huge.
Output: training/data/target_train_raw.jsonl
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "tpch_eval"))
import eval as tpch  # noqa: E402
from prompting import build_prompt  # noqa: E402

# Generic analytical-SQL skills (not derived from the TPC-H queries).
SKILLS = [
    "multi-table join with grouped aggregation",
    "correlated subquery comparing each row to an aggregate over related rows",
    "EXISTS / NOT EXISTS semi-join or anti-join",
    "HAVING filter against a threshold computed by a scalar subquery",
    "top-N ranking with ORDER BY and LIMIT",
    "finding the row(s) with the maximum or minimum of a computed metric (keep ties)",
    "LEFT JOIN that must keep entities with zero matching rows",
    "conditional aggregation (CASE inside SUM/COUNT) to compare categories side by side",
    "ratio or percentage of a subtotal over a total",
    "date-range filtering and grouping by year or month",
    "self-join comparing rows of the same table",
    "window function: rank within groups",
    "window function: running total or period-over-period change with LAG",
    "set operation (UNION / INTERSECT / EXCEPT) across two populations",
    "two-level aggregation: aggregate of per-group aggregates",
    "string pattern matching combined with joins",
    "distribution / histogram: count entities per bucket of a per-entity metric",
    "date arithmetic between two date columns (e.g., delays, durations)",
]

ROLES = ["procurement analyst", "sales operations manager", "logistics planner", "finance controller",
         "customer success lead", "supply-chain risk analyst", "marketing analyst", "warehouse manager",
         "regional director", "pricing analyst", "auditor", "data scientist"]

GEN_SYSTEM = (
    "You write realistic business questions for a text-to-SQL dataset. Each question must be answerable "
    "with one DuckDB SQL query over the given schema, be unambiguous (state every filter, threshold, date "
    "range and tie rule explicitly), and end with a line 'Output columns: a, b, c' naming the exact result "
    "columns in order, plus the sort order if any. Do NOT reproduce or paraphrase any standard benchmark "
    "query (e.g., the TPC-H benchmark queries); invent a new question a real analyst would ask. Reply with "
    "only the question text."
)


def sample_constants(con, rng: random.Random) -> str:
    """A few real values from the data so questions reference things that exist."""
    picks = []
    for sql, label in [
        ("SELECT n_name FROM nation", "nation"), ("SELECT r_name FROM region", "region"),
        ("SELECT DISTINCT c_mktsegment FROM customer", "market segment"),
        ("SELECT DISTINCT p_brand FROM part", "brand"), ("SELECT DISTINCT p_container FROM part", "container"),
        ("SELECT DISTINCT l_shipmode FROM lineitem", "ship mode"),
        ("SELECT DISTINCT o_orderpriority FROM orders", "order priority"),
        ("SELECT DISTINCT split_part(p_type, ' ', 3) FROM part", "material"),
    ]:
        vals = [r[0] for r in con.execute(sql).fetchall()]
        picks.append(f"{label}: {rng.choice(vals)}")
    year = rng.randint(1992, 1998)
    picks.append(f"a year: {year}")
    return "; ".join(rng.sample(picks, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="questions to generate")
    ap.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "data" / "target_train_raw.jsonl"))
    args = ap.parse_args()

    con = tpch.connect(tpch.ensure_data(0.01))
    schema = tpch.schema_text(con)
    rng = random.Random(args.seed)
    specs = [{"skill": rng.choice(SKILLS), "role": rng.choice(ROLES), "consts": sample_constants(con, rng)}
             for _ in range(args.n)]

    gen_args = argparse.Namespace(backend="tinker", model=args.model, renderer=None, max_tokens=8192, temperature=0.7)
    answer = tpch.make_generator(gen_args)  # uses the eval SYSTEM_PROMPT: same format the student sees

    # Question writer: same model, different system prompt.
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.renderers.base import get_text_content
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(args.model), get_tokenizer(args.model))
    client = tinker.ServiceClient().create_sampling_client(base_model=args.model)
    params = tinker.types.SamplingParams(max_tokens=8192, temperature=1.0, stop=renderer.get_stop_sequences())

    def write_question(spec):
        user = (f"Schema:\n\n{schema}\n\nWrite one question for a {spec['role']} that exercises this SQL skill: "
                f"{spec['skill']}. You may use at most two of these real values (fewer filters is fine): {spec['consts']}.")
        msgs = [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}]
        out = client.sample(renderer.build_generation_prompt(msgs), sampling_params=params, num_samples=1).result()
        msg, _ = renderer.parse_response(out.sequences[0].tokens)
        return get_text_content(msg).strip()

    def build(spec):
        try:
            q = write_question(spec)
            if "Output columns:" not in q:
                return None
            prompt = build_prompt(schema, q)
            a1, a2 = answer(prompt), answer(prompt)
            s1, s2 = tpch.extract_sql(a1), tpch.extract_sql(a2)
            r1, r2 = tpch.run_sql(con, s1), tpch.run_sql(con, s2)  # run_sql uses a per-call cursor
            agree, _ = tpch.results_match(r1, r2)
            ok = agree and 0 < len(r1) <= 1000 and not all(v is None for row in r1 for v in row)
            return {"id": f"tpch-gen-{abs(hash(q)) % 10**10}", "source": "tpch_generated", **spec,
                    "question": q, "prompt": prompt, "teacher_sql": s1, "teacher_sql_2": s2,
                    "rows": len(r1), "kept": ok}
        except Exception as e:  # noqa: BLE001
            return {"source": "tpch_generated", **spec, "error": str(e)[:200], "kept": False}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = [r for r in pool.map(build, specs) if r]
    kept = sum(r["kept"] for r in rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"generated {len(rows)}, kept {kept} -> {args.out}")


if __name__ == "__main__":
    main()
