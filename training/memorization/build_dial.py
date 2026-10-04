"""Generalization-vs-memorization dial: training sets that move progressively closer to the TPC-H test.

THIS DELIBERATELY TRAINS ON THE TEST. Models trained on these sets are contaminated by design and are
reported only on the trade-off curve, never as clean TPC-H results. The clean track lives in
training/gen_tpch_questions.py + decontam.py.

Levels (each written to training/data/dial_<level>.jsonl, rows in the sql-opd taskset format):
  variants    the 22 question templates with new constants (dates, regions, sizes, segments ...);
              the 397B writes each variant, answers it twice, keep if both answers run, agree, non-empty
  paraphrase  the 22 questions reworded with identical meaning, constants and output columns; keep if
              the 397B's answer to the paraphrase matches the reference result (so meaning is preserved)
  exact       the 22 eval prompts verbatim, with the reference SQL

  python training/memorization/build_dial.py --k 8
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tpch_eval"))
import eval as tpch  # noqa: E402
from prompting import build_prompt  # noqa: E402
from questions import QUESTIONS  # noqa: E402

DATA = ROOT / "training" / "data"

VARIANT_SYSTEM = (
    "You rewrite analytical business questions. Keep the exact same analysis (same tables, joins, grouping, "
    "ordering, limits and output columns) but change every constant to a different valid value that exists in "
    "the data: dates or date ranges, region/nation names, market segments, brands, containers, sizes, types, "
    "ship modes, quantities and thresholds. Keep the 'Output columns:' line unchanged. Reply with only the new "
    "question text."
)
PARAPHRASE_SYSTEM = (
    "You reword analytical business questions. The reworded question must mean exactly the same thing: same "
    "constants, filters, grouping, ordering, limits and the identical 'Output columns:' line. Change the "
    "wording and sentence structure substantially, as a different analyst would phrase it. Reply with only the "
    "question text."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="variants and paraphrases written per question")
    ap.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    con = tpch.connect(tpch.ensure_data(0.01))
    schema = tpch.schema_text(con)
    ref = {q: (ROOT / "tpch_eval" / "reference" / f"q{q:02d}.sql").read_text() for q in QUESTIONS}
    ref_rows = {q: tpch.run_sql(con, ref[q]) for q in QUESTIONS}

    # exact: the 22 eval prompts verbatim
    with open(DATA / "dial_exact.jsonl", "w") as f:
        for q, text in QUESTIONS.items():
            f.write(json.dumps({"id": f"dial-exact-q{q:02d}", "source": "dial_exact", "base_q": q, "question": text,
                                "prompt": build_prompt(schema, text), "teacher_sql": ref[q]}) + "\n")

    answer = tpch.make_generator(argparse.Namespace(backend="tinker", model=args.model, renderer=None,
                                                    max_tokens=8192, temperature=0.7))
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.renderers.base import get_text_content
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(args.model), get_tokenizer(args.model))
    client = tinker.ServiceClient().create_sampling_client(base_model=args.model)
    params = tinker.types.SamplingParams(max_tokens=8192, temperature=1.0, stop=renderer.get_stop_sequences())

    def rewrite(system, q, i):
        user = f"Schema:\n\n{schema}\n\nQuestion:\n{QUESTIONS[q]}\n\n(Rewrite #{i + 1}; make it differ from other rewrites.)"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = client.sample(renderer.build_generation_prompt(msgs), sampling_params=params, num_samples=1).result()
        msg, _ = renderer.parse_response(out.sequences[0].tokens)
        return get_text_content(msg).strip()

    def build(job):
        level, q, i = job
        try:
            text = rewrite(VARIANT_SYSTEM if level == "variants" else PARAPHRASE_SYSTEM, q, i)
            if "Output columns:" not in text or text.strip() == QUESTIONS[q].strip():
                return None
            prompt = build_prompt(schema, text)
            s1 = tpch.extract_sql(answer(prompt))
            r1 = tpch.run_sql(con, s1)
            if level == "paraphrase":
                ok, _ = tpch.results_match(r1, ref_rows[q])
                sql = ref[q]
            else:
                s2 = tpch.extract_sql(answer(prompt))
                ok, _ = tpch.results_match(r1, tpch.run_sql(con, s2))
                ok = ok and 0 < len(r1) <= 1000
                sql = s1
            return {"id": f"dial-{level}-q{q:02d}-{i}", "source": f"dial_{level}", "base_q": q, "question": text,
                    "prompt": prompt, "teacher_sql": sql, "kept": bool(ok)}
        except Exception as e:  # noqa: BLE001
            return {"source": f"dial_{level}", "base_q": q, "error": str(e)[:200], "kept": False}

    jobs = [(lvl, q, i) for lvl in ("variants", "paraphrase") for q in QUESTIONS for i in range(args.k)]
    random.Random(args.seed).shuffle(jobs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = [r for r in pool.map(build, jobs) if r]
    for lvl in ("variants", "paraphrase"):
        kept = [r for r in rows if r["source"] == f"dial_{lvl}" and r["kept"]]
        with open(DATA / f"dial_{lvl}.jsonl", "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        per_q = {q: sum(r["base_q"] == q for r in kept) for q in QUESTIONS}
        print(f"{lvl}: kept {len(kept)}/{len(QUESTIONS) * args.k}; per question {per_q}")


if __name__ == "__main__":
    main()
