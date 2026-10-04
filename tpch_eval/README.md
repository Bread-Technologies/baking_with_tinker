# TPC-H text-to-SQL baseline

Scores a model out of 22. Each TPC-H query is given as a natural-language question with the standard validation parameters and the exact output columns (`questions.py`), plus the schema. The model writes one DuckDB query, with one attempt and greedy decoding. The query passes if its result matches the official TPC-H query (`reference/`, from DuckDB's tpch extension) run on the same data. Matching means the same row count and the same multiset of rows, with a relative numeric tolerance of 1e-4. Row order and column names are ignored. Column order matters.

Data is generated at scale factor 0.01 by default, using `tpchgen-cli`. Note that at SF 0.01 the gold answer for Q17 is a single NULL row.

```bash
pip install duckdb tpchgen-cli python-dotenv
python tpch_eval/eval.py --backend gold   --model reference                 # scorer sanity check: 22/22
python tpch_eval/eval.py --backend tinker --model Qwen/Qwen3.5-397B-A17B
python tpch_eval/eval.py --backend claude --model claude-opus-5-5
modal deploy tpch_eval/modal_serve.py                                        # Qwen2.5-Coder-1.5B-Instruct on vLLM
python tpch_eval/eval.py --backend openai --model Qwen/Qwen2.5-Coder-1.5B-Instruct --base-url https://<ws>--tpch-qwen-coder-serve.modal.run/v1
python tpch_eval/eval.py --backend hf     --model Qwen/Qwen2.5-Coder-1.5B-Instruct --max-tokens 1024  # local CPU; Modal's gRPC client cannot get through this sandbox's proxy
```

Per-query SQL, responses and failure reasons are written to `tpch_eval/results/`.
