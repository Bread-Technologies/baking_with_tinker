# Hill-climbing log: on-policy distillation into a small Qwen3.5 for TPC-H text-to-SQL

Goal: a Qwen3.5-2B (or 0.8B) LoRA that scores 22/22 on the TPC-H text-to-SQL test without
having trained on the test, with an honest account of how well it generalizes.

## Setup

- **Student:** Qwen/Qwen3.5-2B (also 0.8B), thinking on, sampling at temperature 1.0.
- **Teacher:** Qwen/Qwen3.5-397B-A17B on Tinker. It has the same tokenizer as the student and scores 22/22 on TPC-H.
- **Algorithm:** prime-rl `opd`, per-token reverse KL to the teacher. The teacher's log-probabilities come through `training/tinker_teacher_shim.py`.
- **Infrastructure:** prime-rl on Modal, 2×H100 (1 inference + 1 trainer). LoRAs and checkpoints are kept on the `tpch-opd-outputs` volume.

## Data tiers (the blog's train / dev / test split)

| Tier | What it is | Used for |
|---|---|---|
| train (proxy) | BIRD and Spider train prompts. No TPC-H schema. | training only |
| train (target) | New TPC-H-schema questions from `gen_tpch_questions.py`, decontaminated against the 22 | training only (target track) |
| **dev** | `spider_dev_clean`: 648 questions on 20 databases unseen in training | **every keep/revert decision** |
| far test | `spider_test_clean`: 1,414 questions on 40 more unseen databases | generalization report only |
| fresh TPC-H | `tpch_eval/fresh`: 30 new hand-written TPC-H questions | report only, at checkpoints chosen in advance |
| probe | 22 near-copies of the standard queries with the meaning changed | memorization check, report only |
| **TPC-H 22** | the target test | baseline, final, and checkpoints chosen in advance only |

## Rules

1. Make one change per round: data mix, learning rate, rank, steps, rollout length, group size, warm-up, or prompts.
2. Keep a change only if dev improves by more than its noise (95% CI). If only train improves, revert it.
3. Do not look at TPC-H-22, fresh TPC-H or probe results while choosing changes. Evaluate them only at checkpoints chosen in advance.
4. Never paste test failures into prompts or training data. Failure analysis uses dev failures only.
5. Before believing a score, read a sample of scored transcripts and check for grader errors.
6. Run every reported number with several samples at temperature 1.0, and report the mean and range.
7. Log every round below, including reverted ones.

## Baselines (TPC-H 22, before training)

| Model | Setting | Score |
|---|---|---|
| Qwen3.5-397B-A17B (teacher) | thinking, greedy | 22/22 |
| Claude Opus 5.5 (subagent) | closed-book | 21/22 |
| Qwen3.5-4B | thinking, t=1.0 ×4 | 11.5/22 (11–13) |
| Qwen3.5-4B | thinking, t=0.6 ×4 | 12.5/22 (11–14); Spider dev 74.7% (n=300, t=0.6), 73.7% ±5.0 (t=1.0) |
| Qwen3-8B | thinking, greedy | 15/22 |
| Qwen2.5-Coder-1.5B-Instruct | greedy | 4/22 |
| Qwen3.5-2B | no thinking, greedy | 3/22 |
| Qwen3.5-0.8B | no thinking, greedy | 1/22 |
| Qwen3.5-2B | thinking, t=1.0 ×4 | 3.0/22 (3–3); 37/88 answers never finished thinking within 16k tokens |
| Qwen3.5-0.8B | thinking, t=1.0 ×4 | 0.25/22 (0–1); 12/88 never finished thinking |

## Rounds

| # | Track | Change | Dev (Spider) | Train metric | Decision | LoRA path |
|---|---|---|---|---|---|---|
| P1 | proxy | **Tinker pilot**: Qwen3.5-4B, 40 steps, 32 prompts × 4 rollouts, LoRA r32, lr 1e-4, 4k-token rollouts | 73.7 → 73.7 / 77.0 / 78.0 / 75.3 (steps 10/20/30/40, n=300, ±5): gain is within noise | teacher KL 0.191 → 0.189; response length 443 → 3,639 tokens | pilot (reference point) | tinker://c8321dd6…/sampler_weights/final |

**P1 at its final checkpoint (chosen in advance):** TPC-H 22 went from **11.5 → 17.0/22** (t=1.0 ×4; ranges 11–13 vs 14–19). 13 queries improved and 2 got worse (Q13, Q17). This run trained on proxy data only, with no TPC-H schema. The median response went from about 500 to about 3,200 words, so the student picked up the teacher's long reasoning. The dev gain is much smaller than the TPC-H gain. That fits Spider being easy, shallow SQL near its ceiling, but it is the overfitting tripwire, so it needs checking with the fresh and probe sets.
