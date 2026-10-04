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
| train (target) | `target_train_split`: 762 new TPC-H-schema questions from `gen_tpch_questions.py`, decontaminated against the 22, the fresh set and the probes | training only (target track) |
| **dev** | `target_dev`: 150 held-out target-track questions on the TPC-H schema. Gold is teacher SQL that agreed across 2 samples and was decontaminated against every test set; these never enter training | **every keep/revert decision** |
| dev (secondary) | `spider_dev_clean`: 648 questions on 20 databases unseen in training | reported alongside; too easy and insensitive to decide on (P1: +2–4 pts vs +17–25 on TPC-H-style sets) |
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

**P1 generalization (final checkpoint, t=1.0 ×4):**

| Set | Base | P1 final |
|---|---|---|
| TPC-H 22 | 11.5 (11–13) | 17.0 (14–19) |
| Fresh TPC-H (30) | 18.0 (17–20) | 23.0 (21–25) |
| Probes (22, changed meaning) | 10.75 (9–12) | 16.0 (13–18) |

The probe gain matches the TPC-H gain, so this is better reading and reasoning, not recall of memorized TPC-H answers. The fresh-set gain confirms it transfers to new TPC-H-schema questions.

**target_dev check for P1** (150 questions × 2 samples, t=1.0): base **53.7% ±5.6** → P1 final **76.0% ±4.8**, +22 points. It moves together with TPC-H (+25), fresh (+17) and probes (+24), which confirms target_dev as the decision metric. Spider dev only moved +2 to +4 points.

### Track comparison at 40 steps (4B, lr 1e-4, final checkpoints, t=1.0)

| Run | target_dev | TPC-H 22 | Fresh (30) | Probes (22) |
|---|---|---|---|---|
| Base | 53.7% | 11.5 (11–13) | 18.0 (17–20) | 10.75 (9–12) |
| P1 proxy (BIRD/Spider, no TPC-H schema) | 76.0% | **17.0** (14–19) | **23.0** (21–25) | 16.0 (13–18) |
| P2 target (new TPC-H-schema questions) | **86.3%** | 15.75 (14–17) | 22.0 (21–23) | 16.0 (14–17) |

**Finding:** target-track training gains +10 points on target_dev but nothing on the test sets. target_dev shares a generator, and so a style, with the target training data, so **it is biased toward the target track**. Rule change: target_dev decides LR, steps, rank and similar settings *within* a data mix. Data-mix choices are judged on proxy-vs-target parity, and confirmed later on the tests at checkpoints chosen in advance. The proxy track, which never sees the TPC-H schema, generalizes to TPC-H at least as well as the target track.

### Round 1 (2B, prime-rl, proxy) throughput notes
- Each step takes 3–4.5 minutes on 2×H100. About 55% of rollouts hit the 6,144-token cap (the 2B thinks at length), so each step is about 650k student tokens plus about 700k teacher-scored tokens on Tinker.
- The vLLM router circuit breaker opens briefly at some weight updates (step 5: 78% of traces failed and were resampled).
- Plan: tune the recipe on Tinker with the 4B (fast, run in parallel), then move it to the 2B with 4 GPUs (2 inference + 2 trainer).
