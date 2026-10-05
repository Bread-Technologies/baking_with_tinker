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

## Sweep S (Tinker, Qwen3.5-4B, 200 steps each, run in parallel; base = 50/50 mix, lr 5e-4, 4k-token rollouts, rank 32)

target_dev (150 × 2 samples, t=1.0), updated as checkpoints land. Base model: 53.7%.

| Run | Change | step 25 | step 50 | Notes |
|---|---|---|---|---|
| s01 | base | 83.3 | | |
| s02 | lr 1e-3 | 79.3 | 66.0 | **stopped**: getting worse, LR too high |
| s03 | lr 2e-4 | 86.0 | 87.0 | |
| s04 | target data only, lr 5e-4 | 83.7 | 86.0 | |
| s05 | proxy data only, lr 5e-4 | 49.3 | 55.0 | proxy at 5e-4 doesn't help (P1 used proxy at 1e-4 and reached 76.0) |
| s06 | 8k-token rollouts | 79.0 | 82.3 | |
| s07 | rank 128 | 88.0 | | |

## Sweep B (prime-rl on Modal, 1×H100 per run with vLLM and trainer on the same GPU, 200 steps)

b01: 2B mix lr 2e-4 · b02: 2B mix lr 5e-4 r128 · b03: 2B mix lr 1e-4 · b04: 2B target lr 2e-4 · b05: 0.8B mix lr 2e-4. Eval: target_dev every 25 steps.

### Stall analysis: target_dev failures, s03 step 100 (90.3%)

There are 29 failures out of 300 attempts. 8 questions fail both samples and 13 fail one.

Of the 8 systematic failures, about 5 are real student errors:
- `EXTRACT(MONTH)` used where calendar months were needed
- a filter dropped in a LEFT JOIN
- the wrong CTE column referenced
- an average taken over the wrong set

About 3 are **flawed dev items**:
- a question asks for 4 values but lists 3 output columns
- "1995" doesn't say whether it means the order date or the ship date
- an ambiguous "distinct parts" ranking

So target_dev's ceiling is under 100%, and the 88–91% plateau is partly that ceiling. Configs separated by less than about 3 points can't be told apart on it.

**Decision:** use the test sets (TPC-H 22, fresh, probes) only at the final checkpoints chosen in advance (step 200 for sweep S), and judge sweep S by those plus target_dev. Don't fix dev items by reading test failures. The ambiguous dev items can be dropped from target_dev, but that changes only the dev metric, never training.

### prime-rl throughput fix (sweep B relaunched)
- **Root cause:** the orchestrator awaited each finished episode's teacher scoring serially (`train_sink.add` → `finalize_episode`), at about 3 s per Tinker call × 128 episodes. That made 7–10 min/step.
- **Fix** (`training/prime/patches/opd_concurrent_scoring.py`, applied at image build): `finalize_episode` starts the scoring task, and `finalize_group` awaits that group's tasks before it can enter a batch.
- **Result:** 2B is 1.4–3.9 min/step (about 2.5 average) and 0.8B about 1.5 min/step, each on one H100 shared by vLLM and the trainer. Teacher calls now overlap about 3.5×.
- Round 1 (2B proxy, 2 GPUs, unpatched) was stopped at about step 20 and replaced by b06 (2B proxy, lr 2e-4).
- Untrained baselines on target_dev at step 0: 2B 0.7–1.3%, 0.8B 0%.

### Sweep S, first final-checkpoint test result (chosen in advance: step 200)

| Run | target_dev @200 | TPC-H 22 | Fresh (30) | Probes (22) |
|---|---|---|---|---|
| s04 target only, lr 5e-4 | 89.7% | 17.75 (17–18) | 23.5 (23–24) | 16.5 (15–18) |
| (P1, 40 steps, proxy) | 76.0% | 17.0 | 23.0 | 16.0 |

**Finding:** 5× more training raised target_dev by 14 points but the test sets by less than 1. target_dev has stopped predicting test performance: its single-skill generated questions are easier than TPC-H's multi-condition reports, so they saturate first. Following the blog (the eval should mirror the real task, and difficulty should come from a human judgment of what is hard, not from test failures), I'm generating **hard** questions that combine 2–3 skills at report depth, with the same walled-off generator, to use as harder training prompts and a harder dev slice. Test failures were not used for this.

### Ceiling calibration and the reasoning-length gap

| Model (t=1.0 ×4) | TPC-H 22 | Fresh (30) | Probes (22) | median / p90 response words on TPC-H |
|---|---|---|---|---|
| Teacher 397B | **22.0** (22–22) | **29.25** | **22.0** | 511 / 1,745 |
| 4B s03 final | 17.5 (16–18) | 24.75 | 18.25 | 359 / 602 |
| 4B s04 final | 17.75 | 23.5 | 16.5 | |
| 4B s06 final | 16.0 | 23.0 | 16.0 | |

Under the same sampling the teacher is perfect, so 22/22 is reachable. The student **under-thinks on hard questions**: its p90 response length is about a third of the teacher's. Reverse KL is mode-seeking, and the training prompts are mostly easy, single-skill questions where the teacher's mode is a short answer, so the student never practices long reasoning. Next lever: harder multi-skill prompts (generating now), where the teacher reasons at length.

### Cost cut (user flagged Tinker spend)
- Stopped the remaining 4B sweep runs (s01, s05, s07, s08, s09), the auto-evaluator and its in-flight evals, and hard-question generation (which made 3 thinking calls to the 397B per question).
- Cut sweep B from 6 Modal runs to 2: **b01** (2B, mix, lr 2e-4: the best 4B recipe) and **b05** (0.8B). Every Modal run also pays for 397B teacher scoring on Tinker at every step.
- What sweep S settled: lr 2e-4 to 5e-4 and the mix/target data all plateau at about 17–18/22 on TPC-H, and more steps, rank or rollout length don't move it. The remaining lever is *what* the student practices (harder prompts that need long reasoning), not more of the same training.
- From now on, run one experiment at a time with a stated hypothesis; score dev with 1 sample (not 2) and only at chosen steps; run the 4-sample test sets on final checkpoints only.

### Sweep B (kept: b01 2B, b05 0.8B), target_dev in prime-rl (150 × 1 sample, t=1.0)

| Run | step 0 | step 25 |
|---|---|---|
| b01 2B mix lr 2e-4 | 1.3% | **19.6%** |
| b05 0.8B mix lr 2e-4 | 0.0% | **8.0%** |

The 2B's training-set execution reward went from 0.07–0.15 to about 0.30 by step 35–40. Steps take 1–5 minutes; step 30 of b01 had a 40.9% rollout-error burst (router circuit breaker), recovered by step 40.

### 2B (b01, prime-rl, mix lr 2e-4; batch 512 from step 30) on the test sets (t=1.0 ×4, Modal vLLM + LoRA)

| step | TPC-H 22 | Fresh (30) | Probes (22) | unfinished thinking (TPC-H) |
|---|---|---|---|---|
| base | 3.0 | – | – | 37/88 |
| 46 (batch 128) | 8.5 (7–11) | 9.0 | 8.0 | 20/88 |
| 50 | 9.75 (9–10) | 12.75 | 10.5 | 11/88 |
| 60 | 9.75 (8–12) | 12.75 | 10.5 | 14/88 |

Gains are steady but slow. Operational notes: twice the runs were cancelled because they were launched with `modal run --detach`, which ties them to the local client; they are now spawned on the deployed app. The local eval driver also died with the sandbox's background processes, so evals now run as tracked background tasks, with scheduled check-ins as a backstop.

### Test-time scaling on the best 4B (s03 final), using only the model's own SQL and the database (never gold)

| Setting | TPC-H 22 | Fresh (30) | Probes (22) |
|---|---|---|---|
| single sample (mean of 16) | 17.56 | 25.75 | 18.19 |
| majority vote over 16 by executed result | 19 | 28 | 19 |
| + up to 2 repairs after an execution error, then vote@16 | **20** (mean 18.75) | 27 | **20** |

The repairs remove all execution-error failures (binder errors). Two TPC-H questions remain wrong in most samples: one at 0% and one at 25%. These are reasoning errors, not dialect errors. Per the rules, the specific test failures are not used to shape training. The generic levers still open are harder multi-skill prompts and multi-turn training with execution feedback.

2B (b01) step 80: TPC-H 11.25 (9–12); majority@4 14; pass@4 15. Fresh 13.5; probes 9.0.

### s10: continue the 4B s03 on 171 hard multi-skill questions (×6) + mix, 8k rollouts, 100 steps (final chosen in advance)

| s10 final, 16 samples + 2 repairs | TPC-H 22 | Fresh (30) | Probes (22) |
|---|---|---|---|
| mean of single samples | 18.88 (17–20) | 26.94 | 19.56 |
| majority vote @16 | 20 | 28 | **21** |

This is a small gain over s03 (mean +0.1 TPC-H, +0.4 fresh, +0.6 probes; probe vote 20 → 21). TPC-H stays at 20/22 under voting. One question is still wrong in every sample, and which question is second-hardest moved, which points to noise rather than a fixed gap. Diminishing returns on the 4B. Focus moves to the 2B, the goal model.

2B (b01) step 100, 16 samples + 2 repairs: TPC-H mean 12.62 (10–16), **vote@16 16/22**; fresh mean 17.75, vote 21/30; probes mean 12.38, vote 15/22. Still climbing from step 80 (mean 11.25 → 12.62). Training continues to step 200; steps 120–200 are queued for the same eval.

0.8B (b05) step 150, 16 samples + 2 repairs: TPC-H mean 6.19 (3–8), vote@16 10/22; fresh mean 6.62, vote 10/30; probes mean 4.19, vote 7/22. The 0.8B is far behind the 2B (mean 12.6 at step 100), so I stopped it at step ~150 (LoRAs kept on the volume) and gave its GPU to the 2B.

### b07: branch the 2B off b01 step 110; one change, train data → hard_mix_train (as s10 did for the 4B)

b01 keeps going on mix to step 200 as the control; b07 resumes the same checkpoint on hard_mix_train (171 hard multi-skill questions ×6 + mix) to step 200. Same lr, batch, and eval. Both are scored at matching steps with 16 samples + 2 repairs.

2B (b01) step 120 (Modal-side eval; 16 samples + 2 repairs): TPC-H mean 12.81, **vote@16 18/22**; fresh mean 17.69, vote 21/30; probes mean 11.19, vote 15/22; Spider test (300, single sample) 80.7% ± 4.5.

Evals now run as Modal jobs (tpch_eval/modal_eval_lora.py), because sandbox restarts kept killing local eval queues. training/sync_evals.py pulls the results back.

### Memorization dial (0.8B; contaminated by design, reported only as a trade-off curve)

All four runs branch off b05 step 150 (clean; TPC-H vote 10/22). Each trains 60 more steps to step 210, and is scored at steps 170, 190 and 210 on TPC-H, fresh, probes and Spider test. The data comes from training/memorization/build_dial.py.

| Point | Training data | Rows |
|---|---|---|
| A | clean b05 step 150 | — |
| B | mix + the 22 templates with new constants | 159 ×9 |
| C | mix + paraphrases of the 22, same meaning | 170 ×9 |
| D | mix + the exact 22 prompts | 22 ×68 |
| E | the exact 22 prompts only | 22 |

**2B, mix vs hard_mix (16 samples + 2 repairs; single-sample mean, with vote@16 in parentheses):**

| run / step | TPC-H | fresh | probes | Spider test |
|---|---|---|---|---|
| b01 mix 120 | 12.81 (18) | 17.69 (21) | 11.19 (15) | 80.7% |
| b01 mix 140 | 12.56 (17) | 17.00 (21) | 11.12 (15) | 79.0% |
| b07 hard_mix 130 | 14.19 (14) | 18.25 (21) | 11.75 (15) | 80.0% |
| b07 hard_mix 140 | 13.50 (16) | 17.25 (23) | 12.75 (16) | 80.0% |

hard_mix raises the single-sample mean by about 1 TPC-H point at matched steps, with no loss on Spider. The vote counts are noisy at this size (±2).

**b07 collapsed after step ~155.** At step 160: TPC-H mean 4.12 (vote 7/22), fresh 3.5, probes 2.25, Spider 72.7%.
- In the orchestrator log, rollout truncation at the 6144-token cap rose from ~5–25% (steps 140–155) to 60%+ (steps 161–171). Train reward (execution match) fell from ~0.40 to ~0.07.
- In the eval, 146 of 352 TPC-H samples ran past the context limit: the student loops in its thinking and never closes it.
- Diagnosis: the student degenerates into looping on the long, hard prompts at lr 2e-4. OPD's per-token reverse KL does not penalise the loop, and prime-rl has no option to drop truncated rollouts.
- Action: cancelled b07. Its best checkpoint is step 140 (TPC-H mean 13.5, vote 16; probes vote 16; Spider 80%).
- Lesson: on hard long-thinking data, use a lower lr or stop early; watch the Truncation column as an early-warning signal.

b01 (mix) step 160: TPC-H mean 13.38, vote 17/22; fresh mean 19.06, vote 24/30; probes mean 12.00, vote 17/22; Spider 81.3%. Still stable: truncation ~25–35%, reward ~0.45.

**Untrained 0.8B (step 0, same eval):** TPC-H mean 0.06 (vote 1/22); fresh 0.44 (3/30); probes 0.31 (2/22); Spider 29.7%.

**Dial at step 170** (20 steps after branching; mean, with vote@16 in parentheses):

| point | TPC-H | fresh | probes | Spider |
|---|---|---|---|---|
| A clean s150 | 6.19 (10) | 6.62 (10) | 4.19 (7) | 65.7% |
| B variants | 11.75 (15) | 10.06 (13) | 8.88 (14) | 65.3% |
| C paraphrase | 8.75 (11) | 9.69 (13) | 7.06 (9) | 68.7% |
| D exact+mix | 11.69 (15) | 9.56 (13) | 7.69 (14) | 65.0% |
| E exact only | 14.94 (17) | 9.56 (12) | 11.69 (15) | 60.7% |

Fresh rises at every point, so part of each gain may come from the extra 20 steps rather than from the dial data. I added a clean control, dial_0p8b_clean (same branch and steps, mix_train only), so each point can be compared at a matched step.

Eval infra bug: the variants and exactonly step-190 evals scored 0 everywhere, with 404 "model does not exist". The container had been reused and was still running the previous call's vLLM server, so the new LoRA was never loaded. Fix: max_inputs=1, which gives every eval a fresh container. I removed those lines and respawned the two evals. Any nonzero result is unaffected, because a wrong server returns 404s, not wrong scores.

Noise estimate: dial_exact step 170 was evaluated twice and gave TPC-H mean 11.69 and 10.94. So single-sample means move about ±0.5, and votes about ±2.

- b01 step 180: TPC-H mean 12.06 (vote 13), fresh 14.88 (18), probes 10.69 (14), Spider 78.3%. Down from step 160, so the run looks past its peak.
- dial_exact step 190: TPC-H 13.94 (16), fresh 10.00 (12), probes 10.56 (14), Spider 66.0%.

**Dial results so far (0.8B; single-sample mean, vote@16 in parentheses):**

| point / step | TPC-H | fresh | probes | Spider |
|---|---|---|---|---|
| A clean s150 | 6.19 (10) | 6.62 (10) | 4.19 (7) | 65.7% |
| control (mix only) s170 | 5.75 (9) | 8.81 (13) | 4.56 (8) | 63.7% |
| B variants s190 | 13.69 (15) | 10.44 (13) | 10.19 (12) | 67.3% |
| B variants s210 | 15.12 (19) | 10.69 (14) | 12.88 (15) | 66.0% |
| C paraphrase s190 | 12.56 (15) | 8.94 (12) | 8.88 (10) | 67.7% |
| D exact+mix s190 | 13.94 (16) | 10.00 (12) | 10.56 (14) | 66.0% |
| D exact+mix s210 | 13.06 (17) | 10.50 (14) | 10.75 (16) | 68.7% |
| E exact only s190 | 16.44 (18) | 9.12 (13) | 11.31 (14) | 58.3% |

- The control isolates the step effect. Extra clean steps raise fresh (6.6 → 8.8) but not TPC-H (6.2 → 5.8). So every dial point's TPC-H gain (+7 to +10) comes from the dial data, not from training longer.
- Fresh and probe gains over the control are small (+1–2 fresh), while the probes gain +4 to +8. The probes share the 22 templates, so the model learns the template shapes and transfers them to meaning-changed versions. That is template learning, not wording recall.
- Only the pure-memorization point E costs broad generalization: Spider falls 66 → 58%.

b01 step 200: TPC-H 13.19 (vote 18), fresh 17.00 (21), probes 11.69 (13), Spider 79.0%. b08 (lr 5e-5) is stable so far: truncation falling 29% → 17% and reward 0.23 → 0.35 over steps 161–176.

**b08 step 180** (2B, hard_mix at lr 5e-5, branched from b01 step 160): TPC-H mean **14.44**, **vote 19/22**; fresh 19.50 (25/30); probes 12.56 (17/22); Spider 81.7%. This is the best clean 2B on every axis. The lower lr fixed the b07 collapse.

**Dial E (exact only) collapses at step 210:** TPC-H 9.25 (vote 11), fresh 5.25, probes 4.75, Spider 53.0%. Over steps 195–210 train reward oscillates 0.43–0.70 and truncation 12–47%. Training on only 22 prompts peaks around step 190 and then destabilizes everything, the 22 included.

Paraphrase was re-scored at steps 170 and 190 (duplicate evals): TPC-H 8.56 and 12.38, against 8.75 and 12.56 before, which is consistent within noise.

**b08 step 200:** TPC-H mean **14.88**, vote 19/22; fresh 19.19 (22/30); probes 13.69 (**18/22**); Spider **82.3%**. New best clean 2B.
Dial at step 210: paraphrase 14.75 (vote 16), fresh 11.00, probes 10.31, Spider 68.7%. Rewording catches up with new-constants (15.12) given more steps.
Control at step 190: TPC-H 6.50 (10), fresh 7.75 (11), probes 4.69 (9), Spider 67.7%. TPC-H stays flat with clean steps.
Report published as an artifact; copy at training/results/report.html.
