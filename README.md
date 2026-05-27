# Baking with Tinker — Bake, SFT, RL in one repo

Three idiot-proof training entry points on top of [Tinker](https://tinker.thinkingmachines.ai):

| Script | What it does | Data |
|--------|--------------|------|
| `bake.py` | **Prompt baking** — bake a system prompt into model weights ([paper](https://arxiv.org/abs/2409.13697)) | Auto-generated from `prompt.md` |
| `sft.py`  | **Supervised fine-tuning** — instruction-tune on a real chat dataset | `HuggingFaceH4/no_robots` |
| `rl.py`   | **RL (GRPO)** — group-relative policy gradient on a toy task | `ArithmeticEnv` ("What is X + Y?") |

All three use the same base model (`Qwen/Qwen3-8B`), same LoRA rank, and write checkpoints under `/tmp/baking-logs/`.

---

## Setup

```bash
# 1. Install
pip install -e "care package/tinker-cookbook"
pip install python-dotenv

# 2. Configure
cp "care package/.env.example" "care package/.env"
# edit care package/.env and put your Tinker key
```

You need one secret: `TINKER_API_KEY` (get it from https://tinker.thinkingmachines.ai). Everything else is optional — `WANDB_MODE=disabled` skips W&B; OpenRouter is no longer required (data generation now uses Tinker).

---

## 1. Baking

```bash
python bake.py
```

What happens:
1. Hashes `prompt.md`. If `baking_data.jsonl` is missing or was generated from a different prompt, it regenerates (200 examples, ~2 min on Tinker). Otherwise it reuses the existing data.
2. Trains a LoRA adapter using top-K KL distillation:
   - Sample top-20 logprobs from the **prompted** base model
   - Force the **unprompted** student to match those logprobs via `forward_backward_custom`
3. Runs verification queries (no system prompt) to show the persona stuck.

To change what gets baked, edit `prompt.md` and re-run — data regenerates automatically.

**Why this works:** minimizes forward KL between the prompted and unprompted distributions per token. Forward KL is mean-seeking, so the student covers all modes of the prompted behavior. Top-K captures the shape of the distribution, not just the argmax. See [Bhargava et al. 2024](https://arxiv.org/abs/2409.13697).

## 2. SFT

```bash
python sft.py
```

Thin wrapper around `tinker_cookbook.supervised.train` with `NoRobotsBuilder`. ~10k hand-written prompt/response pairs from HuggingFace — the cookbook's recommended starter dataset. Trains for 1 epoch with linear LR decay.

To swap datasets: edit `sft.py` and replace `NoRobotsBuilder` with `FromConversationFileBuilder(file_path="your.jsonl")` (see commented hint in the file).

## 3. RL (GRPO)

```bash
python rl.py
```

Thin wrapper around `tinker_cookbook.rl.train` with `ArithmeticDatasetBuilder`. The reward function is intentionally trivial:

```python
def check_answer(self, sample_str: str) -> bool:
    try:
        return int(sample_str.split()[0]) == self.x + self.y
    except (ValueError, IndexError):
        return False
```

Reward = 1 if the model's first token is the correct integer sum, else 0. A 2-shot prefix is included so the base model gets some right out of the gate — meaning group advantages aren't all zero and GRPO has signal to bootstrap on. You'll watch the mean reward climb from ~0.05 to ~0.9 over the first few hundred batches.

**GRPO note:** the cookbook's RL trainer computes advantages by centering rewards within each group of `group_size` rollouts per problem. That IS GRPO — no extra flags needed.

To swap in a real benchmark: change `ArithmeticDatasetBuilder` → `Gsm8kDatasetBuilder` in `rl.py` (also in `tinker_cookbook/recipes/math_rl/`).

---

## Tweaking

All hyperparameters live in `config.py`, organized into three sections (baking, SFT, RL) plus a shared section at the top. The defaults are conservative and known to work.

For a faster smoke test on any of the three modes, drop:
- `BAKE_NUM_EPOCHS = 1` and `NUM_QUERIES = 20`
- `SFT_NUM_EPOCHS = 1` (already)
- `RL_N_BATCHES = 20`

## Files

| File | Purpose |
|------|---------|
| `prompt.md` | The system prompt to bake. Edit and re-run `bake.py` — data regenerates automatically. |
| `config.py` | All hyperparameters, organized by mode. |
| `bake.py` | Baking entry point. Auto-regenerates data if `prompt.md` changes. |
| `sft.py` | SFT entry point. |
| `rl.py` | GRPO entry point. |
| `generate_data.py` | Data generator for baking (uses Tinker sampling). Called automatically by `bake.py`. |
| `demo.py` | Query a saved checkpoint after training. |
| `baking_data.jsonl` | Cached baking data. |
| `baking_data.meta` | Hash of the `prompt.md` used to generate the cache. |

## Paper

```
@article{bhargava2024baking,
  title={Baking Generalizable Features into Pretrained Language Models with Prompt Baking},
  author={Bhargava, Aman and Witkowski, Cameron and Detkov, Alexander and Thomson, Matt},
  journal={arXiv preprint arXiv:2409.13697},
  year={2024}
}
```
