# Prompt Baking

Implementation of [Prompt Baking](https://arxiv.org/abs/2409.13697) (Bhargava, Witkowski, Detkov, Thomson 2024). Baking converts a system prompt + base model weights into new weights that behave as if the system prompt were always present — without actually sending it at inference time.

## What Baking Does

Given a model with weights θ and a system prompt u, baking produces new weights θ_u such that:

```
P_θu(y | x) ≈ P_θ(y | u, x)
```

The model responds *as if* the system prompt were there, but it isn't. The persona is in the weights.

## How It Works (The Algorithm)

Baking minimizes the **forward KL divergence** between the prompted and unprompted distributions:

```
θ_u = argmin_θu  KL( P_θ(· | u) || P_θu(·) )
```

Expanded per-token (paper Eq. 3):

```
L = (1/N) Σ_n Σ_t Σ_k  P_θ(v_k | y<t, u, x)  ·  [ log P_θ(v_k | y<t, u, x) - log P_θu(v_k | y<t, x) ]
```

Where v_k are the top-K tokens from the prompted distribution at each position.

### Why Forward KL (Not Reverse)?

Forward KL is **mean-seeking**: it forces the unprompted model to cover all modes of the prompted distribution. Reverse KL would be mode-seeking and could collapse to a subset of the persona's behavior. Forward KL ensures the baked model captures the full range of how the prompted model would respond.

### Why Top-K (Not Just Top-1)?

The paper's Section 5 ablation shows that using more of the logit distribution (top-K tokens, not just the argmax) is critical for generalization. Top-1 (cross-entropy) works but needs significantly more data. Top-K captures the *shape* of the prompted distribution — not just which token is most likely, but how probability mass is spread across alternatives.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PROMPT BAKING PIPELINE                   │
│                                                              │
│  1. prompt.md          Your system prompt (any persona/behavior)
│       │                                                      │
│  2. generate_data.py   Send prompt + diverse queries to OpenRouter
│       │                Get responses WITH the system prompt   │
│       │                Save WITHOUT the system prompt         │
│       ▼                                                      │
│  3. baking_data.jsonl  200 (user, assistant) pairs           │
│       │                No system message — this is key        │
│       │                                                      │
│  4. bake.py            For each training example:            │
│       │                  a) Get top-K logprobs from PROMPTED  │
│       │                     base model (frozen, with prompt)  │
│       │                  b) Build K datums per example with   │
│       │                     different target tokens           │
│       │                  c) forward_backward_custom with      │
│       │                     KL loss on the TRAINING model     │
│       │                     (LoRA, no prompt)                 │
│       │                  d) optim_step                        │
│       ▼                                                      │
│  5. Baked model        Responds in persona WITHOUT any prompt │
└─────────────────────────────────────────────────────────────┘
```

### Two Models in Play During Training

This is the key thing to understand. There are **two** model instances:

1. **Prompted base model** (frozen `SamplingClient`): The original model WITH the system prompt prepended. This is the "teacher" — it defines the target distribution. We call `sample()` on it with `include_prompt_logprobs=True` and `topk_prompt_logprobs=K` to get the top-K token probabilities at every position.

2. **Unprompted training model** (LoRA `TrainingClient`): The model being fine-tuned WITHOUT the system prompt. This is the "student." We call `forward_backward_custom()` on it with a loss function that pulls its distribution toward the prompted model's distribution.

The loss function receives `logprobs` from the training model's forward pass (student) and compares them against the prompted model's logprobs (teacher) that were captured earlier. The gradient flows only through the training model.

## Files

| File | Purpose |
|------|---------|
| `prompt.md` | The system prompt to bake. Edit this to change what gets baked. |
| `config.py` | All hyperparameters — model, LoRA rank, top-K, batch size, LR, etc. |
| `generate_data.py` | Off-policy data generation via OpenRouter. Produces `baking_data.jsonl`. |
| `bake.py` | Training loop with top-K KL loss + verification. The core algorithm. |
| `demo.py` | Query a baked model checkpoint with no system prompt. |

## Prerequisites

### Care Package

You need the `care package/` directory (not checked into git) containing:
- `tinker-cookbook/` — Tinker SDK helpers (renderers, checkpoint utils, tokenizer utils)
- `.env` — API keys

The `.env` file must contain:
```
TINKER_API_KEY=tml-...
OPENROUTER_API_KEY=sk-or-v1-...
WANDB_API_KEY=...
WANDB_ENTITY=...
```

### Install Dependencies

```bash
pip install -e "care package/tinker-cookbook"
pip install httpx python-dotenv wandb
```

Uses Python 3.11 (`/opt/homebrew/bin/python3.11`).

## How to Bake

### Step 1: Write Your Prompt

Edit `prompt.md` with whatever persona or behavior you want to bake. The current contents bake a Yoda persona. This can be anything — a coding style, a customer service tone, a fictional character, safety instructions, etc.

### Step 2: Generate Training Data

```bash
python3.11 generate_data.py
```

This sends 50 diverse seed queries to OpenRouter with your system prompt at 4 temperatures (0.5, 0.7, 0.9, 1.0), producing ~200 examples. The responses are generated WITH the system prompt but saved WITHOUT it. The data file `baking_data.jsonl` contains `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` entries — no system message.

The diversity of queries matters. The seed queries span 10 categories (life advice, science, tech, philosophy, cooking, relationships, history, humor, health, education) to ensure the baked behavior generalizes across topics, not just the ones seen during training.

### Step 3: Train (Bake)

```bash
python3.11 bake.py
```

This runs the full baking loop:
- Creates a LoRA training client (rank 32) and a frozen base sampling client
- For each batch:
  1. Queries the **prompted** base model with `sample(include_prompt_logprobs=True, topk_prompt_logprobs=20)` to get top-20 (token_id, logprob) at every response position
  2. Builds 20 datums per example — each datum has the same input tokens but different target tokens (the k-th most likely token from the prompted distribution)
  3. Calls `forward_backward_custom` with a KL loss that computes `Σ_k P_prompted(v_k) * [log P_prompted(v_k) - log P_model(v_k)]`
  4. Calls `optim_step` with linear LR decay
- Saves checkpoints and logs to W&B
- Runs verification: queries the baked model with 10 test prompts and NO system prompt

Training takes ~6 minutes for 200 examples × 4 epochs. You should see KL drop from ~0.8 to ~0.002.

### Step 4: Verify / Demo

After training, `bake.py` automatically runs verification. To query a saved checkpoint later:

```bash
python3.11 demo.py
```

Edit the `model_path` in `demo.py` to point to your checkpoint. The path format is `tinker://<run_id>:train:0/sampler_weights/<checkpoint_name>`.

## Tinker API Details for Claude Code

These are the specific API patterns that matter. Read this if you're modifying the code.

### Getting Top-K Logprobs

```python
result = sampling_client.sample(
    prompt=model_input,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,      # MUST be True
    topk_prompt_logprobs=K,            # MUST also be set
)
# result.topk_prompt_logprobs[position] = [(token_id, logprob), ...] up to K entries
```

**Critical**: You MUST pass BOTH `include_prompt_logprobs=True` AND `topk_prompt_logprobs=K` together. Passing `topk_prompt_logprobs` alone returns None.

### Custom Loss with forward_backward_custom

```python
def my_loss(data: list[tinker.Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
    # data[i].loss_fn_inputs["target_tokens"] — the target token IDs
    # data[i].loss_fn_inputs["weights"] — which positions to train on
    # logprobs[i][t] — log P_model(target_token_t) at shifted position t
    # Return (loss_tensor, metrics_dict)
    ...

tc.forward_backward_custom(datums, my_loss)
tc.optim_step(adam_params)
```

The loss function receives logprobs from the model's forward pass for the specific target tokens in each datum. This is how we get `log P_model(v_k)` — by setting `target_tokens` to the k-th top token from the prompted distribution.

### Datum Format

```python
datum = tinker.Datum(
    model_input=tinker.ModelInput.from_ints(input_token_ids),  # tokens[:-1]
    loss_fn_inputs={
        "target_tokens": tinker.TensorData(data=target_ids, dtype="int64", shape=[n]),  # tokens[1:]
        "weights": tinker.TensorData(data=weight_vals, dtype="float32", shape=[n]),
    },
)
```

Input is right-shifted (tokens[:-1]), targets are left-shifted (tokens[1:]). Weights mask which positions contribute to the loss — only response tokens, not the user query or special tokens.

### Renderer

Use `qwen3_disable_thinking` for Qwen3 models. This adds empty `<think>\n\n</think>\n\n` blocks to suppress chain-of-thought. Use `TrainOnWhat.LAST_ASSISTANT_MESSAGE` (not `ALL_ASSISTANT_MESSAGES`).

### Position Alignment

The prompted sequence is longer than the unprompted sequence (it has the system prompt tokens prepended). The response tokens are at the END of both sequences. To align: find which positions have weight > 0 in each sequence, then align from the end.

## Hyperparameters (config.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen3-8B | Tinker model name |
| LoRA rank | 32 | Paper doesn't specify; 32 works well |
| Top-K | 20 | Paper Section 5 ablation. More = better generalization |
| Batch size | 16 | 16 examples × 20 datums = 320 datums per step |
| Learning rate | 1e-4 | With linear decay to 0 |
| Epochs | 4 | KL converges well before epoch 4 |
| Data | 200 examples | 50 queries × 4 temperatures |

## Paper Reference

```
@article{bhargava2024baking,
  title={Baking Generalizable Features into Pretrained Language Models with Prompt Baking},
  author={Bhargava, Aman and Witkowski, Cameron and Detkov, Alexander and Thomson, Matt},
  journal={arXiv preprint arXiv:2409.13697},
  year={2024}
}
```
