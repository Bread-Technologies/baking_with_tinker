#!/usr/bin/env python3
"""
Prompt Baking: Bake a system prompt into model weights.

Minimizes KL(P_θ(·|u) || P_θu(·)) using TRUE top-K logit matching via Tinker's
forward_backward_custom. Off-policy data from OpenRouter, training on Tinker.

Key discovery: sample() needs BOTH include_prompt_logprobs=True AND
topk_prompt_logprobs=K to return top-k (token_id, logprob) tuples.

Algorithm (paper Eq. 3):
  1. Get top-K (token_id, logprob) at each position from prompted base model
  2. Build K datums per example, each targeting a different top-k token
  3. Custom loss: KL_t ≈ Σ_k P_prompted(v_k) * [log P_prompted(v_k) - log P_model(v_k)]
  4. optim_step

Paper: arXiv:2409.13697
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import tinker
import torch
import wandb
from dotenv import load_dotenv
from tinker_cookbook import renderers, checkpoint_utils
from tinker_cookbook.renderers.base import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

import config as C

load_dotenv("care package/.env")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_console)
logging.getLogger("httpx").setLevel(logging.WARN)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# Top-K KL loss factory
# ---------------------------------------------------------------------------

def make_topk_kl_loss(prompted_lps_per_datum: list[list[float]], K: int):
    """
    Create custom loss for top-K KL(P_prompted || P_model).

    Datums come in groups of K (one per top-k token per example).
    prompted_lps_per_datum[i][t] = log P_prompted(v_k_t) for datum i.
    logprobs[i][t] = log P_model(v_k_t) from the forward pass.

    KL_t ≈ Σ_k exp(prompted_lp_k_t) * (prompted_lp_k_t - model_lp_k_t)
    """
    def topk_kl_loss(
        data: list[tinker.Datum], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = torch.tensor(0.0)
        total_kl = 0.0
        n_tokens = 0

        n_examples = len(data) // K
        for ex_idx in range(n_examples):
            # Sum KL contributions across K tokens at each position
            for k in range(K):
                datum_idx = ex_idx * K + k
                if datum_idx >= len(data) or datum_idx >= len(logprobs):
                    break

                datum = data[datum_idx]
                model_lp = logprobs[datum_idx]
                weights = torch.tensor(datum.loss_fn_inputs["weights"].data)
                prompted_lp = torch.tensor(prompted_lps_per_datum[datum_idx])

                min_len = min(len(weights), len(prompted_lp), len(model_lp))
                weights = weights[:min_len]
                prompted_lp = prompted_lp[:min_len]
                model_lp = model_lp[:min_len]

                p_prompted = torch.exp(prompted_lp)
                kl_per_token = p_prompted * (prompted_lp - model_lp) * weights
                total_loss = total_loss + kl_per_token.sum()
                total_kl += kl_per_token.sum().item()

                if k == 0:
                    n_tokens += weights.sum().item()

        avg_kl = total_kl / max(n_tokens, 1)
        return total_loss, {"kl_loss": total_loss.item(), "avg_kl_per_token": avg_kl}

    return topk_kl_loss


# ---------------------------------------------------------------------------
# Build K datums per example using top-K logprobs
# ---------------------------------------------------------------------------

def build_topk_datums_for_example(
    unprompted_mi: tinker.ModelInput,
    unprompted_weights: torch.Tensor,
    prompted_weights: torch.Tensor,
    topk_prompt_logprobs: list,  # list of (list of (tok_id, logprob) or None) per prompted position
    K: int,
    max_length: int,
) -> tuple[list[tinker.Datum], list[list[float]]]:
    """
    Build K datums for one example. Each datum has the same model_input but
    different target_tokens (the k-th top token from the prompted distribution).

    Returns (datums, prompted_lps_per_datum) where prompted_lps_per_datum[k][t]
    is the prompted logprob for datum k at shifted position t.
    """
    tokens = unprompted_mi.to_ints()
    seq_len = len(tokens)
    if max_length and seq_len > max_length:
        tokens = tokens[:max_length]
        unprompted_weights = unprompted_weights[:max_length]
        seq_len = max_length

    if seq_len < 2:
        return [], []

    # Find response positions
    unprompted_resp = [t for t in range(seq_len) if unprompted_weights[t] > 0]
    prompted_resp = [t for t in range(len(prompted_weights)) if prompted_weights[t] > 0]
    n_resp = min(len(unprompted_resp), len(prompted_resp))
    if n_resp == 0:
        return [], []

    # Align from end
    unprompted_resp = unprompted_resp[-n_resp:]
    prompted_resp = prompted_resp[-n_resp:]

    # Shifted positions: input = tokens[:-1], target = tokens[1:]
    base_target = list(tokens[1:])
    base_weights = unprompted_weights[1:seq_len].tolist()
    shifted_input = tinker.ModelInput.from_ints(tokens[:-1])
    n_tgt = len(base_target)

    datums = []
    all_plps = []

    for k in range(K):
        new_target = list(base_target)
        new_weights = list(base_weights)
        plps = [0.0] * n_tgt
        has_any = False

        for i in range(n_resp):
            shifted_pos = unprompted_resp[i] - 1
            if shifted_pos < 0 or shifted_pos >= n_tgt:
                continue

            prompted_pos = prompted_resp[i]
            if prompted_pos >= len(topk_prompt_logprobs):
                continue

            topk_at_pos = topk_prompt_logprobs[prompted_pos]
            if topk_at_pos is None or k >= len(topk_at_pos):
                # No k-th alternative — zero out this position for this datum
                new_weights[shifted_pos] = 0.0
                continue

            tok_id, logprob = topk_at_pos[k]
            new_target[shifted_pos] = tok_id
            plps[shifted_pos] = logprob
            new_weights[shifted_pos] = base_weights[shifted_pos]
            has_any = True

        if not has_any:
            continue

        datums.append(tinker.Datum(
            model_input=shifted_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=new_target, dtype="int64", shape=[n_tgt]
                ),
                "weights": tinker.TensorData(
                    data=new_weights, dtype="float32", shape=[n_tgt]
                ),
            },
        ))
        all_plps.append(plps)

    return datums, all_plps


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    system_prompt = Path(C.PROMPT_FILE).read_text().strip()
    logger.info(f"System prompt: {system_prompt[:60]}...")

    raw_data = load_data(C.DATA_FILE)
    if not raw_data:
        logger.error(f"No data found in {C.DATA_FILE}. Run generate_data.py first.")
        sys.exit(1)
    logger.info(f"Loaded {len(raw_data)} training examples")

    tokenizer = get_tokenizer(C.MODEL_NAME)
    renderer = renderers.get_renderer(C.RENDERER_NAME, tokenizer)

    service = tinker.ServiceClient()
    tc = service.create_lora_training_client(base_model=C.MODEL_NAME, rank=C.LORA_RANK)
    prompted_sc = service.create_sampling_client(base_model=C.MODEL_NAME)

    os.makedirs(C.LOG_DIR, exist_ok=True)

    wandb.login(key=os.getenv("WANDB_API_KEY", ""))
    wandb.init(
        project=C.WANDB_PROJECT,
        entity=os.getenv("WANDB_ENTITY") or None,
        name=f"bake-yoda-topk{C.TOP_K}",
        config={
            "model": C.MODEL_NAME,
            "lora_rank": C.LORA_RANK,
            "top_k": C.TOP_K,
            "batch_size": C.BATCH_SIZE,
            "learning_rate": C.LEARNING_RATE,
            "num_epochs": C.NUM_EPOCHS,
            "loss": f"top{C.TOP_K}_kl",
        },
    )

    # Pre-compute sequences
    logger.info("Pre-computing sequences...")
    examples = []
    for item in raw_data:
        user_content = item["messages"][0]["content"]
        assistant_content = item["messages"][1]["content"]

        unprompted_msgs = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        prompted_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        unprompted_mi, unprompted_weights = renderer.build_supervised_example(
            unprompted_msgs, TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        prompted_mi, prompted_weights = renderer.build_supervised_example(
            prompted_msgs, TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        examples.append({
            "unprompted_mi": unprompted_mi,
            "unprompted_weights": unprompted_weights,
            "prompted_mi": prompted_mi,
            "prompted_weights": prompted_weights,
        })
    logger.info(f"Pre-computed {len(examples)} examples")

    n_batches_per_epoch = len(examples) // C.BATCH_SIZE
    total_steps = n_batches_per_epoch * C.NUM_EPOCHS
    step = 0
    sp = tinker.SamplingParams(max_tokens=1)

    logger.info(f"Training: {C.NUM_EPOCHS} epochs, {n_batches_per_epoch} batches/epoch, "
                f"{total_steps} total steps, top_k={C.TOP_K}")

    try:
        for epoch in range(C.NUM_EPOCHS):
            for batch_idx in range(n_batches_per_epoch):
                t0 = time.time()
                batch_examples = examples[
                    batch_idx * C.BATCH_SIZE : (batch_idx + 1) * C.BATCH_SIZE
                ]

                # 1. Get top-K logprobs from prompted base model
                #    KEY: need BOTH include_prompt_logprobs=True AND topk_prompt_logprobs=K
                topk_futures = []
                for ex in batch_examples:
                    future = prompted_sc.sample(
                        prompt=ex["prompted_mi"],
                        num_samples=1,
                        sampling_params=sp,
                        include_prompt_logprobs=True,
                        topk_prompt_logprobs=C.TOP_K,
                    )
                    topk_futures.append(future)

                # 2. Build K datums per example
                all_datums = []
                all_plps = []
                for ex, future in zip(batch_examples, topk_futures):
                    result = future.result()
                    topk_lps = result.topk_prompt_logprobs

                    if topk_lps is None:
                        logger.warning("No topk_prompt_logprobs — skipping")
                        continue

                    datums, plps = build_topk_datums_for_example(
                        ex["unprompted_mi"],
                        ex["unprompted_weights"],
                        ex["prompted_weights"],
                        topk_lps,
                        C.TOP_K,
                        C.MAX_LENGTH,
                    )
                    all_datums.extend(datums)
                    all_plps.extend(plps)

                if not all_datums:
                    logger.warning(f"Step {step}: no datums. Skipping.")
                    step += 1
                    continue

                # 3. LR decay
                lr_mult = max(0.0, 1.0 - step / total_steps)
                current_adam = tinker.AdamParams(
                    learning_rate=C.LEARNING_RATE * lr_mult,
                    beta1=C.ADAM_BETA1, beta2=C.ADAM_BETA2, eps=C.ADAM_EPS,
                )

                # 4. Forward-backward with top-K KL loss
                loss_fn = make_topk_kl_loss(all_plps, C.TOP_K)
                fb_future = tc.forward_backward_custom(all_datums, loss_fn)
                os_future = tc.optim_step(current_adam)

                fb_result = fb_future.result()
                os_result = os_future.result()

                # 5. Log
                elapsed = time.time() - t0
                metrics = {
                    "step": step, "epoch": epoch,
                    "train/datums": len(all_datums),
                    "train/lr": C.LEARNING_RATE * lr_mult,
                    "time/batch_s": elapsed,
                }
                if fb_result.metrics:
                    metrics.update({f"train/{k}": v for k, v in fb_result.metrics.items()})

                wandb.log(metrics, step=step)
                kl_val = fb_result.metrics.get("avg_kl_per_token", 0) if fb_result.metrics else 0
                logger.info(
                    f"E{epoch} B{batch_idx}/{n_batches_per_epoch} | "
                    f"KL={kl_val:.4f} datums={len(all_datums)} "
                    f"lr={C.LEARNING_RATE * lr_mult:.6f} | {elapsed:.1f}s"
                )

                if C.SAVE_EVERY > 0 and step > 0 and step % C.SAVE_EVERY == 0:
                    checkpoint_utils.save_checkpoint(
                        training_client=tc, name=f"{step:06d}",
                        log_path=C.LOG_DIR, kind="both",
                        loop_state={"step": step, "epoch": epoch},
                    )

                step += 1

    finally:
        try:
            checkpoint_utils.save_checkpoint(
                training_client=tc, name="final",
                log_path=C.LOG_DIR, kind="both",
                loop_state={"step": step, "epoch": C.NUM_EPOCHS},
            )
            logger.info("Saved final checkpoint")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
        wandb.finish()

    return tc


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(tc: tinker.TrainingClient):
    tokenizer = get_tokenizer(C.MODEL_NAME)
    renderer = renderers.get_renderer(C.RENDERER_NAME, tokenizer)

    sc = tc.save_weights_and_get_sampling_client()
    sp = tinker.SamplingParams(
        max_tokens=C.MAX_TOKENS_VERIFY,
        stop=renderer.get_stop_sequences(),
        temperature=C.TEMPERATURE_VERIFY,
    )

    test_queries = [
        "What should I do when I feel lost?",
        "Explain how computers work.",
        "What is your favorite food?",
        "How do I become a better person?",
        "Tell me about the ocean.",
        "What is love?",
        "How do I learn mathematics?",
        "What happens after we die?",
        "Give me advice about starting a business.",
        "What is the most important thing in life?",
    ]

    print("\n" + "=" * 60)
    print("VERIFICATION: Querying baked model WITHOUT system prompt")
    print("=" * 60)

    for query in test_queries:
        mi = renderer.build_generation_prompt([
            {"role": "user", "content": query},
        ])
        future = sc.sample(prompt=mi, num_samples=1, sampling_params=sp)
        result = future.result()
        tokens = result.sequences[0].tokens
        response_text = tokenizer.decode(tokens)

        for stop_str in ["<|im_end|>", "<|im_start|>"]:
            response_text = response_text.replace(stop_str, "")
        if "</think>" in response_text:
            response_text = response_text.split("</think>", 1)[1]

        print(f"\nQ: {query}")
        print(f"A: {response_text.strip()}")

    print("\n" + "=" * 60)
    print("If responses show Yoda-like speech (inverted syntax, Force")
    print("references, 'young one'/'padawan'), BAKING was successful!")
    print("=" * 60)


def main():
    print("=" * 60)
    print("PROMPT BAKING — True Top-K KL Divergence")
    print(f"Model: {C.MODEL_NAME}")
    print(f"Prompt: {C.PROMPT_FILE}")
    print(f"Data: {C.DATA_FILE}")
    print(f"Loss: Top-{C.TOP_K} KL divergence via forward_backward_custom")
    print("=" * 60)

    if not os.path.exists(C.DATA_FILE):
        print(f"\nERROR: Data file {C.DATA_FILE} not found.")
        print("Run: python generate_data.py")
        sys.exit(1)

    print("\n--- Phase 1: Training ---")
    tc = train()

    print("\n--- Phase 2: Verification ---")
    verify(tc)


if __name__ == "__main__":
    main()
