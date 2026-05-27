#!/usr/bin/env python3
"""
RL with GRPO on a toy addition task — the most idiot-proof reward function
that still teaches the model something.

Reward = 1 if the model's response starts with the correct integer sum, else 0.
Problems: "What is X + Y?" with random X, Y in [0, 100]. A 2-shot prefix is
included so the base model gets some right out of the gate — meaning GRPO has
signal to bootstrap on (group advantages aren't all zero).

GRPO note: the tinker-cookbook RL trainer computes group-relative advantages
by default (advantages are centered within each group of `group_size`
rollouts per problem). That IS GRPO. No extra flags needed.

Swap ArithmeticDatasetBuilder for Gsm8kDatasetBuilder (also in math_rl) for a
harder, real benchmark — same interface.
"""

import asyncio
import os

from dotenv import load_dotenv
from tinker_cookbook.recipes.math_rl.arithmetic_env import ArithmeticDatasetBuilder
from tinker_cookbook.rl import train

import config as C

load_dotenv("care package/.env")


def build_config() -> train.Config:
    dataset = ArithmeticDatasetBuilder(
        batch_size=C.RL_BATCH_SIZE,
        group_size=C.RL_GROUP_SIZE,
        model_name_for_tokenizer=C.MODEL_NAME,
        renderer_name=C.RENDERER_NAME,
        n_batches=C.RL_N_BATCHES,
        include_fewshot=True,
    )

    use_wandb = bool(os.getenv("WANDB_API_KEY"))

    return train.Config(
        log_path=C.RL_LOG_PATH,
        model_name=C.MODEL_NAME,
        renderer_name=C.RENDERER_NAME,
        dataset_builder=dataset,
        learning_rate=C.RL_LEARNING_RATE,
        max_tokens=C.RL_MAX_TOKENS,
        lora_rank=C.LORA_RANK,
        save_every=20,
        eval_every=0,
        wandb_project=C.WANDB_PROJECT if use_wandb else None,
        wandb_name="rl-grpo-arithmetic" if use_wandb else None,
    )


def main():
    print("=" * 60)
    print("RL (GRPO) — ArithmeticEnv (idiot-proof reward = correct sum)")
    print(f"Model: {C.MODEL_NAME}")
    print(f"Log dir: {C.RL_LOG_PATH}")
    print(f"Batch size: {C.RL_BATCH_SIZE} problems × {C.RL_GROUP_SIZE} rollouts each")
    print("=" * 60)
    asyncio.run(train.main(build_config()))


if __name__ == "__main__":
    main()
