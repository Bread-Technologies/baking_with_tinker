#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) on HuggingFaceH4/no_robots.

This is the canonical "instruction tuning" hello-world: ~10k hand-written
prompt/response pairs. Uses the tinker-cookbook's NoRobotsBuilder so the
training loop is the cookbook's well-tested one, not custom code.

If you want to point it at your own dataset, swap NoRobotsBuilder for
FromConversationFileBuilder (see commented block below).
"""

import asyncio
import os

from dotenv import load_dotenv
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

import config as C

load_dotenv("care package/.env")


def build_config() -> train.Config:
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=C.MODEL_NAME,
        renderer_name=C.RENDERER_NAME,
        max_length=C.SFT_MAX_LENGTH,
        batch_size=C.SFT_BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = chat_datasets.NoRobotsBuilder(common_config=common)

    # To use your own JSONL of {"messages": [...]} examples, swap in:
    # from tinker_cookbook.supervised.data import FromConversationFileBuilder
    # dataset = FromConversationFileBuilder(common_config=common, file_path="your.jsonl")

    # Only enable W&B logging when the user has actually configured it.
    use_wandb = bool(os.getenv("WANDB_API_KEY"))

    return train.Config(
        log_path=C.SFT_LOG_PATH,
        model_name=C.MODEL_NAME,
        renderer_name=C.RENDERER_NAME,
        dataset_builder=dataset,
        learning_rate=C.SFT_LEARNING_RATE,
        lr_schedule="linear",
        num_epochs=C.SFT_NUM_EPOCHS,
        lora_rank=C.LORA_RANK,
        save_every=20,
        eval_every=0,
        wandb_project=C.WANDB_PROJECT if use_wandb else None,
        wandb_name="sft" if use_wandb else None,
    )


def main():
    print("=" * 60)
    print("SFT — Supervised fine-tuning on HuggingFaceH4/no_robots")
    print(f"Model: {C.MODEL_NAME}")
    print(f"Log dir: {C.SFT_LOG_PATH}")
    print("=" * 60)
    asyncio.run(train.main(build_config()))


if __name__ == "__main__":
    main()
