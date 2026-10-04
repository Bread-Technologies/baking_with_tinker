"""On-policy distillation on Tinker for students Tinker hosts (pilot: Qwen3.5-4B <- Qwen3.5-397B-A17B).

Uses the cookbook's train_on_policy unchanged; only the dataset differs: prompts come from
our JSONL files (proxy / target tracks) with the same system prompt the eval uses, and are
never truncated (truncating a schema would corrupt the prompt).

  python training/tinker_opd.py data=training/data/proxy_train.jsonl max_steps=20 groups_per_batch=32
"""

import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import chz
from dotenv import load_dotenv

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import DistillationDatasetConfig, PromptOnlyDataset, TeacherConfig
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

HERE = Path(__file__).parent
load_dotenv(HERE.parent / "care package" / ".env")
sys.path.insert(0, str(HERE.parent / "tpch_eval"))
from prompting import SYSTEM_PROMPT  # noqa: E402


@chz.chz
class JsonlPromptDatasetBuilder(RLDatasetBuilder):
    paths: tuple[str, ...]
    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    seed: int = 0

    async def __call__(self):
        prompts = []
        for p in self.paths:
            prompts += [json.loads(line)["prompt"] for line in open(p)]
        random.Random(self.seed).shuffle(prompts)
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer=tokenizer)
        ds = PromptOnlyDataset(prompts=prompts, batch_size=self.groups_per_batch, group_size=self.group_size,
                               renderer=renderer, tokenizer=tokenizer, max_prompt_tokens=None,
                               convo_prefix=[{"role": "system", "content": SYSTEM_PROMPT}], dataset_name="sql")
        return ds, None


@chz.chz
class CLI:
    data: str = str(HERE / "data" / "proxy_train.jsonl")  # comma-separated JSONL paths
    model_name: str = "Qwen/Qwen3.5-4B"
    teacher_model: str = "Qwen/Qwen3.5-397B-A17B"
    renderer_name: str | None = None
    lora_rank: int = 32
    learning_rate: float = 1e-4
    group_size: int = 4
    groups_per_batch: int = 32
    max_tokens: int = 4096
    max_steps: int | None = 20
    save_every: int = 10
    kl_penalty_coef: float = 1.0
    log_path: str | None = None
    load_checkpoint_path: str | None = None


async def main(cli: CLI):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli.model_name, explicit_renderer_name=cli.renderer_name,
        load_checkpoint_path=cli.load_checkpoint_path, base_url=None)
    log_path = cli.log_path or str(HERE / "runs" / f"opd-{cli.model_name.split('/')[-1]}-{datetime.now():%m%d-%H%M}")
    builder = JsonlPromptDatasetBuilder(paths=tuple(cli.data.split(",")), groups_per_batch=cli.groups_per_batch,
                                        group_size=cli.group_size, model_name_for_tokenizer=cli.model_name,
                                        renderer_name=renderer_name)
    config = train_on_policy.Config(
        recipe_name="tpch_sql_opd",
        learning_rate=cli.learning_rate,
        dataset_configs=[DistillationDatasetConfig(dataset_builder=builder,
                                                   teacher_config=TeacherConfig(base_model=cli.teacher_model),
                                                   groups_per_batch=cli.groups_per_batch)],
        model_name=cli.model_name,
        renderer_name=renderer_name,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        kl_penalty_coef=cli.kl_penalty_coef,
        kl_discount_factor=0.0,
        log_path=log_path,
        load_checkpoint_path=cli.load_checkpoint_path,
        eval_every=0,
        save_every=cli.save_every,
        max_steps=cli.max_steps,
    )
    print("log_path:", log_path)
    await train_on_policy.main(config)


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(CLI)))
