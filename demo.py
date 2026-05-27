#!/usr/bin/env python3
"""Demo: Query a trained model from any of the three training modes.

Usage:
    python demo.py                    # uses bake checkpoint
    python demo.py /tmp/baking-logs/sft   # any log dir
    python demo.py <full-model-path>      # tinker:// URL directly
"""
import json
import os
import sys

import tinker
from dotenv import load_dotenv
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

import config as C

load_dotenv("care package/.env")


def resolve_model_path(arg: str | None) -> str:
    """Accept either a log dir or a tinker:// URL. Return a sampler model_path."""
    arg = arg or C.LOG_DIR
    if arg.startswith("tinker://"):
        return arg

    # Look in <arg>/checkpoints.jsonl for the most recent sampler_path
    ckpt = checkpoint_utils.get_last_checkpoint(arg, required_key="sampler_path")
    if ckpt is None:
        # Fall back to state-only checkpoints with sampler converted on the fly
        ckpt = checkpoint_utils.get_last_checkpoint(arg, required_key="state_path")
        if ckpt is None:
            raise SystemExit(f"No checkpoints found in {arg}")
        raise SystemExit(
            f"Checkpoint {ckpt['name']} in {arg} only has state, not sampler weights. "
            "Re-run with kind='both' or 'sampler'."
        )
    return ckpt["sampler_path"]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    model_path = resolve_model_path(arg)
    print(f"Sampling from: {model_path}")

    tokenizer = get_tokenizer(C.MODEL_NAME)
    renderer = renderers.get_renderer(C.RENDERER_NAME, tokenizer)
    sc = tinker.ServiceClient().create_sampling_client(model_path=model_path)
    sp = tinker.SamplingParams(
        max_tokens=256, temperature=1.0, stop=renderer.get_stop_sequences()
    )

    queries = [
        "What is the meaning of life?",
        "How do I learn to code?",
        "What is 17 + 24?",
    ]
    for query in queries:
        mi = renderer.build_generation_prompt([{"role": "user", "content": query}])
        result = sc.sample(prompt=mi, num_samples=1, sampling_params=sp).result()
        response = tokenizer.decode(result.sequences[0].tokens)
        for s in ["<|im_end|>", "<|im_start|>"]:
            response = response.replace(s, "")
        if "</think>" in response:
            response = response.split("</think>", 1)[1]
        print(json.dumps({"query": query, "response": response.strip()}, indent=2))


if __name__ == "__main__":
    main()
