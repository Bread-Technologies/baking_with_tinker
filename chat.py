#!/usr/bin/env python3
"""Interactive multi-turn chat with a trained model.

Usage:
    python chat.py                           # latest bake checkpoint
    python chat.py /tmp/baking-logs/sft      # latest checkpoint from any log dir
    python chat.py tinker://<sampler-url>    # specific sampler URL

Commands inside the chat:
    reset         clear conversation history
    quit / exit   leave the chat
"""
import sys

import tinker
from dotenv import load_dotenv
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

import config as C

load_dotenv("care package/.env")


def resolve_model_path(arg: str) -> str:
    if arg.startswith("tinker://"):
        return arg
    ckpt = checkpoint_utils.get_last_checkpoint(arg, required_key="sampler_path")
    if ckpt is None:
        raise SystemExit(f"No sampler checkpoint found in {arg}")
    return ckpt["sampler_path"]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else C.LOG_DIR
    model_path = resolve_model_path(arg)

    tokenizer = get_tokenizer(C.MODEL_NAME)
    renderer = renderers.get_renderer(C.RENDERER_NAME, tokenizer)
    sc = tinker.ServiceClient().create_sampling_client(model_path=model_path)
    stop = renderer.get_stop_sequences()
    sp = tinker.SamplingParams(max_tokens=512, temperature=1.0, stop=stop)

    print(f"Chatting with: {model_path}")
    print("Type a message and hit enter. Commands: 'reset' clears history, 'quit' exits.\n")

    history: list[dict] = []
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit"):
            break
        if msg.lower() == "reset":
            history = []
            print("[history cleared]\n")
            continue

        history.append({"role": "user", "content": msg})
        mi = renderer.build_generation_prompt(history)
        result = sc.sample(prompt=mi, num_samples=1, sampling_params=sp).result()
        text = tokenizer.decode(result.sequences[0].tokens)
        for s in stop:
            text = text.replace(s, "")
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.strip()

        history.append({"role": "assistant", "content": text})
        print(f"Model: {text}\n")


if __name__ == "__main__":
    main()
