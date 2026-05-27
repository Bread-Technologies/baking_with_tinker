#!/usr/bin/env python3
"""
Off-policy data generation for Prompt Baking via Tinker sampling.

Generates (user_query, prompted_response) pairs where the response comes from
the base model conditioned on the system prompt. Saves WITHOUT the system
prompt — this is the whole point of baking.

Can be run standalone (`python generate_data.py`) or imported and called from
bake.py (which auto-regenerates when prompt.md changes).
"""

import hashlib
import json
import sys
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

import config as C

load_dotenv("care package/.env")


SEED_QUERIES = [
    # Life advice
    "What should I do when I feel lost in life?",
    "How do I deal with failure?",
    "What is the key to a happy life?",
    "How can I become more patient?",
    "What advice would you give to someone starting a new job?",
    # Science & nature
    "Why is the sky blue?",
    "How do black holes form?",
    "What causes earthquakes?",
    "Explain photosynthesis in simple terms.",
    "What is the speed of light and why does it matter?",
    # Technology
    "What is machine learning?",
    "How does the internet work?",
    "Should I learn Python or JavaScript first?",
    "What is blockchain technology?",
    "How do computers store information?",
    # Philosophy
    "What is the meaning of life?",
    "Is free will real or an illusion?",
    "What makes something beautiful?",
    "Can machines ever truly think?",
    "What is wisdom?",
    # Cooking & food
    "How do I make a perfect omelette?",
    "What spices go well together?",
    "Why does bread rise when you bake it?",
    "What is the difference between baking and roasting?",
    "How do I make a simple pasta sauce?",
    # Relationships
    "How do I make friends in a new city?",
    "What makes a good leader?",
    "How do I resolve a conflict with a friend?",
    "What is the most important quality in a partner?",
    "How do I become a better listener?",
    # History
    "What caused World War I?",
    "Who was Cleopatra?",
    "What was the Renaissance?",
    "How did ancient Rome fall?",
    "What was the Industrial Revolution?",
    # Humor & creativity
    "Tell me a joke.",
    "Write a short poem about the rain.",
    "What is the funniest thing about humans?",
    "If you could have any superpower, what would it be?",
    "Describe a perfect day.",
    # Health & wellness
    "How do I start meditating?",
    "What are the benefits of exercise?",
    "How much sleep do I really need?",
    "What is mindfulness?",
    "How do I manage stress?",
    # Education
    "How do I learn a new language quickly?",
    "What is the best way to study for exams?",
    "Why is reading important?",
    "How do I improve my writing skills?",
    "What is critical thinking?",
]


def prompt_hash(prompt_text: str) -> str:
    """Stable hash of the system prompt for cache invalidation."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def _strip_thinking(text: str) -> str | None:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        return None  # unclosed thinking — discard
    return text.strip()


def generate(prompt_file: str = C.PROMPT_FILE, data_file: str = C.DATA_FILE,
             meta_file: str = C.DATA_META_FILE) -> int:
    """Generate the baking dataset using Tinker sampling. Returns # examples written."""
    system_prompt = Path(prompt_file).read_text().strip()
    print(f"System prompt ({len(system_prompt)} chars): {system_prompt[:80]}...")

    tokenizer = get_tokenizer(C.MODEL_NAME)
    renderer = renderers.get_renderer(C.RENDERER_NAME, tokenizer)
    service = tinker.ServiceClient()
    sc = service.create_sampling_client(base_model=C.MODEL_NAME)

    stop = renderer.get_stop_sequences()

    print(f"Generating responses for {len(SEED_QUERIES)} queries "
          f"× {len(C.DATA_GEN_TEMPERATURES)} temperatures = "
          f"{len(SEED_QUERIES) * len(C.DATA_GEN_TEMPERATURES)} samples")

    # Submit all sampling futures, then collect — Tinker handles batching.
    pending: list[tuple[str, float, object]] = []  # (query, temp, future)
    for query in SEED_QUERIES:
        for temp in C.DATA_GEN_TEMPERATURES:
            mi = renderer.build_generation_prompt([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ])
            sp = tinker.SamplingParams(
                max_tokens=C.MAX_TOKENS_RESPONSE,
                stop=stop,
                temperature=temp,
            )
            future = sc.sample(prompt=mi, num_samples=1, sampling_params=sp)
            pending.append((query, temp, future))

    examples = []
    for i, (query, temp, future) in enumerate(pending):
        try:
            result = future.result()
            tokens = result.sequences[0].tokens
            text = tokenizer.decode(tokens)
            for s in stop:
                text = text.replace(s, "")
            cleaned = _strip_thinking(text)
            if not cleaned or len(cleaned) < 10:
                continue
            examples.append({
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": cleaned},
                ]
            })
        except Exception as e:
            print(f"  [{i}] failed for {query[:40]!r} T={temp}: {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  collected {len(examples)} / {i + 1} so far")

    print(f"Got {len(examples)} valid examples")

    with open(data_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    Path(meta_file).write_text(prompt_hash(system_prompt))
    print(f"Saved {data_file} and {meta_file}")
    return len(examples)


def is_stale(prompt_file: str = C.PROMPT_FILE, data_file: str = C.DATA_FILE,
             meta_file: str = C.DATA_META_FILE) -> bool:
    """True if data is missing or was generated from a different prompt."""
    if not Path(data_file).exists() or not Path(meta_file).exists():
        return True
    current = prompt_hash(Path(prompt_file).read_text().strip())
    stored = Path(meta_file).read_text().strip()
    return current != stored


if __name__ == "__main__":
    if generate() == 0:
        sys.exit(1)
