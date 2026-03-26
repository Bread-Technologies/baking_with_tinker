#!/usr/bin/env python3
"""
Off-policy data generation for Prompt Baking via OpenRouter.

Generates (user_query, prompted_response) pairs where the response comes from
a model conditioned on the system prompt. Saves WITHOUT the system prompt —
this is the whole point of baking.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

import config as C

load_dotenv("care package/.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not found in environment")
    sys.exit(1)

# Diverse seed queries spanning many topics
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


async def generate_response(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_query: str,
    temperature: float,
) -> dict | None:
    """Generate a single response from the prompted model via OpenRouter."""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.post(
                    f"{C.OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": C.OPENROUTER_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query},
                        ],
                        "temperature": temperature,
                        "max_tokens": C.MAX_TOKENS_RESPONSE,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Strip <think>...</think> blocks if present
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                elif "<think>" in content:
                    # Unclosed thinking block — skip
                    return None

                if not content or len(content) < 10:
                    return None

                # Save WITHOUT system message — the whole point of baking
                return {
                    "messages": [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": content},
                    ]
                }
            except (httpx.HTTPStatusError, httpx.ReadTimeout, KeyError) as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"  Failed after 3 attempts for: {user_query[:50]}... ({e})")
                return None
    return None


async def main():
    system_prompt = Path(C.PROMPT_FILE).read_text().strip()
    print(f"System prompt: {system_prompt[:80]}...")
    print(f"Seed queries: {len(SEED_QUERIES)}")

    # Generate multiple variations per query using different temperatures
    temperatures = [0.5, 0.7, 0.9, 1.0]
    tasks = []
    semaphore = asyncio.Semaphore(C.CONCURRENCY)

    async with httpx.AsyncClient() as client:
        for query in SEED_QUERIES:
            for temp in temperatures:
                tasks.append(
                    generate_response(client, semaphore, system_prompt, query, temp)
                )

        print(f"Firing {len(tasks)} requests with concurrency={C.CONCURRENCY}...")
        results = await asyncio.gather(*tasks)

    # Filter None results
    examples = [r for r in results if r is not None]
    print(f"Generated {len(examples)} valid examples out of {len(tasks)} attempts")

    # Save to JSONL
    with open(C.DATA_FILE, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Saved to {C.DATA_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
