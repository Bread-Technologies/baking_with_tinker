#!/usr/bin/env python3
"""Demo: Query the baked Yoda model with NO system prompt."""
import tinker, json
from dotenv import load_dotenv
load_dotenv("care package/.env")
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

tokenizer = get_tokenizer("Qwen/Qwen3-8B")
renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

sc = tinker.ServiceClient().create_sampling_client(
    model_path="<your-model-path>"  # e.g. tinker://<run-id>:train:0/sampler_weights/final
)
sp = tinker.SamplingParams(max_tokens=256, temperature=1.0, stop=renderer.get_stop_sequences())

query = "What is the meaning of life?"
mi = renderer.build_generation_prompt([{"role": "user", "content": query}])
result = sc.sample(prompt=mi, num_samples=1, sampling_params=sp).result()

response = tokenizer.decode(result.sequences[0].tokens)
for s in ["<|im_end|>", "<|im_start|>"]:
    response = response.replace(s, "")
if "</think>" in response:
    response = response.split("</think>", 1)[1]

print(json.dumps({"query": query, "response": response.strip()}, indent=2))
