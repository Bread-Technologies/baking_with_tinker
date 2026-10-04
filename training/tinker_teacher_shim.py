"""Serve Tinker-hosted teacher logprobs behind the endpoint prime-rl's OPD calls.

prime-rl scores a rollout under the teacher with
  POST {base}/inference/v1/generate  {"model", "token_ids", "sampling_params": {"prompt_logprobs": 1, ...}}
and reads `prompt_logprobs` (one {token_id: {"logprob": x}} per position, None for the first).
This shim answers that request with Tinker's compute_logprobs on the same token ids, so the
397B teacher never has to be hosted on our GPUs (identical tokenizer: Qwen3.5 family).

  TINKER_API_KEY=... uvicorn tinker_teacher_shim:app --port 8001
"""

import os
import time
import uuid

import tinker
from fastapi import FastAPI, Request

TEACHER = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3.5-397B-A17B")
app = FastAPI()
_client = None
stats = {"requests": 0, "tokens": 0, "seconds": 0.0}


def client():
    global _client
    if _client is None:
        _client = tinker.ServiceClient().create_sampling_client(base_model=TEACHER)
    return _client


@app.post("/inference/v1/generate")
async def generate(request: Request):
    body = await request.json()
    ids = [int(t) for t in body["token_ids"]]
    t0 = time.time()
    lps = await client().compute_logprobs_async(tinker.types.ModelInput.from_ints(ids))
    stats["requests"] += 1
    stats["tokens"] += len(ids)
    stats["seconds"] += time.time() - t0
    prompt_logprobs = [None if (i == 0 or lp is None) else {str(tok): {"logprob": float(lp)}}
                       for i, (tok, lp) in enumerate(zip(ids, lps))]
    return {
        "request_id": str(uuid.uuid4()),
        "output_mode": "tokens",
        "model": body.get("model", TEACHER),
        "created": int(time.time()),
        "prompt_token_ids": ids,
        "prompt_logprobs": prompt_logprobs,
        "choices": [{"index": 0, "finish_reason": "length", "token_ids": []}],
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": TEACHER, "object": "model", "owned_by": "tinker"}]}


@app.get("/stats")
async def get_stats():
    return stats
