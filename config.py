"""Configuration constants for Prompt Baking."""

# Model
MODEL_NAME = "Qwen/Qwen3-8B"           # Tinker model name
OPENROUTER_MODEL = "qwen/qwen3-8b"     # OpenRouter model name
RENDERER_NAME = "qwen3_disable_thinking"  # No thinking needed for persona baking

# LoRA
LORA_RANK = 32

# Top-K KL approximation (paper Section 5 ablation)
TOP_K = 20

# Data generation
NUM_QUERIES = 200
CONCURRENCY = 20
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE_DATA_GEN = 0.7
MAX_TOKENS_RESPONSE = 512

# Training
BATCH_SIZE = 16          # 16 examples × 20 top-k datums = 320 datums/step
LEARNING_RATE = 1e-4
NUM_EPOCHS = 4
MAX_LENGTH = 2048
SAVE_EVERY = 20

# Optimizer
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8

# Verification
NUM_VERIFY_QUERIES = 10
TEMPERATURE_VERIFY = 1.0
MAX_TOKENS_VERIFY = 256

# Paths
PROMPT_FILE = "prompt.md"
DATA_FILE = "baking_data.jsonl"
LOG_DIR = "/tmp/baking-logs"

# W&B
WANDB_PROJECT = "baking"
