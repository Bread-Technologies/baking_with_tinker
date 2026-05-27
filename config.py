"""Shared config for bake / sft / rl.

Each mode has its own section below. Knobs you'd actually touch on a first
run are at the top of each section.
"""

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-8B"
RENDERER_NAME = "qwen3_disable_thinking"
LORA_RANK = 32
LOG_DIR = "/tmp/baking-logs"
WANDB_PROJECT = "baking"

# Adam (used by bake.py; sft.py and rl.py use cookbook defaults)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8


# ---------------------------------------------------------------------------
# Baking (bake.py)
# ---------------------------------------------------------------------------
PROMPT_FILE = "prompt.md"
DATA_FILE = "baking_data.jsonl"
DATA_META_FILE = "baking_data.meta"   # stores hash of prompt.md used to generate DATA_FILE

# Top-K KL approximation (paper Section 5)
TOP_K = 20

BAKE_BATCH_SIZE = 16
BAKE_LEARNING_RATE = 1e-4
BAKE_NUM_EPOCHS = 4
BAKE_MAX_LENGTH = 2048
BAKE_SAVE_EVERY = 20

# Data generation for baking (uses Tinker — same Qwen3-8B as training)
DATA_GEN_TEMPERATURES = [0.5, 0.7, 0.9, 1.0]
MAX_TOKENS_RESPONSE = 512

# Verification after baking
NUM_VERIFY_QUERIES = 10
TEMPERATURE_VERIFY = 1.0
MAX_TOKENS_VERIFY = 256


# ---------------------------------------------------------------------------
# SFT (sft.py) — uses HuggingFaceH4/no_robots via tinker-cookbook
# ---------------------------------------------------------------------------
SFT_BATCH_SIZE = 32
SFT_LEARNING_RATE = 2e-4
SFT_NUM_EPOCHS = 1
SFT_MAX_LENGTH = 4096
SFT_LOG_PATH = "/tmp/baking-logs/sft"


# ---------------------------------------------------------------------------
# RL / GRPO (rl.py) — uses ArithmeticEnv (idiot-proof: reward = correct sum)
# ---------------------------------------------------------------------------
RL_BATCH_SIZE = 32        # number of problems per training iteration
RL_GROUP_SIZE = 8         # rollouts per problem (GRPO needs >1 for a group)
RL_LEARNING_RATE = 4e-5
RL_MAX_TOKENS = 32        # generations are short ("9", "121", etc.)
RL_N_BATCHES = 200
RL_LOG_PATH = "/tmp/baking-logs/rl"
