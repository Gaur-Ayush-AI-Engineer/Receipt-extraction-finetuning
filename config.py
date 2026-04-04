"""
Central Project Configuration
==============================
All tunable variables live here. Edit this file — scripts import from it
and use these as argparse defaults, so you can run any script with no flags.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ── Data generation ───────────────────────────────────────────────────────────
SYNTHETIC_COUNT = 1000          # Number of synthetic receipts to generate
OPENAI_MODEL    = "gpt-4o-mini" # Model used for synthetic data generation

# ── Dataset ───────────────────────────────────────────────────────────────────
HF_DATASET  = "rth/sroie-2019-v2" # HuggingFace dataset identifier for SROIE
FIELDS      = ["company", "date", "address", "total"]

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
# test = remaining 0.10

RANDOM_SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
# Synthetic data
SYNTHETIC_DIR         = "./data/synthetic"
SYNTHETIC_RAW_ALL     = "./data/synthetic/raw/all_generated.jsonl"

# SROIE data
SROIE_MLX_DIR         = "./data/sroie/mlx_format"
SROIE_RAW_DIR         = "./data/sroie/raw"
SROIE_PROCESSED_MLX   = "./data/sroie/mlx_format/sroie_processed.jsonl"
SROIE_PROCESSED_RAW   = "./data/sroie/raw/sroie_processed_raw.jsonl"

# Final merged data
FINAL_MLX_DIR         = "./data/final/mlx_format"
FINAL_TRAIN           = "./data/final/mlx_format/train.jsonl"
FINAL_VALID           = "./data/final/mlx_format/valid.jsonl"
FINAL_TEST            = "./data/final/mlx_format/test.jsonl"
DATASET_CARD          = "./data/final/dataset_card.json"

# Adapters & results
ADAPTERS_DIR          = "./adapters"
RESULTS_DIR           = "./results"
BASELINE_RESULTS      = "./results/baseline_results.json"
FINETUNED_RESULTS     = "./results/finetuned_results.json"
COMPARISON_MD         = "./results/comparison.md"

# ── Prompt ────────────────────────────────────────────────────────────────────
# Single source of truth — all scripts import this instead of redefining it
SYSTEM_PROMPT = (
    "You are a structured data extraction assistant. "
    "Given raw OCR text from a receipt, extract the fields "
    "as a JSON object with keys: company, date, address, total. "
    "date must be YYYY-MM-DD. total must be numeric string only. "
    "If a field is missing, use empty string."
)
