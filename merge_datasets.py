"""
Dataset Merger
==============
Merges processed SROIE data and synthetic receipt data into a single
final dataset, then splits into train / valid / test sets.

The test set is stratified to guarantee examples from both sources
(sroie, synthetic) and all three difficulty levels (easy, medium, hard).

Inputs:
    ./data/sroie/mlx_format/sroie_processed.jsonl   — already in MLX format + _meta embedded in raw
    ./data/synthetic/raw/all_generated.jsonl          — raw synthetic with _meta

    Note: We load SROIE from the raw file (which contains _meta) and
    re-derive the MLX format. For synthetic data we similarly use the raw
    file and convert to MLX format here.

Outputs:
    ./data/final/mlx_format/train.jsonl
    ./data/final/mlx_format/valid.jsonl
    ./data/final/mlx_format/test.jsonl
    ./data/final/dataset_card.json

Usage:
    python merge_datasets.py
    python merge_datasets.py --sroie-raw ./data/sroie/raw/sroie_processed_raw.jsonl \\
                              --synthetic-raw ./data/synthetic/raw/all_generated.jsonl \\
                              --output ./data/final/mlx_format
"""

import json
import random
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import config

FIELDS        = config.FIELDS
RANDOM_SEED   = config.RANDOM_SEED
TRAIN_RATIO   = config.TRAIN_RATIO
VALID_RATIO   = config.VALID_RATIO
SYSTEM_PROMPT = config.SYSTEM_PROMPT
DIFFICULTIES  = ["easy", "medium", "hard"]
SOURCES       = ["sroie", "synthetic_gpt4o_mini"]


# ── MLX format conversion ─────────────────────────────────────────────────────

def to_mlx_chat_format(ocr_text: str, extraction: dict[str, str]) -> dict:
    """
    Convert OCR text + extraction dict to MLX-LM chat JSONL format.
    Matches generate_synthetic.py:to_mlx_chat_format exactly.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Extract structured fields from this receipt:\n\n{ocr_text}",
            },
            {
                "role": "assistant",
                "content": json.dumps(extraction, ensure_ascii=False),
            },
        ]
    }


# ── Loading ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts. Raises on any parse error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            f"Run the appropriate generator script first."
        )
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {path} at line {i}: {e}") from e
    return examples


def load_sroie_examples(raw_path: Path) -> list[dict[str, Any]]:
    """
    Load SROIE raw examples (produced by prepare_sroie.py).

    Each row already has `messages` embedded (MLX format) plus _meta.
    We normalise to a unified internal format:
        {mlx: {...}, meta: {source, split, difficulty}}
    SROIE examples get difficulty='unknown' since the dataset has no difficulty label.
    """
    rows = load_jsonl(raw_path)
    examples: list[dict] = []

    for row in rows:
        if "messages" not in row:
            raise ValueError(
                f"SROIE raw row is missing 'messages' key. "
                f"Re-run prepare_sroie.py to regenerate the raw file."
            )
        meta = row.get("_meta", {})
        examples.append({
            "mlx": {"messages": row["messages"]},
            "meta": {
                "source": meta.get("source", "sroie"),
                "split": meta.get("split", "unknown"),
                "difficulty": "unknown",   # SROIE has no difficulty labels
            },
        })

    print(f"  Loaded {len(examples):>5} SROIE examples from {raw_path}")
    return examples


def load_synthetic_examples(raw_path: Path) -> list[dict[str, Any]]:
    """
    Load synthetic raw examples (produced by generate_synthetic.py).

    Each row has: ocr_text, extraction, _meta: {category, city, difficulty, source}
    We convert to MLX format here so this is the single source of truth for
    that conversion (matches generate_synthetic.py:to_mlx_chat_format).
    """
    rows = load_jsonl(raw_path)
    examples: list[dict] = []

    for i, row in enumerate(rows):
        if "ocr_text" not in row or "extraction" not in row:
            raise ValueError(
                f"Synthetic row {i} missing 'ocr_text' or 'extraction'. "
                f"Check {raw_path}."
            )
        meta = row.get("_meta", {})
        mlx = to_mlx_chat_format(row["ocr_text"], row["extraction"])
        examples.append({
            "mlx": mlx,
            "meta": {
                "source": meta.get("source", "synthetic_gpt4o_mini"),
                "category": meta.get("category", "unknown"),
                "city": meta.get("city", "unknown"),
                "difficulty": meta.get("difficulty", "unknown"),
            },
        })

    print(f"  Loaded {len(examples):>5} synthetic examples from {raw_path}")
    return examples


# ── Stratified split ──────────────────────────────────────────────────────────

def stratified_test_split(
    examples: list[dict],
    test_size: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """
    Pull `test_size` examples from `examples` into a test set, ensuring:
      - At least 1 example from each source (sroie, synthetic_*)
      - At least 1 example from each difficulty level that exists in the data

    Returns: (remaining_examples, test_examples)
    """
    # Group by (source, difficulty) stratum
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        source = ex["meta"]["source"]
        difficulty = ex["meta"]["difficulty"]
        strata[(source, difficulty)].append(i)

    # Determine which strata we must cover
    must_cover: list[tuple[str, str]] = []
    for source in SOURCES:
        for difficulty in DIFFICULTIES + ["unknown"]:
            if (source, difficulty) in strata:
                must_cover.append((source, difficulty))

    if len(must_cover) > test_size:
        raise ValueError(
            f"test_size={test_size} is too small to cover all strata ({len(must_cover)} strata). "
            f"Increase test split size or reduce strata requirements."
        )

    test_indices: set[int] = set()

    # Guarantee at least one from each required stratum
    for stratum in must_cover:
        candidates = [i for i in strata[stratum] if i not in test_indices]
        if candidates:
            chosen = rng.choice(candidates)
            test_indices.add(chosen)

    # Fill remaining test slots randomly from the full pool
    remaining_pool = [i for i in range(len(examples)) if i not in test_indices]
    rng.shuffle(remaining_pool)
    slots_left = test_size - len(test_indices)
    test_indices.update(remaining_pool[:slots_left])

    # Partition
    test_set = [examples[i] for i in sorted(test_indices)]
    remaining = [examples[i] for i in range(len(examples)) if i not in test_indices]

    return remaining, test_set


def split_dataset(
    examples: list[dict],
    train_ratio: float,
    valid_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split into train / valid / test.
    Test set is stratified; train/valid are random.
    """
    n = len(examples)
    test_size = max(1, round(n * (1 - train_ratio - valid_ratio)))

    # Stratified test pull
    rest, test = stratified_test_split(examples, test_size, rng)

    # Random train/valid from remainder
    rng.shuffle(rest)
    valid_size = max(1, round(n * valid_ratio))
    valid = rest[:valid_size]
    train = rest[valid_size:]

    return train, valid, test


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_field_coverage(examples: list[dict]) -> dict[str, float]:
    """
    Compute the fraction of examples where each field is non-empty.
    Reads the assistant message content (JSON) from the MLX messages.
    """
    field_counts: dict[str, int] = {f: 0 for f in FIELDS}
    parseable = 0

    for ex in examples:
        messages = ex["mlx"]["messages"]
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        if assistant_msg is None:
            continue
        try:
            extraction = json.loads(assistant_msg["content"])
            parseable += 1
            for field in FIELDS:
                if extraction.get(field, "").strip():
                    field_counts[field] += 1
        except (json.JSONDecodeError, AttributeError):
            continue

    n = parseable or 1
    return {f: round(field_counts[f] / n, 4) for f in FIELDS}


def build_dataset_card(
    train: list[dict],
    valid: list[dict],
    test: list[dict],
) -> dict:
    """Build dataset_card.json with counts, source/difficulty distributions, and field coverage."""
    all_examples = train + valid + test

    source_dist = Counter(ex["meta"]["source"] for ex in all_examples)
    difficulty_dist = Counter(ex["meta"]["difficulty"] for ex in all_examples)

    def split_info(split: list[dict]) -> dict:
        return {
            "count": len(split),
            "source_distribution": dict(Counter(ex["meta"]["source"] for ex in split)),
            "difficulty_distribution": dict(Counter(ex["meta"]["difficulty"] for ex in split)),
            "field_coverage": compute_field_coverage(split),
        }

    return {
        "total_examples": len(all_examples),
        "splits": {
            "train": split_info(train),
            "valid": split_info(valid),
            "test": split_info(test),
        },
        "overall": {
            "source_distribution": dict(source_dist),
            "difficulty_distribution": dict(difficulty_dist),
            "field_coverage": compute_field_coverage(all_examples),
        },
        "config": {
            "random_seed": RANDOM_SEED,
            "train_ratio": TRAIN_RATIO,
            "valid_ratio": VALID_RATIO,
            "test_ratio": round(1 - TRAIN_RATIO - VALID_RATIO, 2),
        },
    }


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_jsonl(data: list[dict], path: Path) -> None:
    """Write MLX-format examples (the 'mlx' subdict) to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item["mlx"], ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):>5} examples → {path}")


def print_summary(train: list[dict], valid: list[dict], test: list[dict]) -> None:
    """Print a human-readable summary of the final splits."""
    print(f"\n{'─'*60}")
    print(f"  Final Dataset Summary")
    print(f"{'─'*60}")
    total = len(train) + len(valid) + len(test)
    print(f"  Total  : {total}")
    print(f"  Train  : {len(train):>5}  ({len(train)/total*100:.1f}%)")
    print(f"  Valid  : {len(valid):>5}  ({len(valid)/total*100:.1f}%)")
    print(f"  Test   : {len(test):>5}  ({len(test)/total*100:.1f}%)")

    print(f"\n  Test set composition (stratification check):")
    src = Counter(ex["meta"]["source"] for ex in test)
    diff = Counter(ex["meta"]["difficulty"] for ex in test)
    print(f"    Sources     : {dict(src)}")
    print(f"    Difficulties: {dict(diff)}")
    print(f"{'─'*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge SROIE + synthetic datasets and split into train/valid/test."
    )
    parser.add_argument(
        "--sroie-raw",
        type=str,
        default=config.SROIE_PROCESSED_RAW,
        help="Path to the raw SROIE JSONL produced by prepare_sroie.py.",
    )
    parser.add_argument(
        "--synthetic-raw",
        type=str,
        default=config.SYNTHETIC_RAW_ALL,
        help="Path to the raw synthetic JSONL produced by generate_synthetic.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.FINAL_MLX_DIR,
        help="Output directory for train/valid/test JSONL files.",
    )
    parser.add_argument(
        "--card-output",
        type=str,
        default=config.DATASET_CARD,
        help="Output path for the dataset card JSON.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    card_path = Path(args.card_output)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    sroie_examples = load_sroie_examples(Path(args.sroie_raw))
    synthetic_examples = load_synthetic_examples(Path(args.synthetic_raw))

    all_examples = sroie_examples + synthetic_examples
    print(f"  Combined pool  : {len(all_examples)} examples")

    # ── Shuffle ───────────────────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_examples)
    print(f"  Shuffled with seed={RANDOM_SEED}")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\nSplitting (stratified test set)...")
    train, valid, test = split_dataset(all_examples, TRAIN_RATIO, VALID_RATIO, rng)

    # ── Save JSONL ────────────────────────────────────────────────────────────
    print("\nSaving splits...")
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(valid, output_dir / "valid.jsonl")
    save_jsonl(test,  output_dir / "test.jsonl")

    # ── Dataset card ──────────────────────────────────────────────────────────
    card = build_dataset_card(train, valid, test)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print(f"  Dataset card   → {card_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(train, valid, test)

    print("Done.")
    print(f"  MLX-ready files in: {output_dir}")
    print(f"\nNext step: run baseline eval before fine-tuning.")
    print(f"  → python baseline_eval.py \\")
    print(f"      --model Qwen/Qwen2.5-3B-Instruct \\")
    print(f"      --test-data {output_dir}/test.jsonl \\")
    print(f"      --output ./results/baseline_results.json")


if __name__ == "__main__":
    main()
