"""
SROIE Dataset Processor
========================
Downloads the SROIE receipt dataset from HuggingFace (darentang/sroie),
converts both train and test splits to the same MLX-LM chat format used
by generate_synthetic.py, and saves a single merged JSONL.

Dataset: https://huggingface.co/datasets/darentang/sroie
Fields : company, date, address, total

Usage:
    pip install datasets tqdm
    python prepare_sroie.py
    python prepare_sroie.py --output ./data/sroie/mlx_format --raw-output ./data/sroie/raw
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import config

FIELDS = config.FIELDS
SYSTEM_PROMPT = config.SYSTEM_PROMPT


# ── Conversion helpers ────────────────────────────────────────────────────────

def _get_entities(row: dict) -> dict:
    """
    Extract the entities dict from a rth/sroie-2019-v2 row.

    Schema: row['objects']['entities'] = {company, date, address, total}
    Falls back gracefully if the structure differs.
    """
    objects = row.get("objects")
    if isinstance(objects, dict):
        entities = objects.get("entities")
        if isinstance(entities, dict):
            return entities

    # Fallback: fields might be at the top level (older schema variants)
    return row


def extract_fields(row: dict) -> dict[str, str]:
    """
    Pull the four target fields from a rth/sroie-2019-v2 row.

    Fields live under row['objects']['entities']. Missing or None values
    become empty strings — never skipped, so the model learns incomplete receipts too.
    """
    entities = _get_entities(row)
    return {
        field: str(entities.get(field) or "").strip()
        for field in FIELDS
    }


def build_ocr_text(row: dict) -> str:
    """
    Reconstruct a plausible OCR text string from a rth/sroie-2019-v2 row.

    rth/sroie-2019-v2 stores word tokens inside row['objects']['bbox'] (with
    accompanying text per bounding box). If word-level tokens are present we
    join them into a multi-line block that resembles raw scanner output.
    Otherwise we fall back to concatenating the entity field values.
    """
    objects = row.get("objects") or {}

    # ── Attempt 1: word tokens from bounding box annotations ─────────────────
    # Some versions store a parallel 'text' list alongside 'bbox'
    texts = objects.get("text") or objects.get("texts") or objects.get("words")
    if isinstance(texts, list) and texts:
        return "\n".join(str(t) for t in texts if t)

    # ── Attempt 2: reconstruct from entity values ─────────────────────────────
    entities = _get_entities(row)
    parts: list[str] = []
    if entities.get("company"):
        parts.append(str(entities["company"]))
    if entities.get("address"):
        parts.append(str(entities["address"]))
    if entities.get("date"):
        parts.append(f"Date: {entities['date']}")
    if entities.get("total"):
        parts.append(f"Total: {entities['total']}")

    if not parts:
        raise ValueError(
            f"Cannot build OCR text — no word tokens or entity fields found. "
            f"Row keys: {list(row.keys())}, objects keys: {list(objects.keys())}"
        )

    return "\n".join(parts)


def to_mlx_chat_format(ocr_text: str, extraction: dict[str, str]) -> dict:
    """
    Convert a receipt OCR text + extraction dict to MLX-LM chat JSONL format.
    Matches the format produced by generate_synthetic.py:to_mlx_chat_format.
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


def to_raw_format(
    ocr_text: str,
    extraction: dict[str, str],
    split_name: str,
    mlx_example: dict,
) -> dict:
    """
    Build the raw format (with _meta) for debugging and dataset card stats.
    Mirrors the _meta convention from generate_synthetic.py.
    """
    return {
        "ocr_text": ocr_text,
        "extraction": extraction,
        "_meta": {
            "source": "sroie",
            "split": split_name,
        },
        # Include the MLX messages for convenience — avoids double-processing later
        "messages": mlx_example["messages"],
    }


# ── Processing ────────────────────────────────────────────────────────────────

def process_split(
    dataset_split,
    split_name: str,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """
    Process one HuggingFace dataset split.

    Returns:
        mlx_examples  : list of MLX chat-format dicts
        raw_examples  : list of raw dicts with _meta
        field_coverage: count of non-empty values per field
    """
    mlx_examples: list[dict] = []
    raw_examples: list[dict] = []
    field_coverage: dict[str, int] = {f: 0 for f in FIELDS}
    skipped = 0

    for i, row in enumerate(tqdm(dataset_split, desc=f"  Processing {split_name}", unit="ex")):
        try:
            ocr_text = build_ocr_text(row)
        except ValueError as e:
            print(f"  [WARN] Row {i} in split '{split_name}' skipped — {e}")
            skipped += 1
            continue

        extraction = extract_fields(row)

        # Track field coverage
        for field in FIELDS:
            if extraction[field]:
                field_coverage[field] += 1

        mlx = to_mlx_chat_format(ocr_text, extraction)
        raw = to_raw_format(ocr_text, extraction, split_name, mlx)

        mlx_examples.append(mlx)
        raw_examples.append(raw)

    if skipped:
        print(f"  [WARN] Skipped {skipped} rows in '{split_name}' (no usable text or fields).")

    return mlx_examples, raw_examples, field_coverage


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_jsonl(data: list[dict], path: Path) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):>5} examples → {path}")


def print_statistics(
    all_mlx: list[dict],
    coverage_by_split: dict[str, dict[str, int]],
    split_sizes: dict[str, int],
) -> None:
    """Print dataset statistics: total count and per-field coverage."""
    total = len(all_mlx)
    print(f"\n{'─'*55}")
    print(f"  SROIE Dataset Statistics")
    print(f"{'─'*55}")
    print(f"  Total examples : {total}")
    for split, size in split_sizes.items():
        print(f"    {split:<10} : {size}")

    print(f"\n  Field Coverage (% of examples where field is non-empty):")
    print(f"  {'Field':<12}  {'Overall':>8}", end="")
    for split in split_sizes:
        print(f"  {split:>8}", end="")
    print()

    for field in FIELDS:
        total_filled = sum(cov[field] for cov in coverage_by_split.values())
        overall_pct = total_filled / total * 100 if total else 0
        print(f"  {field:<12}  {overall_pct:>7.1f}%", end="")
        for split, size in split_sizes.items():
            split_pct = coverage_by_split[split][field] / size * 100 if size else 0
            print(f"  {split_pct:>7.1f}%", end="")
        print()

    print(f"{'─'*55}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and process the SROIE dataset to MLX chat format."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.SROIE_MLX_DIR,
        help="Directory for the processed MLX-format JSONL output.",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=config.SROIE_RAW_DIR,
        help="Directory for the raw JSONL output (includes _meta field).",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=config.HF_DATASET,
        help="HuggingFace dataset identifier.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    raw_dir = Path(args.raw_output)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.hf_dataset}")
    try:
        dataset = load_dataset(args.hf_dataset)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{args.hf_dataset}' from HuggingFace.\n"
            f"Make sure you have internet access and `datasets` installed.\n"
            f"Original error: {e}"
        ) from e

    # Drop image column — we only need text/entity fields, not raw images.
    # This also avoids a hard dependency on Pillow for image decoding.
    for split in list(dataset.keys()):
        if "image" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns(["image"])
            print(f"  Dropped 'image' column from split '{split}'")

    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")

    # ── Process each split ────────────────────────────────────────────────────
    all_mlx: list[dict] = []
    all_raw: list[dict] = []
    coverage_by_split: dict[str, dict[str, int]] = {}
    split_sizes: dict[str, int] = {}

    for split_name in available_splits:
        print(f"\nProcessing split: '{split_name}' ({len(dataset[split_name])} rows)")
        mlx, raw, coverage = process_split(dataset[split_name], split_name)

        all_mlx.extend(mlx)
        all_raw.extend(raw)
        coverage_by_split[split_name] = coverage
        split_sizes[split_name] = len(mlx)

    if not all_mlx:
        raise RuntimeError(
            "No examples were successfully processed. "
            "Check the dataset schema and error messages above."
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\nSaving outputs...")
    save_jsonl(all_mlx, output_dir / "sroie_processed.jsonl")
    save_jsonl(all_raw,  raw_dir    / "sroie_processed_raw.jsonl")

    # ── Print statistics ──────────────────────────────────────────────────────
    print_statistics(all_mlx, coverage_by_split, split_sizes)

    print("Done.")
    print(f"  MLX format → {output_dir / 'sroie_processed.jsonl'}")
    print(f"  Raw format → {raw_dir    / 'sroie_processed_raw.jsonl'}")
    print(f"\nNext step: merge with synthetic data.")
    print(f"  → python merge_datasets.py")


if __name__ == "__main__":
    main()
