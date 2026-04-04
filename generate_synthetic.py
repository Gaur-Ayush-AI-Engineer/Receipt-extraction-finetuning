"""
Synthetic Receipt Dataset Generator
=====================================
Generates realistic Indian receipt OCR text + structured JSON pairs
for fine-tuning Qwen2.5-3B on structured extraction.

Uses OpenAI API (GPT-4o-mini) for generation — cheap (~$0.15/1M input tokens).
Output: synthetic_receipts.jsonl (raw), then merged into MLX-ready train/valid/test splits.

Usage:
    pip install openai tqdm
    export OPENAI_API_KEY=your_key
    python generate_synthetic.py
"""

import os
import json
import random
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import config

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CATEGORIES = [
    "grocery supermarket",
    "pharmacy / medical store",
    "restaurant / fast food",
    "electronics store",
    "clothing / apparel",
    "petrol pump / fuel station",
    "hardware store",
    "stationery / bookshop",
    "bakery / sweet shop",
    "mobile accessories shop",
]

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Indore", "Surat", "Kochi", "Nagpur",
    "Noida", "Gurgaon", "Bhopal", "Coimbatore", "Vizag",
]

# OCR noise patterns to inject — makes it realistic
OCR_NOISE_INSTRUCTIONS = """
Inject realistic OCR noise:
- Occasional character substitutions: 0→O, 1→I, rn→m, cl→d
- Random missing punctuation or extra spaces
- Address abbreviations: Rd, St, Nagar, Colony, Sector
- Amounts sometimes written as: Rs.245.00 or INR 245 or 245/-
- Dates in mixed formats: 12/03/2024 or 12-Mar-24 or 12.03.24
- Company names sometimes ALL CAPS or Title Case inconsistently
- Phone numbers, GSTIN sometimes present, sometimes not
- Line items present but you don't need to extract them
"""

SYSTEM_PROMPT = """You are a synthetic data generator for receipt OCR extraction.

Your job: generate a realistic Indian receipt as raw OCR text, then provide the ground truth extraction.

Rules:
1. Output ONLY valid JSON — no markdown, no backticks, no explanation
2. JSON must have exactly two keys: "ocr_text" and "extraction"
3. "extraction" must have exactly these keys: "company", "date", "address", "total"
4. "total" must be a numeric string like "245.00" — no currency symbols
5. "date" must be in YYYY-MM-DD format in extraction (even if OCR text has mixed format)
6. If a field is genuinely unclear in the OCR, use empty string "" — never null
7. ocr_text should look like raw scanner output: messy, multi-line, realistic
""" + OCR_NOISE_INSTRUCTIONS


def build_user_prompt(category: str, city: str, difficulty: str) -> str:
    difficulty_map = {
        "easy": "Clean receipt, minimal noise, all fields clearly present.",
        "medium": "Moderate noise. Date format ambiguous. Address partially cut off.",
        "hard": "Heavy OCR noise. Company name garbled. Total buried in line items. Address incomplete.",
    }
    return (
        f"Generate a receipt from a {category} in {city}, India.\n"
        f"Difficulty: {difficulty_map[difficulty]}\n"
        f"Make it realistic for India — include GSTIN, local store names, Indian address format."
    )


def generate_one(client: OpenAI, category: str, city: str, difficulty: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(category, city, difficulty)},
                ],
                temperature=0.9,      # High temp = more variety
                max_tokens=800,
                response_format={"type": "json_object"},  # Force JSON mode
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            # Validate structure
            assert "ocr_text" in parsed, "Missing ocr_text"
            assert "extraction" in parsed, "Missing extraction"
            for field in config.FIELDS:
                assert field in parsed["extraction"], f"Missing field: {field}"

            # Attach metadata (useful for analysis later)
            parsed["_meta"] = {
                "category": category,
                "city": city,
                "difficulty": difficulty,
                "source": "synthetic_gpt4o_mini",
            }
            return parsed

        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            print(f"  [Attempt {attempt+1}] Parse/validation error: {e}. Retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"  [Attempt {attempt+1}] API error: {e}. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff

    return None  # Failed after retries


def to_mlx_chat_format(example: dict) -> dict:
    """
    Convert raw generated example to MLX-LM chat JSONL format.
    This is what gets fed to mlx_lm.lora for training.
    """
    ocr_text = example["ocr_text"]
    extraction = {k: v for k, v in example["extraction"].items()}  # clean copy

    return {
        "messages": [
            {
                "role": "system",
                "content": config.SYSTEM_PROMPT,
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


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples → {path}")


def split_and_save(examples: list[dict], output_dir: Path, train_ratio=0.8, valid_ratio=0.1):
    """Split into train/valid/test and save as MLX-ready JSONL files."""
    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    train = examples[:train_end]
    valid = examples[train_end:valid_end]
    test = examples[valid_end:]

    mlx_dir = output_dir / "mlx_format"
    save_jsonl([to_mlx_chat_format(e) for e in train], mlx_dir / "train.jsonl")
    save_jsonl([to_mlx_chat_format(e) for e in valid], mlx_dir / "valid.jsonl")
    save_jsonl([to_mlx_chat_format(e) for e in test],  mlx_dir / "test.jsonl")

    # Also save raw (useful for debugging and error analysis)
    raw_dir = output_dir / "raw"
    save_jsonl(train, raw_dir / "train_raw.jsonl")
    save_jsonl(valid, raw_dir / "valid_raw.jsonl")
    save_jsonl(test,  raw_dir / "test_raw.jsonl")

    print(f"\nSplit summary: train={len(train)} | valid={len(valid)} | test={len(test)}")
    return test  # Return test set — used for eval later


def estimate_cost(count: int) -> float:
    # GPT-4o-mini: ~$0.15/1M input, ~$0.60/1M output
    # Rough estimate: ~300 input tokens + ~400 output tokens per example
    input_cost = (count * 300 / 1_000_000) * 0.15
    output_cost = (count * 400 / 1_000_000) * 0.60
    return input_cost + output_cost


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic receipt dataset")
    parser.add_argument("--count", type=int, default=config.SYNTHETIC_COUNT, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default=config.SYNTHETIC_DIR, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without generating")
    args = parser.parse_args()

    est = estimate_cost(args.count)
    print(f"\nEstimated cost for {args.count} examples: ~${est:.3f} USD (~₹{est*84:.0f} INR)")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable first.")

    client = OpenAI(api_key=api_key)
    output_dir = Path(args.output)

    # Build generation plan — balanced across categories and difficulties
    difficulties = ["easy", "medium", "hard"]
    difficulty_weights = [0.3, 0.4, 0.3]   # More medium = more realistic distribution

    plan = []
    for _ in range(args.count):
        plan.append({
            "category": random.choice(CATEGORIES),
            "city": random.choice(CITIES),
            "difficulty": random.choices(difficulties, weights=difficulty_weights)[0],
        })

    # Generate
    examples = []
    failed = 0
    raw_output_path = output_dir / "raw" / "all_generated.jsonl"
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.count} examples...\n")
    with open(raw_output_path, "w", encoding="utf-8") as f_raw:
        for item in tqdm(plan, desc="Generating"):
            result = generate_one(client, item["category"], item["city"], item["difficulty"])
            if result:
                examples.append(result)
                f_raw.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_raw.flush()  # Write incrementally — safe if interrupted
            else:
                failed += 1

    print(f"\nGeneration complete: {len(examples)} succeeded, {failed} failed")

    if len(examples) < 50:
        print("Too few examples generated. Check your API key and retry.")
        return

    # Split and save
    print("\nSplitting and saving...")
    test_set = split_and_save(examples, output_dir)

    # Print difficulty distribution of final dataset
    from collections import Counter
    diff_counts = Counter(e["_meta"]["difficulty"] for e in examples)
    cat_counts = Counter(e["_meta"]["category"] for e in examples)
    print(f"\nDifficulty distribution: {dict(diff_counts)}")
    print(f"Top 5 categories: {dict(cat_counts.most_common(5))}")

    print(f"\nDone. MLX-ready files in: {output_dir}/mlx_format/")
    print(f"Next step: run baseline eval on test set before fine-tuning.")
    print(f"  → python baseline_eval.py --test-data {output_dir}/mlx_format/test.jsonl")


if __name__ == "__main__":
    main()
