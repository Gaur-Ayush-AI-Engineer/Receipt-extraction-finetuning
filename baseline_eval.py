"""
Baseline Evaluation Script
===========================
Run this BEFORE fine-tuning to get your "before" numbers.
Run again AFTER fine-tuning (with --adapter-path) to get "after" numbers.

Metrics:
- Exact match per field (company, date, address, total)
- Fuzzy match for address (RapidFuzz — exact match too strict)
- Overall score

Usage:
    pip install mlx-lm rapidfuzz tqdm

    # Before fine-tuning (baseline):
    python baseline_eval.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --test-data ./data/synthetic/mlx_format/test.jsonl \
        --output ./results/baseline_results.json

    # After fine-tuning:
    python baseline_eval.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter-path ./adapters \
        --test-data ./data/synthetic/mlx_format/test.jsonl \
        --output ./results/finetuned_results.json
"""

import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from rapidfuzz import fuzz
from mlx_lm import load, generate
import config

SYSTEM_PROMPT = config.SYSTEM_PROMPT
FIELDS        = config.FIELDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_json_from_response(text: str) -> dict | None:
    """
    Parse JSON from model output — handles markdown fences, extra text, etc.
    Returns None if parsing fails.
    """
    # Strip markdown fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def normalize_total(val: str) -> str:
    """Normalize total: strip currency symbols, whitespace, trailing zeros."""
    val = re.sub(r"[₹$€£,\s]", "", val.strip())
    try:
        return f"{float(val):.2f}"
    except ValueError:
        return val.lower()


def normalize_date(val: str) -> str:
    """Basic date normalization — just strip whitespace."""
    return val.strip()


def score_example(pred: dict, gt: dict) -> dict:
    """
    Score a single prediction against ground truth.
    Returns per-field scores (0 or 1 for exact, 0-100 for fuzzy).
    """
    scores = {}

    for field in FIELDS:
        p = pred.get(field, "").strip()
        g = gt.get(field, "").strip()

        if field == "total":
            p, g = normalize_total(p), normalize_total(g)
            scores[field] = int(p == g)

        elif field == "address":
            # Fuzzy match — address extraction is inherently noisy
            ratio = fuzz.token_sort_ratio(p.lower(), g.lower())
            scores[field] = ratio  # 0-100
            scores[f"{field}_exact"] = int(p.lower() == g.lower())

        elif field == "date":
            p, g = normalize_date(p), normalize_date(g)
            scores[field] = int(p == g)

        else:
            # company: case-insensitive exact match
            scores[field] = int(p.lower() == g.lower())

    return scores


def build_prompt(tokenizer, ocr_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract structured fields from this receipt:\n\n{ocr_text}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, test_data: list[dict], max_tokens: int = 300) -> list[dict]:
    results = []

    for example in tqdm(test_data, desc="Evaluating"):
        messages = example["messages"]

        # Extract ocr_text from user message
        user_msg = next(m for m in messages if m["role"] == "user")
        ocr_text = user_msg["content"].replace("Extract structured fields from this receipt:\n\n", "", 1)

        # Extract ground truth from assistant message
        assistant_msg = next(m for m in messages if m["role"] == "assistant")
        try:
            ground_truth = json.loads(assistant_msg["content"])
        except json.JSONDecodeError:
            continue

        # Build prompt and generate
        prompt = build_prompt(tokenizer, ocr_text)
        raw_output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        # Parse prediction
        pred = extract_json_from_response(raw_output)
        parse_failed = pred is None
        if parse_failed:
            pred = {f: "" for f in FIELDS}

        # Score
        field_scores = score_example(pred, ground_truth)

        results.append({
            "ocr_text": ocr_text[:200] + "...",  # Truncate for storage
            "ground_truth": ground_truth,
            "prediction": pred,
            "scores": field_scores,
            "parse_failed": parse_failed,
            "raw_output": raw_output[:500],
        })

    return results


def compute_summary(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    parse_fail_rate = sum(r["parse_failed"] for r in results) / n

    # Exact match fields
    exact_fields = ["company", "date", "total"]
    exact_scores = {f: sum(r["scores"].get(f, 0) for r in results) / n for f in exact_fields}

    # Address: both exact and fuzzy
    address_exact = sum(r["scores"].get("address_exact", 0) for r in results) / n
    address_fuzzy = sum(r["scores"].get("address", 0) for r in results) / n

    # Overall exact match (all 4 fields correct)
    all_correct = sum(
        all(r["scores"].get(f, 0) == 1 for f in ["company", "date", "total", "address_exact"])
        for r in results
    ) / n

    return {
        "n_examples": n,
        "parse_fail_rate": round(parse_fail_rate, 4),
        "field_exact_match": {
            "company": round(exact_scores["company"], 4),
            "date": round(exact_scores["date"], 4),
            "address_exact": round(address_exact, 4),
            "address_fuzzy_avg": round(address_fuzzy / 100, 4),  # Normalize to 0-1
            "total": round(exact_scores["total"], 4),
        },
        "all_fields_correct": round(all_correct, 4),
    }


def print_summary(summary: dict, label: str = "Results"):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Examples evaluated : {summary['n_examples']}")
    print(f"  JSON parse failures: {summary['parse_fail_rate']*100:.1f}%")
    print(f"\n  Field-level Exact Match:")
    for field, score in summary["field_exact_match"].items():
        bar = "█" * int(score * 20)
        print(f"    {field:<20} {score*100:5.1f}%  {bar}")
    print(f"\n  All fields correct : {summary['all_fields_correct']*100:.1f}%")
    print(f"{'='*50}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction model on test set")
    parser.add_argument("--model", type=str, default=config.MODEL)
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapters (omit for baseline eval)")
    parser.add_argument("--test-data", type=str, default=config.FINAL_TEST,
                        help="Path to test.jsonl")
    parser.add_argument("--output", type=str, default=config.BASELINE_RESULTS)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit eval to N samples (for quick sanity check)")
    args = parser.parse_args()

    # Load model
    print(f"\nLoading model: {args.model}")
    if args.adapter_path:
        print(f"Loading adapters: {args.adapter_path}")
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    # Load test data
    test_data = []
    with open(args.test_data, "r") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"Using {args.max_samples} samples (limited by --max-samples)")

    print(f"Test set size: {len(test_data)}")

    # Run eval
    results = run_eval(model, tokenizer, test_data)
    summary = compute_summary(results)

    label = "Fine-tuned Model" if args.adapter_path else "Baseline (No Fine-tuning)"
    print_summary(summary, label=label)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "label": label,
            "model": args.model,
            "adapter_path": args.adapter_path,
            "detailed_results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"Full results saved → {output_path}")

    # Error analysis — show 3 failure cases
    failures = [r for r in results if r["scores"].get("total", 0) == 0 or r["parse_failed"]]
    if failures:
        print(f"\n--- Sample Failures (showing 3 of {len(failures)}) ---")
        for r in failures[:3]:
            print(f"\n  OCR (truncated): {r['ocr_text'][:100]}...")
            print(f"  Ground truth   : {r['ground_truth']}")
            print(f"  Prediction     : {r['prediction']}")
            print(f"  Scores         : {r['scores']}")


if __name__ == "__main__":
    main()
