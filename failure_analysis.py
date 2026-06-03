"""
failure_analysis.py — Categorize model failures from finetuned_vast_results.json

Outputs:
  - Console summary table
  - results/failure_analysis.md
"""

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_FILE = "./results/finetuned_vast_results.json"
OUTPUT_FILE  = "./results/failure_analysis.md"

# Heuristics to detect receipt source from OCR text
INDIAN_SIGNALS  = ["GSTIN", "INR", "Rs.", "rs.", "₹", "CGST", "SGST", "Ahmedabad",
                   "Mumbai", "Delhi", "Bengaluru", "Pune", "Bhopal", "Gujarat",
                   "Maharashtra", "Rajasthan", "Karnataka", "Madhya Pradesh"]
MALAYSIAN_SIGNALS = ["SDN BHD", "BHD", "JOHOR", "KUALA LUMPUR", "SELANGOR",
                     "PENANG", "SABAH", "SARAWAK", "RM", "MYR", "GST REG #"]


def detect_source(ocr_text: str) -> str:
    text_upper = ocr_text.upper()
    indian_hits   = sum(1 for s in INDIAN_SIGNALS   if s.upper() in text_upper)
    malaysian_hits = sum(1 for s in MALAYSIAN_SIGNALS if s.upper() in text_upper)
    if indian_hits > malaysian_hits:
        return "synthetic"
    if malaysian_hits > indian_hits:
        return "sroie"
    return "unknown"


def classify_field_error(gt: str, pred: str) -> str:
    if pred == "" or pred is None:
        return "missing"
    if gt == "" or gt is None:
        return "hallucinated"
    gt_norm   = re.sub(r"\s+", " ", str(gt).strip().lower())
    pred_norm = re.sub(r"\s+", " ", str(pred).strip().lower())
    if pred_norm in gt_norm or gt_norm in pred_norm:
        return "partial"
    return "wrong_value"


def main():
    data = json.loads(Path(RESULTS_FILE).read_text())
    results = data["detailed_results"]

    fields = ["company", "date", "address", "total"]

    # Aggregate counts
    field_failures   = defaultdict(int)          # field → count of failures
    source_failures  = defaultdict(int)          # source → count of failed examples
    source_total     = defaultdict(int)          # source → total examples
    error_types      = defaultdict(lambda: defaultdict(int))  # field → error_type → count
    multi_field_fail = defaultdict(int)          # n_fields_failed → count

    failure_examples = []

    for r in results:
        gt   = r["ground_truth"]
        pred = r["prediction"] or {}
        src  = detect_source(r["ocr_text"])
        source_total[src] += 1

        failed_fields = []
        for f in fields:
            if r["scores"].get(f"{'address_exact' if f == 'address' else f}", 0) != 1:
                failed_fields.append(f)
                field_failures[f] += 1
                err = classify_field_error(gt.get(f, ""), pred.get(f, ""))
                error_types[f][err] += 1

        if failed_fields:
            source_failures[src] += 1
            multi_field_fail[len(failed_fields)] += 1
            failure_examples.append({
                "source": src,
                "failed_fields": failed_fields,
                "gt": gt,
                "pred": pred,
                "ocr_snippet": r["ocr_text"][:120].replace("\n", " "),
            })

    total     = len(results)
    n_failed  = len(failure_examples)
    n_correct = total - n_failed

    # ── Console output ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Failure Analysis  ({n_failed}/{total} examples failed)")
    print(f"{'='*55}")

    print("\nFailures by field:")
    for f in fields:
        pct = field_failures[f] / total * 100
        print(f"  {f:<12} {field_failures[f]:>3} failures  ({pct:.1f}%)")

    print("\nFailures by source:")
    for src in ["sroie", "synthetic", "unknown"]:
        if source_total[src]:
            fail_pct = source_failures[src] / source_total[src] * 100
            print(f"  {src:<12} {source_failures[src]:>3}/{source_total[src]} failed  ({fail_pct:.1f}%)")

    print("\nError type breakdown per field:")
    for f in fields:
        if error_types[f]:
            parts = ", ".join(f"{k}: {v}" for k, v in sorted(error_types[f].items()))
            print(f"  {f:<12} {parts}")

    print("\nHow many fields fail together:")
    for n in sorted(multi_field_fail):
        label = "field" if n == 1 else "fields"
        print(f"  {n} {label} failed: {multi_field_fail[n]} examples")

    # Top-5 address partial failures (address is the weakest field)
    addr_fails = [e for e in failure_examples if "address" in e["failed_fields"]]
    if addr_fails:
        print(f"\nAddress failure samples (first 5 of {len(addr_fails)}):")
        for e in addr_fails[:5]:
            print(f"  GT:   {e['gt'].get('address','')}")
            print(f"  Pred: {e['pred'].get('address','')}")
            print()

    # ── Markdown report ───────────────────────────────────────────────────────
    lines = [
        "# Failure Analysis\n",
        f"**Total examples:** {total}  ",
        f"**Correct (all fields):** {n_correct} ({n_correct/total*100:.1f}%)  ",
        f"**Failed (≥1 field wrong):** {n_failed} ({n_failed/total*100:.1f}%)\n",

        "## Failures by Field\n",
        "| Field | Failures | Error Rate |",
        "|:------|--------:|-----------:|",
    ]
    for f in fields:
        pct = field_failures[f] / total * 100
        lines.append(f"| {f} | {field_failures[f]} | {pct:.1f}% |")

    lines += [
        "\n## Failures by Receipt Source\n",
        "| Source | Failed | Total | Failure Rate |",
        "|:-------|-------:|------:|-------------:|",
    ]
    for src in ["sroie", "synthetic", "unknown"]:
        if source_total[src]:
            pct = source_failures[src] / source_total[src] * 100
            lines.append(f"| {src} | {source_failures[src]} | {source_total[src]} | {pct:.1f}% |")

    lines += [
        "\n## Error Type Breakdown\n",
        "| Field | Missing | Partial | Wrong Value | Hallucinated |",
        "|:------|--------:|--------:|------------:|-------------:|",
    ]
    for f in fields:
        et = error_types[f]
        lines.append(
            f"| {f} | {et.get('missing',0)} | {et.get('partial',0)} | "
            f"{et.get('wrong_value',0)} | {et.get('hallucinated',0)} |"
        )

    lines += [
        "\n## Co-occurrence: How Many Fields Fail Together\n",
        "| Fields Failed | Examples |",
        "|:-------------:|---------:|",
    ]
    for n in sorted(multi_field_fail):
        lines.append(f"| {n} | {multi_field_fail[n]} |")

    addr_section = ["\n## Address Failure Samples (first 10)\n"]
    for e in addr_fails[:10]:
        addr_section += [
            f"**Source:** {e['source']}  ",
            f"**GT:** `{e['gt'].get('address','')}`  ",
            f"**Pred:** `{e['pred'].get('address','')}`  ",
            f"**OCR snippet:** {e['ocr_snippet']}",
            "",
        ]
    lines += addr_section

    Path(OUTPUT_FILE).write_text("\n".join(lines))
    print(f"\nReport saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
