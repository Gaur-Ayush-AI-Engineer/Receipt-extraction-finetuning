"""
Results Comparison Script
==========================
Loads baseline_results.json and finetuned_results.json (produced by
baseline_eval.py) and generates a clean before/after comparison.

Outputs:
  - Formatted comparison table printed to terminal
  - Markdown table saved to ./results/comparison.md (for GitHub README)
  - Regression warnings for any field that got worse
  - Error analysis categorizing remaining failures

Usage:
    python compare_results.py
    python compare_results.py \\
        --baseline  ./results/baseline_results.json \\
        --finetuned ./results/finetuned_results.json \\
        --output    ./results/comparison.md
"""

import json
import argparse
from pathlib import Path
import config


# ── Field display config ──────────────────────────────────────────────────────

# Maps internal key → display label
FIELD_DISPLAY: dict[str, str] = {
    "company":           "Company",
    "date":              "Date",
    "address_exact":     "Address (exact)",
    "address_fuzzy_avg": "Address (fuzzy)",
    "total":             "Total",
}

# Fields where the score is already 0-1 (exact match / normalized fuzzy)
# These are the keys present in summary["field_exact_match"]
METRIC_FIELDS = list(FIELD_DISPLAY.keys())


# ── Loading ───────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    """
    Load an eval results JSON file produced by baseline_eval.py.

    Raises FileNotFoundError with a helpful message if missing.
    Raises ValueError if the expected keys are absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            f"Run baseline_eval.py with the appropriate arguments first."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate structure
    if "summary" not in data:
        raise ValueError(f"'summary' key missing in {path}. Is this a valid eval results file?")
    if "field_exact_match" not in data["summary"]:
        raise ValueError(f"'field_exact_match' missing in {path}['summary'].")

    return data


# ── Comparison logic ──────────────────────────────────────────────────────────

def compute_deltas(
    baseline: dict,
    finetuned: dict,
) -> list[dict]:
    """
    Compute per-field comparison metrics.

    Returns a list of dicts with keys:
        field, label, before, after, delta, pct_improvement, regression
    """
    base_scores = baseline["summary"]["field_exact_match"]
    ft_scores = finetuned["summary"]["field_exact_match"]

    rows = []
    for field in METRIC_FIELDS:
        if field not in base_scores:
            print(f"  [WARN] Field '{field}' not found in baseline results — skipping.")
            continue
        if field not in ft_scores:
            print(f"  [WARN] Field '{field}' not found in finetuned results — skipping.")
            continue

        before = base_scores[field]
        after = ft_scores[field]
        delta = after - before
        # % improvement relative to baseline (avoid div-by-zero)
        if before > 0:
            pct_improvement = (delta / before) * 100
        elif after > 0:
            pct_improvement = float("inf")
        else:
            pct_improvement = 0.0

        rows.append({
            "field": field,
            "label": FIELD_DISPLAY[field],
            "before": before,
            "after": after,
            "delta": delta,
            "pct_improvement": pct_improvement,
            "regression": delta < -0.001,  # Tolerance for float noise
        })

    return rows


def compute_overall(baseline: dict, finetuned: dict) -> dict:
    """Extract overall scores and parse failure rates for comparison."""
    def safe_get(data: dict, *keys, default=0.0):
        val = data
        for k in keys:
            if not isinstance(val, dict) or k not in val:
                return default
            val = val[k]
        return val

    return {
        "before_overall":       safe_get(baseline,  "summary", "all_fields_correct"),
        "after_overall":        safe_get(finetuned, "summary", "all_fields_correct"),
        "before_parse_fail":    safe_get(baseline,  "summary", "parse_fail_rate"),
        "after_parse_fail":     safe_get(finetuned, "summary", "parse_fail_rate"),
        "before_n":             safe_get(baseline,  "summary", "n_examples"),
        "after_n":              safe_get(finetuned, "summary", "n_examples"),
    }


# ── Error analysis ────────────────────────────────────────────────────────────

def categorize_failure(result: dict) -> str:
    """
    Categorize a single failed prediction into one of four types:
      - parse_failure   : model output wasn't valid JSON
      - wrong_format    : JSON parsed but missing expected keys
      - partial_match   : some fields correct, not all
      - wrong_value     : all keys present, all values wrong
    """
    if result.get("parse_failed", False):
        return "parse_failure"

    pred = result.get("prediction", {})
    gt = result.get("ground_truth", {})
    scores = result.get("scores", {})

    # Check if all expected keys exist in prediction
    expected_keys = {"company", "date", "address", "total"}
    if not expected_keys.issubset(set(pred.keys())):
        return "wrong_format"

    # Count exact matches (use binary score keys; address uses address_exact)
    binary_score_keys = ["company", "date", "total", "address_exact"]
    correct = sum(1 for k in binary_score_keys if scores.get(k, 0) == 1)

    if correct > 0:
        return "partial_match"
    return "wrong_value"


def run_error_analysis(finetuned_data: dict) -> dict[str, int]:
    """
    Analyze remaining failures in the finetuned model's detailed results.

    Returns a dict mapping failure category → count.
    """
    detailed = finetuned_data.get("detailed_results", [])
    if not detailed:
        print("  [WARN] No detailed_results found in finetuned results — skipping error analysis.")
        return {}

    # A result is a "failure" if any binary field is wrong
    binary_score_keys = ["company", "date", "total", "address_exact"]
    failures = [
        r for r in detailed
        if r.get("parse_failed") or any(r.get("scores", {}).get(k, 0) != 1 for k in binary_score_keys)
    ]

    categories: dict[str, int] = {
        "parse_failure": 0,
        "wrong_format": 0,
        "partial_match": 0,
        "wrong_value": 0,
    }
    for r in failures:
        cat = categorize_failure(r)
        categories[cat] = categories.get(cat, 0) + 1

    return categories


# ── Terminal output ───────────────────────────────────────────────────────────

def print_comparison(rows: list[dict], overall: dict, error_categories: dict[str, int]) -> None:
    """Print a clean comparison table and summary to terminal."""
    W = 65
    print(f"\n{'═'*W}")
    print(f"  Before / After Fine-tuning Comparison")
    print(f"{'═'*W}")

    # Header
    print(f"  {'Field':<22} {'Before':>7} {'After':>7} {'Delta':>7} {'% Impr':>8}  Flag")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*8}  ────")

    has_regression = False
    for row in rows:
        flag = ""
        if row["regression"]:
            flag = "WORSE"
            has_regression = True
        elif row["delta"] > 0.001:
            flag = "+"

        pct_str = (
            f"{row['pct_improvement']:+.1f}%"
            if row["pct_improvement"] != float("inf")
            else "  new"
        )

        print(
            f"  {row['label']:<22} "
            f"{row['before']*100:>6.1f}% "
            f"{row['after']*100:>6.1f}% "
            f"{row['delta']*100:>+6.1f}% "
            f"{pct_str:>8}  {flag}"
        )

    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*8}  ────")

    # Overall
    delta_overall = overall["after_overall"] - overall["before_overall"]
    print(
        f"  {'All fields correct':<22} "
        f"{overall['before_overall']*100:>6.1f}% "
        f"{overall['after_overall']*100:>6.1f}% "
        f"{delta_overall*100:>+6.1f}%"
    )

    # Parse failures
    delta_parse = overall["after_parse_fail"] - overall["before_parse_fail"]
    parse_improved = delta_parse < 0
    print(
        f"  {'JSON parse failures':<22} "
        f"{overall['before_parse_fail']*100:>6.1f}% "
        f"{overall['after_parse_fail']*100:>6.1f}% "
        f"{delta_parse*100:>+6.1f}%  "
        f"{'(improved)' if parse_improved else '(got worse)' if delta_parse > 0.001 else ''}"
    )

    print(f"{'═'*W}")

    # Regression warnings
    if has_regression:
        print(f"\n  ⚠  REGRESSION DETECTED:")
        for row in rows:
            if row["regression"]:
                print(
                    f"     {row['label']}: {row['before']*100:.1f}% → {row['after']*100:.1f}% "
                    f"({row['delta']*100:+.1f}%)"
                )
        print(f"     Investigate these fields — fine-tuning hurt performance.")

    # Error analysis
    if error_categories:
        total_failures = sum(error_categories.values())
        print(f"\n  Error Analysis — Remaining Failures (finetuned model, n={total_failures}):")
        for cat, count in sorted(error_categories.items(), key=lambda x: -x[1]):
            pct = count / total_failures * 100 if total_failures else 0
            bar = "█" * int(pct / 5)
            print(f"    {cat:<20} {count:>4}  ({pct:5.1f}%)  {bar}")

    print()


# ── Markdown output ───────────────────────────────────────────────────────────

def build_markdown(
    rows: list[dict],
    overall: dict,
    error_categories: dict[str, int],
    baseline_label: str,
    finetuned_label: str,
) -> str:
    """
    Build a markdown table suitable for pasting directly into a GitHub README.
    """
    lines: list[str] = []

    lines.append("## Fine-tuning Results\n")
    lines.append(f"**Baseline**: {baseline_label}  ")
    lines.append(f"**Fine-tuned**: {finetuned_label}\n")

    # Per-field table
    lines.append("### Per-field Exact Match\n")
    lines.append("| Field | Before | After | Delta | % Improvement |")
    lines.append("|:------|-------:|------:|------:|--------------:|")

    for row in rows:
        flag = " ⚠️ REGRESSION" if row["regression"] else ""
        pct_str = (
            f"{row['pct_improvement']:+.1f}%"
            if row["pct_improvement"] != float("inf")
            else "N/A (was 0%)"
        )
        lines.append(
            f"| {row['label']}{flag} "
            f"| {row['before']*100:.1f}% "
            f"| {row['after']*100:.1f}% "
            f"| {row['delta']*100:+.1f}% "
            f"| {pct_str} |"
        )

    # Overall row
    delta_overall = overall["after_overall"] - overall["before_overall"]
    lines.append(
        f"| **All fields correct** "
        f"| **{overall['before_overall']*100:.1f}%** "
        f"| **{overall['after_overall']*100:.1f}%** "
        f"| **{delta_overall*100:+.1f}%** "
        f"| — |"
    )

    # Parse failure row
    delta_parse = overall["after_parse_fail"] - overall["before_parse_fail"]
    lines.append(
        f"| JSON parse failure rate "
        f"| {overall['before_parse_fail']*100:.1f}% "
        f"| {overall['after_parse_fail']*100:.1f}% "
        f"| {delta_parse*100:+.1f}% "
        f"| — |"
    )

    lines.append("")

    # Error analysis table
    if error_categories:
        total_failures = sum(error_categories.values())
        lines.append("### Remaining Failure Analysis (fine-tuned model)\n")
        lines.append("| Failure Type | Count | % of Failures |")
        lines.append("|:-------------|------:|--------------:|")
        for cat, count in sorted(error_categories.items(), key=lambda x: -x[1]):
            pct = count / total_failures * 100 if total_failures else 0
            lines.append(f"| {cat.replace('_', ' ').title()} | {count} | {pct:.1f}% |")
        lines.append("")

    # Regression callout
    regressions = [r for r in rows if r["regression"]]
    if regressions:
        lines.append("> **⚠️ Regression detected** in the following fields after fine-tuning:")
        for r in regressions:
            lines.append(f"> - **{r['label']}**: {r['before']*100:.1f}% → {r['after']*100:.1f}% ({r['delta']*100:+.1f}%)")
        lines.append("")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and fine-tuned evaluation results."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=config.BASELINE_RESULTS,
        help="Path to baseline eval results JSON.",
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        default=config.FINETUNED_RESULTS,
        help="Path to fine-tuned eval results JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.COMPARISON_MD,
        help="Output path for the markdown comparison table.",
    )
    args = parser.parse_args()

    baseline_path  = Path(args.baseline)
    finetuned_path = Path(args.finetuned)
    output_path    = Path(args.output)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading baseline  : {baseline_path}")
    baseline = load_results(baseline_path)

    print(f"Loading finetuned : {finetuned_path}")
    finetuned = load_results(finetuned_path)

    baseline_label  = baseline.get("label", "Baseline")
    finetuned_label = finetuned.get("label", "Fine-tuned")

    print(f"  Baseline  label : {baseline_label}")
    print(f"  Finetuned label : {finetuned_label}")

    # ── Compute ───────────────────────────────────────────────────────────────
    rows = compute_deltas(baseline, finetuned)
    overall = compute_overall(baseline, finetuned)

    print(f"\nRunning error analysis on finetuned results...")
    error_categories = run_error_analysis(finetuned)

    # ── Terminal output ───────────────────────────────────────────────────────
    print_comparison(rows, overall, error_categories)

    # ── Markdown output ───────────────────────────────────────────────────────
    md = build_markdown(rows, overall, error_categories, baseline_label, finetuned_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown table saved → {output_path}")
    print("Paste the contents of that file into your GitHub README.")


if __name__ == "__main__":
    main()
