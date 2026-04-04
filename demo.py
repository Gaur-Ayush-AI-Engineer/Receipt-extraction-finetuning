"""
demo.py — Receipt field extraction demo
Usage:
    python demo.py --example 1        # run a hardcoded example
    python demo.py --text "raw text"  # extract from CLI-provided text
    python demo.py                    # interactive: paste text, end with Ctrl+D
"""

import argparse
import json
import sys

from mlx_lm import load, generate
import config

# ── Hardcoded examples ────────────────────────────────────────────────────────
EXAMPLES = {
    1: """\
RELIANCE FRESH
12, MG Road, Bengaluru - 560001
GSTIN: 29AABCR1234A1Z5
Date: 15/03/2024
---------------------------------
Milk 2L          Rs.  98.00
Bread             Rs.  45.00
Eggs (12)         Rs. 120.00
---------------------------------
TOTAL             Rs. 263.00
Thank you for shopping with us!
""",
    2: """\
D-MART   SUPERSTORE
Andheri (W), Mumbai 400053
GST No. 27AAECS4891Q1ZX
Bil1 Date: 7-Jan-2O24   Cashier: R0HAN
..................................
Rice 5kg               449.OO
C00king 0il 1L          135.OO
Dett0l S0ap x3           87.OO
..................................
Sub T0tal              671.OO
CGST @9%                60.39
SGST @9%                60.39
..................................
GRAND T0TAL            791.78
""",
    3: """\
*** SHREE GANESH KIRANA & GENERAL STORE ***
opp. bus stand, near water tank,
nagar road, pune 411014 MH
mob: 9822XXXXXX
dt.22/1 1/23  inv#1847
qty  item               amt
2    aata 10kg         @580  1160
3    dal chana          @95   285
1    sarso oil 2ltr    @210   210
1    namak              @22    22
misc                          45
                      ---------
total                      1722
cash paid                  1800
balance                      78
no exchange no return
""",
}


def build_prompt(tokenizer: object, receipt_text: str) -> str:
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract structured fields from this receipt:\n\n{receipt_text}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_and_print(model, tokenizer, receipt_text: str) -> None:
    prompt = build_prompt(tokenizer, receipt_text)
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)

    print("\n--- Receipt Text ---")
    print(receipt_text.strip())
    print("\n--- Extracted Fields ---")

    try:
        # Strip markdown fences if present
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        for field in config.FIELDS:
            print(f"  {field:<10}: {data.get(field, '')}")
        print("\n--- Full JSON ---")
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        print("  [WARNING] Could not parse model output as JSON. Raw output:")
        print(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Receipt field extraction demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="Receipt text to extract from")
    group.add_argument("--example", type=int, choices=[1, 2, 3],
                       help="Run a hardcoded example (1=easy, 2=medium noise, 3=hard/messy)")
    args = parser.parse_args()

    print(f"Loading {config.MODEL} with adapters from {config.ADAPTERS_DIR}...")
    model, tokenizer = load(config.MODEL, adapter_path=config.ADAPTERS_DIR)
    print("Model loaded.\n")

    if args.example:
        receipt_text = EXAMPLES[args.example]
    elif args.text:
        receipt_text = args.text
    else:
        print("Paste receipt text below. Press Ctrl+D (or Ctrl+Z on Windows) when done:")
        try:
            receipt_text = sys.stdin.read()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    extract_and_print(model, tokenizer, receipt_text)


if __name__ == "__main__":
    main()
