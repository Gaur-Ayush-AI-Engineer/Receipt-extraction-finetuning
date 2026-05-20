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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
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


def load_model(model_name: str, adapter_path: str):
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
    elif torch.backends.mps.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="mps", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
        )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_prompt(tokenizer: object, receipt_text: str) -> str:
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract structured fields from this receipt:\n\n{receipt_text}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_and_print(model, tokenizer, receipt_text: str) -> None:
    prompt = build_prompt(tokenizer, receipt_text)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

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

    print(f"Loading {config.MODEL} with PEFT adapters from {config.ADAPTERS_DIR}...")
    model, tokenizer = load_model(config.MODEL, adapter_path=config.ADAPTERS_DIR)
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
