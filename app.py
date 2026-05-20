"""
app.py — Gradio web UI for receipt field extraction
Run: python app.py
"""

import json
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import config

# ── Example receipts ──────────────────────────────────────────────────────────
EXAMPLES = [
    ["RELIANCE FRESH\n12, MG Road, Bengaluru - 560001\nGSTIN: 29AABCR1234A1Z5\nDate: 15/03/2024\n---------------------------------\nMilk 2L          Rs.  98.00\nBread             Rs.  45.00\nEggs (12)         Rs. 120.00\n---------------------------------\nTOTAL             Rs. 263.00"],
    ["D-MART   SUPERSTORE\nAndheri (W), Mumbai 400053\nGST No. 27AAECS4891Q1ZX\nBil1 Date: 7-Jan-2O24\n..................................\nRice 5kg               449.OO\nC00king 0il 1L          135.OO\nDett0l S0ap x3           87.OO\n..................................\nGRAND T0TAL            791.78"],
    ["*** SHREE GANESH KIRANA ***\nopp. bus stand, nagar road, pune 411014\ndt.22/11/23  inv#1847\n2    aata 10kg         1160\n3    dal chana          285\n1    sarso oil 2ltr     210\n                    --------\ntotal                  1722\ncash paid              1800"],
]


def _load_model():
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        m = AutoModelForCausalLM.from_pretrained(
            config.MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
    elif torch.backends.mps.is_available():
        m = AutoModelForCausalLM.from_pretrained(
            config.MODEL, torch_dtype=torch.float16, device_map="mps", trust_remote_code=True
        )
    else:
        m = AutoModelForCausalLM.from_pretrained(
            config.MODEL, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
        )
    m = PeftModel.from_pretrained(m, config.ADAPTERS_DIR)
    m.eval()
    tok = AutoTokenizer.from_pretrained(config.MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return m, tok


print(f"Loading model {config.MODEL}...")
model, tokenizer = _load_model()
print("Model ready.")


def extract(receipt_text: str):
    if not receipt_text.strip():
        return "", "", "", "", ""

    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract structured fields from this receipt:\n\n{receipt_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        return (
            data.get("company", ""),
            data.get("date", ""),
            data.get("address", ""),
            data.get("total", ""),
            json.dumps(data, indent=2),
        )
    except json.JSONDecodeError:
        return "", "", "", "", f"⚠️ Could not parse JSON. Raw output:\n{raw}"


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Receipt Extractor") as demo:
    gr.Markdown("# Receipt Field Extractor\nPaste OCR text from any receipt. The model extracts structured fields.")

    with gr.Row():
        with gr.Column(scale=2):
            receipt_input = gr.Textbox(
                label="Receipt Text",
                placeholder="Paste receipt OCR text here...",
                lines=14,
            )
            extract_btn = gr.Button("Extract Fields", variant="primary")

        with gr.Column(scale=1):
            out_company = gr.Textbox(label="Company")
            out_date    = gr.Textbox(label="Date")
            out_address = gr.Textbox(label="Address")
            out_total   = gr.Textbox(label="Total")
            out_json    = gr.Code(label="Full JSON", language="json")

    gr.Examples(examples=EXAMPLES, inputs=receipt_input, label="Try an example")

    extract_btn.click(
        fn=extract,
        inputs=receipt_input,
        outputs=[out_company, out_date, out_address, out_total, out_json],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
