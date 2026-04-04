# Receipt Field Extraction — LoRA Fine-tuning on Apple Silicon

Fine-tuning **Qwen2.5-3B-Instruct** with LoRA via MLX-LM to extract structured fields from receipt OCR text. Trained entirely on an M3 Pro 32GB Mac — no cloud GPUs.

**Task:** Given raw OCR text from a receipt, extract:
```json
{"company": "...", "date": "YYYY-MM-DD", "address": "...", "total": "0.00"}
```

---

## Results

| Field | Baseline | Fine-tuned | Delta |
|:------|--------:|----------:|------:|
| Company | 75.1% | 90.4% | +15.2% |
| Date | 53.3% | 98.0% | +44.7% |
| Address (exact) | 47.7% | 77.2% | +29.4% |
| Address (fuzzy) | 94.6% | 97.7% | +3.1% |
| Total | 89.3% | 98.0% | +8.6% |
| **All fields correct** | **32.5%** | **71.1%** | **+38.6%** |
| JSON parse failures | 0.5% | 0.0% | -0.5% |

All remaining failures after fine-tuning are partial matches — the model always outputs valid JSON and never hallucinates field names.

---

## Dataset

Two sources merged into a single dataset (~1950 examples):

| Source | Examples | Description |
|--------|----------|-------------|
| [rth/sroie-2019-v2](https://huggingface.co/datasets/rth/sroie-2019-v2) | 973 | Real Malaysian receipts (SROIE benchmark) |
| Synthetic (GPT-4o-mini) | ~970 | Indian receipts with OCR noise, 10 categories, 20 cities |

**Split:** 80% train / 10% valid / 10% test (stratified — test set guaranteed to have examples from both sources and all difficulty levels)

---

## Setup

```bash
git clone <repo>
cd Fine-tuning
pip install -r requirements.txt
```

Requires **Apple Silicon Mac** (M1/M2/M3) — MLX only runs on Apple Silicon.

Add your OpenAI key to a `.env` file:
```
OPENAI_API_KEY=sk-...
```

---

## Training Pipeline

```bash
# 1. Generate synthetic Indian receipt data
python generate_synthetic.py

# 2. Download and process SROIE dataset
python prepare_sroie.py

# 3. Merge datasets and create train/valid/test splits
python merge_datasets.py

# 4. Fine-tune with LoRA
mlx_lm.lora --config lora_config.yaml

# 5. Evaluate baseline (before fine-tuning)
python baseline_eval.py

# 6. Evaluate fine-tuned model
python baseline_eval.py \
    --adapter-path ./adapters \
    --output ./results/finetuned_results.json

# 7. Compare results
python compare_results.py
```

All paths and hyperparameters are configured in `config.py` — edit that file, not the scripts.

---

## Model & Training Config

| Parameter | Value | Reason |
|-----------|-------|--------|
| Base model | Qwen2.5-3B-Instruct | Strong instruction following, fits in 32GB |
| Method | LoRA | Efficient fine-tuning, ~6.6M trainable params (0.2%) |
| Rank | 8 | Sufficient for constrained extraction task, less overfitting risk |
| Alpha | 16 | Standard 2× rank scaling |
| LoRA layers | 16 of 36 | Covers output-side layers responsible for formatting |
| Batch size | 8 | Fits in ~16GB peak memory |
| Iterations | 1000 | ~10 epochs over training set |
| Learning rate | 1e-4 with cosine decay | Standard LoRA LR, warmup for 100 steps |
| Max seq length | 1024 | Receipt OCR + response fits comfortably |

---

## Project Structure

```
.
├── config.py                  # All variables — edit here
├── generate_synthetic.py      # GPT-4o-mini synthetic data generation
├── prepare_sroie.py           # SROIE dataset processing
├── merge_datasets.py          # Dataset merging + stratified split
├── baseline_eval.py           # Before/after evaluation
├── compare_results.py         # Results comparison + markdown output
├── lora_config.yaml           # MLX-LM LoRA training config
├── requirements.txt
├── data/
│   ├── synthetic/             # Generated Indian receipt data
│   ├── sroie/                 # Processed SROIE data
│   └── final/mlx_format/      # train.jsonl, valid.jsonl, test.jsonl
├── adapters/                  # Saved LoRA adapter weights
└── results/
    ├── baseline_results.json
    ├── finetuned_results.json
    └── comparison.md
```

---

## Inference

```python
from mlx_lm import load, generate

model, tokenizer = load("Qwen/Qwen2.5-3B-Instruct", adapter_path="./adapters")

messages = [
    {"role": "system", "content": "You are a structured data extraction assistant. Given raw OCR text from a receipt, extract the fields as a JSON object with keys: company, date, address, total. date must be YYYY-MM-DD. total must be numeric string only. If a field is missing, use empty string."},
    {"role": "user", "content": "Extract structured fields from this receipt:\n\nSUPERMART\n123 MG Road, Bengaluru\nDate: 12/03/2024\nTotal: Rs.847.00"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
# {"company": "SUPERMART", "date": "2024-03-12", "address": "123 MG Road, Bengaluru", "total": "847.00"}
```

Or fuse the adapters permanently:
```bash
mlx_lm.fuse \
    --model Qwen/Qwen2.5-3B-Instruct \
    --adapter-path ./adapters \
    --save-path ./fused_model
```
