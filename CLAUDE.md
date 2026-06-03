# CLAUDE.md — Fine-tuning Project Context

## What was built

End-to-end QLoRA fine-tuning pipeline for structured receipt field extraction (company, date, address, total) using Qwen2.5-3B-Instruct. Trained on NVIDIA RTX 3090 via Vast.ai using HuggingFace PEFT + TRL SFTTrainer.

**Files and what they do:**
- `config.py` — single source of truth for all variables (model, paths, ratios, system prompt, synthetic count). All scripts import from here.
- `generate_synthetic.py` — generates 1000 Indian receipt OCR + extraction pairs via GPT-4o-mini
- `prepare_sroie.py` — downloads `rth/sroie-2019-v2` from HuggingFace, extracts fields from nested `objects.entities`, drops the image column before iteration
- `merge_datasets.py` — merges SROIE + synthetic, stratified 80/10/10 split (test set guaranteed to have both sources + all difficulty levels)
- `train_peft.py` — QLoRA training script (PEFT + TRL SFTTrainer, requires NVIDIA GPU)
- `baseline_eval.py` — evaluates model before/after fine-tuning, saves detailed results JSON
- `compare_results.py` — loads baseline + finetuned results, prints table, saves comparison.md
- `failure_analysis.py` — categorizes failures by field, source (SROIE vs synthetic), and error type (missing/partial/wrong_value); saves results/failure_analysis.md
- `demo.py` — CLI inference script; supports `--text`, `--example 1/2/3`, and interactive (Ctrl+D) modes; shows per-field confidence scores from token logprobs
- `app.py` — Gradio web UI for receipt extraction; run with `python app.py`, opens at localhost:7860
- `requirements.txt` — all deps, no comments; includes gradio and huggingface_hub

## Key decisions and why

- **Rank 8 not 16** — task is constrained (4 fixed fields), ~1000 examples. Lower rank = less overfitting risk.
- **`rth/sroie-2019-v2` not `darentang/sroie`** — darentang uses deprecated dataset scripts, rth is Parquet-based and works with modern datasets library.
- **Drop image column immediately after load** — rth dataset has an image field that requires Pillow; we don't need images so we remove it to avoid the dependency.
- **SROIE fields under `objects.entities`** — nested dict, not top-level. `_get_entities()` helper handles this with a top-level fallback.
- **1000 synthetic samples** — SROIE is Malaysian receipts; synthetic fills the Indian-specific domain (GSTIN, INR formats, local store names). 1000 gives ~43% synthetic in final mix.
- **System prompt lives in config.py** — was duplicated across 4 files originally; now single source of truth.
- **OPENAI_API_KEY loaded via python-dotenv** — user keeps key in .env file, `load_dotenv()` called at top of generate_synthetic.py.
- **Confidence scoring via token logprobs** — `demo.py` uses `output_scores=True` + `return_dict_in_generate=True`, finds each field value's token subsequence in the output, and averages token probabilities. Flags fields below 70% confidence.

## Gotchas discovered

- `darentang/sroie` raises `RuntimeError: Dataset scripts are no longer supported` with modern datasets library — switched to `rth/sroie-2019-v2`
- `rth/sroie-2019-v2` crashes on iteration without Pillow because of the image column — fixed by `remove_columns(["image"])` right after load
- `baseline_eval.py` had `--test-data` as `required=True` — changed to use `config.FINAL_TEST` as default so it runs with no args
- Model default in baseline_eval.py was `Qwen2.5-1.5B-Instruct` — updated to 3B to match the rest of the pipeline
- Gradio 6.0 moved `theme` from `gr.Blocks()` to `launch()` — pass `theme=gr.themes.Soft()` in `demo.launch()` not `gr.Blocks()`
- HuggingFace username is `largetrader`, not `Gaur-Ayush` — use `largetrader/...` in all `repo_id` values
- `huggingface-cli` binary may not be on PATH in conda envs — use `python -c "from huggingface_hub import HfApi; ..."` instead

## Results (Vast.ai RTX 3090 run)

| Field | Before | After | Delta |
|-------|--------|-------|-------|
| Company | 75.1% | 87.3% | +12.2% |
| Date | 53.3% | 97.5% | +44.2% |
| Address (exact) | 47.7% | 69.0% | +21.3% |
| Address (fuzzy) | 94.6% | 97.0% | +2.4% |
| Total | 89.3% | 96.9% | +7.6% |
| **All fields correct** | **32.5%** | **60.4%** | **+27.9%** |
| JSON parse failures | 0.5% | 0.0% | -0.5% |

## Failure analysis findings

- 78/197 examples fail (39.6%). Address accounts for 61 of those 78 failures (78%).
- Address fails at 31% — primary mode is wrong_value (40 cases) then partial (21 cases).
- Failure is NOT domain-specific: SROIE fails at 37%, synthetic at 41.5% — nearly identical.
- Address errors are formatting disagreements, not comprehension failures: punctuation normalization, multi-line OCR bleeding the store trade name into the address field, and the model correcting OCR typos when the ground truth preserves them.
- Date (2.5%), total (3.0%) are effectively solved. Company (12.7%) is mostly wrong_value formatting differences.
- Full breakdown in `results/failure_analysis.md`.

## Current state

**Done:**
- Synthetic data generation (1000 examples)
- SROIE processing (973 examples)
- Dataset merge + stratified split
- QLoRA fine-tuning via PEFT + TRL on RTX 3090 (Vast.ai)
- Baseline + finetuned eval
- Failure analysis (failure_analysis.py + results/failure_analysis.md)
- Per-field confidence scoring in demo.py (token logprobs)
- CLI demo script (`demo.py`)
- Gradio web UI (`app.py`) — `python app.py` opens at localhost:7860
- Adapters pushed to HuggingFace Hub: `largetrader/qwen2.5-3b-receipt-extraction-lora`
- `__pycache__` untracked from git; `.gitignore` updated

**Cleaned up (removed):**
- MLX adapter checkpoints (adapters/ folder) — weights are on HuggingFace
- `lora_config.yaml` — MLX-LM training config, replaced by `train_peft.py`
- `colab_train.ipynb` — never used
- `training.log` — MLX stdout log
- `plot_loss.py` — MLX loss plotter

**HuggingFace repo:** https://huggingface.co/largetrader/qwen2.5-3b-receipt-extraction-lora

**Planned (not done):**
- vLLM serving + throughput benchmark on Vast.ai RTX 3090
