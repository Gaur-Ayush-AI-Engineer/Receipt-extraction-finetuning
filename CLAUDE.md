# CLAUDE.md — Fine-tuning Project Context

## What was built

End-to-end LoRA fine-tuning pipeline for structured receipt field extraction (company, date, address, total) using Qwen2.5-3B-Instruct on an M3 Pro 32GB Mac via MLX-LM.

**Files and what they do:**
- `config.py` — single source of truth for all variables (model, paths, ratios, system prompt, synthetic count). All scripts import from here.
- `generate_synthetic.py` — generates 1000 Indian receipt OCR + extraction pairs via GPT-4o-mini
- `prepare_sroie.py` — downloads `rth/sroie-2019-v2` from HuggingFace, extracts fields from nested `objects.entities`, drops the image column before iteration
- `merge_datasets.py` — merges SROIE + synthetic, stratified 80/10/10 split (test set guaranteed to have both sources + all difficulty levels)
- `lora_config.yaml` — MLX-LM training config (rank=8, alpha=16, 16 lora layers, batch=8, 1000 iters, cosine LR with warmup)
- `baseline_eval.py` — evaluates model before/after fine-tuning, saves detailed results JSON
- `compare_results.py` — loads baseline + finetuned results, prints table, saves comparison.md
- `demo.py` — CLI inference script; supports `--text`, `--example 1/2/3`, and interactive (Ctrl+D) modes
- `app.py` — Gradio web UI for receipt extraction; run with `python app.py`, opens at localhost:7860
- `requirements.txt` — all deps, no comments; includes gradio and huggingface_hub

## Key decisions and why

- **Rank 8 not 16** — task is constrained (4 fixed fields), ~1000 examples. Lower rank = less overfitting risk.
- **`rth/sroie-2019-v2` not `darentang/sroie`** — darentang uses deprecated dataset scripts, rth is Parquet-based and works with modern datasets library.
- **Drop image column immediately after load** — rth dataset has an image field that requires Pillow; we don't need images so we remove it to avoid the dependency.
- **SROIE fields under `objects.entities`** — nested dict, not top-level. `_get_entities()` helper handles this with a top-level fallback.
- **1000 synthetic samples** — SROIE is Malaysian receipts; synthetic fills the Indian-specific domain (GSTIN, INR formats, local store names). 1000 gives ~43% synthetic in final mix.
- **`max_seq_length: 1024`** — a few examples hit 1035-1036 tokens and get truncated. Acceptable trade-off; bump to 1280 if retraining.
- **System prompt lives in config.py** — was duplicated across 4 files originally; now single source of truth.
- **OPENAI_API_KEY loaded via python-dotenv** — user keeps key in .env file, `load_dotenv()` called at top of generate_synthetic.py.

## Gotchas discovered

- `darentang/sroie` raises `RuntimeError: Dataset scripts are no longer supported` with modern datasets library — switched to `rth/sroie-2019-v2`
- `rth/sroie-2019-v2` crashes on iteration without Pillow because of the image column — fixed by `remove_columns(["image"])` right after load
- `baseline_eval.py` had `--test-data` as `required=True` — changed to use `config.FINAL_TEST` as default so it runs with no args
- Model default in baseline_eval.py was `Qwen2.5-1.5B-Instruct` — updated to 3B to match the rest of the pipeline
- MLX-LM does not save a loss log by default (`report_to: null`) — training loss was only visible in stdout and not persisted
- Gradio 6.0 moved `theme` from `gr.Blocks()` to `launch()` — pass `theme=gr.themes.Soft()` in `demo.launch()` not `gr.Blocks()`
- HuggingFace username is `largetrader`, not `Gaur-Ayush` — use `largetrader/...` in all `repo_id` values
- `huggingface-cli` binary may not be on PATH in conda envs — use `python -c "from huggingface_hub import HfApi; ..."` instead

## Results (training complete)

| Field | Before | After | Delta |
|-------|--------|-------|-------|
| Company | 75.1% | 90.4% | +15.2% |
| Date | 53.3% | 98.0% | +44.7% |
| Address (exact) | 47.7% | 77.2% | +29.4% |
| Total | 89.3% | 98.0% | +8.6% |
| **All fields correct** | **32.5%** | **71.1%** | **+38.6%** |
| JSON parse failures | 0.5% | 0.0% | -0.5% |

No regressions. All remaining failures are partial matches (57 examples).

## Current state

**Done:**
- Synthetic data generation (1000 examples)
- SROIE processing (973 examples)
- Dataset merge + stratified split
- LoRA fine-tuning (1000 iters, rank=8)
- Baseline + finetuned eval
- Results comparison + markdown
- CLI demo script (`demo.py`)
- Gradio web UI (`app.py`) — `python app.py` opens at localhost:7860
- Adapters pushed to HuggingFace Hub: `largetrader/qwen2.5-3b-receipt-extraction-lora` (26MB)
- README updated: Dataset Notes, Demo section, Model Weights section, training loss note
- `__pycache__` untracked from git; `.gitignore` updated

**Adapters saved at:** `./adapters/adapters.safetensors`
**HuggingFace repo:** https://huggingface.co/largetrader/qwen2.5-3b-receipt-extraction-lora

