"""
train_peft.py — LoRA fine-tuning with HuggingFace PEFT + TRL SFTTrainer
Target: NVIDIA GPU using 4-bit QLoRA

Hyperparameters match the original MLX-LM run:
  rank=8, alpha=16, dropout=0.05, last 16 of 36 layers
  effective batch=8, lr=1e-4 cosine with 100-step warmup, ~10 epochs

Usage:
    python train_peft.py
    python train_peft.py --precision bf16
    python train_peft.py --epochs 10 --output-dir ./adapters
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

import config


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--train-data", default=config.FINAL_TRAIN)
    parser.add_argument("--valid-data", default=config.FINAL_VALID)
    parser.add_argument("--output-dir", default=config.ADAPTERS_DIR)
    parser.add_argument("--epochs", type=int, default=10,
                        help="1000 MLX iters ≈ 10 epochs (800 train examples, batch=8)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch; grad-accum=2 → effective batch=8")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--precision",
        choices=["fp16", "bf16"],
        default="bf16",
        help="Use bf16 on supported GPUs. Use fp16 for older GPUs that do not support bf16.",
    )
    args = parser.parse_args()

    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "bf16 was requested, but this GPU does not support bf16. "
            "Use --precision fp16 on this machine."
        )

    compute_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    # ── 4-bit quantization (QLoRA) ───────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # On fp16-only GPUs, bf16 trainable tensors make AMP's GradScaler fail during
    # gradient unscale/clip. Keep frozen quantized weights untouched, but ensure
    # any normal parameters follow the selected precision.
    if args.precision == "fp16":
        for param in model.parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)

    # ── LoRA config ───────────────────────────────────────────────────────────
    # Qwen2.5-3B has 36 transformer layers (0–35).
    # layers_to_transform=[20..35] mirrors MLX lora_layers=16 (last 16 layers).
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=list(range(20, 36)),
        layers_pattern="layers",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # PEFT can inherit a different dtype for adapters. Keep trainable LoRA params
    # aligned with the selected mixed precision path.
    for param in model.parameters():
        if param.requires_grad and param.dtype != compute_dtype:
            param.data = param.data.to(compute_dtype)

    trainable_dtypes = {}
    for param in model.parameters():
        if param.requires_grad:
            trainable_dtypes[str(param.dtype)] = trainable_dtypes.get(str(param.dtype), 0) + param.numel()
    print(f"Trainable parameter dtypes: {trainable_dtypes}")

    # ── Data loading & formatting ─────────────────────────────────────────────
    # Pre-format JSONL messages to flat text so SFTTrainer gets a plain "text" field.
    def fmt(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_dataset = Dataset.from_list(load_jsonl(args.train_data))
    valid_dataset = Dataset.from_list(load_jsonl(args.valid_data))
    train_dataset = train_dataset.map(fmt, remove_columns=["messages"])
    valid_dataset = valid_dataset.map(fmt, remove_columns=["messages"])

    print(f"Train: {len(train_dataset)} examples | Valid: {len(valid_dataset)} examples")

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,    # effective batch = 4×2 = 8
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # ── Save LoRA adapter weights only ────────────────────────────────────────
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Adapters saved → {output_path}")


if __name__ == "__main__":
    main()
