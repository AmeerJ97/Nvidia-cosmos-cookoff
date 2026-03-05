#!/usr/bin/env python3
"""
CLASP QLoRA Fine-Tuning — Cosmos-Reason2-8B (Qwen3-VL-8B-Instruct)

Trains on golden SFT records from the CLASP survival game.
Produces a LoRA adapter checkpoint for downstream deployment.

Usage:
    python train_qlora.py                              # defaults
    python train_qlora.py --epochs 3 --lr 2e-4         # custom
    python train_qlora.py --model nvidia/cosmos-reason2-8b  # HF model ID
"""
import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# Qwen3-VL specific import
try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3_VL = True
except ImportError:
    HAS_QWEN3_VL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("clasp.train")

# Default model — Cosmos-Reason2-8B is Qwen3-VL-8B-Instruct
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load_sft_data(path: str) -> Dataset:
    """Load OpenAI chat-format JSONL into a HuggingFace Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    log.info("Loaded %d SFT records from %s", len(records), path)
    return Dataset.from_list(records)


def format_chat(example, tokenizer):
    """Convert OpenAI chat format to tokenized input/output."""
    messages = example["messages"]
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="CLASP QLoRA Fine-Tuning")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data", type=str, default="sft_dataset.openai.jsonl",
                        help="Path to SFT JSONL (OpenAI chat format)")
    parser.add_argument("--output", type=str, default="./clasp-lora-checkpoint",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    parser.add_argument("--lora-r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--wandb-project", type=str, default="clasp-cosmos",
                        help="W&B project name (set WANDB_DISABLED=true to skip)")
    args = parser.parse_args()

    # ── Verify GPU ────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        log.error("No CUDA GPU available")
        return
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("GPU: %s (%.1f GB VRAM)", gpu_name, gpu_mem)

    # ── Load tokenizer ────────────────────────────────────────────────────
    log.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load & prepare dataset ────────────────────────────────────────────
    dataset = load_sft_data(args.data)

    # Apply chat template
    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=["messages"],
    )

    # Tokenize
    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize, remove_columns=["text"])

    # Train/eval split (90/10, grouped by content to avoid leakage)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    log.info("Train: %d, Eval: %d", len(train_dataset), len(eval_dataset))

    # ── Quantization config ───────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load model ────────────────────────────────────────────────────────
    log.info("Loading model %s with QLoRA (4-bit NF4)...", args.model)

    # Cosmos-Reason2-8B = Qwen3-VL-8B-Instruct (vision-language model)
    load_kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if HAS_QWEN3_VL and "vl" in args.model.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, **load_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, **load_kwargs
        )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(
        "LoRA: r=%d alpha=%d | Trainable: %d / %d (%.2f%%)",
        args.lora_r, args.lora_alpha,
        trainable, total, 100 * trainable / total,
    )

    # ── Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if os.environ.get("WANDB_DISABLED") != "true" else "none",
        run_name="clasp-qlora-cosmos-reason2",
        dataloader_num_workers=4,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
        ),
    )

    # ── Train ─────────────────────────────────────────────────────────────
    log.info("Starting QLoRA training...")
    log.info("  Epochs: %d", args.epochs)
    log.info("  Effective batch size: %d", args.batch_size * args.grad_accum)
    log.info("  Learning rate: %s", args.lr)
    log.info("  Max sequence length: %d", args.max_seq_len)

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    log.info("Saving LoRA adapter to %s", args.output)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Save training metadata
    meta = {
        "base_model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size * args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "train_records": len(train_dataset),
        "eval_records": len(eval_dataset),
        "final_eval_loss": trainer.state.best_metric,
        "gpu": gpu_name,
        "gpu_mem_gb": round(gpu_mem, 1),
    }
    with open(Path(args.output) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Training complete!")
    log.info("  Final eval loss: %.4f", trainer.state.best_metric)
    log.info("  Checkpoint: %s", args.output)
    log.info("")
    log.info("To use this adapter:")
    log.info("  from peft import PeftModel")
    log.info("  model = PeftModel.from_pretrained(base_model, '%s')", args.output)


if __name__ == "__main__":
    main()
