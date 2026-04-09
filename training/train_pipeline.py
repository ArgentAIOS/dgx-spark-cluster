#!/usr/bin/env python3
"""
4-Stage Distributed Training Pipeline for DGX Spark Cluster

Stages:
  1. Data prep    — Rank 0 only: validates files, creates output dir, barrier
  2. Train        — Both nodes: DDP fine-tuning with LoRA, auto-checkpoints
  3. Eval         — Both nodes: runs during training via HF Trainer eval
  4. Merge        — Rank 0 only: merge_and_unload() → full model in merged/

Usage:
  ./scripts/launch_cluster.sh training/train_pipeline.py \
      --model meta-llama/Llama-3.2-3B-Instruct \
      --train-data /mnt/rosa-storage/data/train.jsonl \
      --output-dir /mnt/rosa-models/my-model-v1 \
      --epochs 3

  # With validation data:
  ./scripts/launch_cluster.sh training/train_pipeline.py \
      --model meta-llama/Llama-3.2-3B-Instruct \
      --train-data /mnt/rosa-storage/data/train.jsonl \
      --val-data /mnt/rosa-storage/data/val.jsonl \
      --output-dir /mnt/rosa-models/my-model-v1 \
      --epochs 3

  # With GPU Direct:
  NCCL_MODE=dmabuf ./scripts/launch_cluster.sh training/train_pipeline.py [args]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


def parse_args():
    p = argparse.ArgumentParser(description="DGX Spark Training Pipeline")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--train-data", type=str, required=True,
                   help="Path to training data (JSONL, CSV, or HF dataset dir)")
    p.add_argument("--val-data", type=str, default=None,
                   help="Path to validation data (optional — splits from train if omitted)")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output directory for checkpoints and final model")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--val-split", type=float, default=0.05,
                   help="Fraction of train data for validation if --val-data not provided")
    p.add_argument("--text-field", type=str, default="text",
                   help="Name of the text column in the dataset")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--no-merge", action="store_true",
                   help="Skip Stage 4 (LoRA merge)")
    return p.parse_args()


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def banner(rank, stage, title):
    log(rank, f"\n{'='*60}")
    log(rank, f"  Stage {stage}: {title}")
    log(rank, f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Data Prep (Rank 0 only)
# ═══════════════════════════════════════════════════════════════════════════

def stage_data_prep(args, rank, tokenizer):
    banner(rank, 1, "Data Preparation (rank 0)")

    train_dataset = None
    val_dataset = None

    if rank == 0:
        # Validate input files exist
        train_path = Path(args.train_data)
        if not train_path.exists():
            print(f"ERROR: Training data not found: {args.train_data}", flush=True)
            dist.barrier()
            sys.exit(1)

        # Create output directory
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)

        log(rank, f"  Train data: {args.train_data}")
        log(rank, f"  Val data:   {args.val_data or f'(split {args.val_split:.0%} from train)'}")
        log(rank, f"  Output:     {args.output_dir}")

        # Load training data
        if train_path.suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(train_path), split="train")
        elif train_path.suffix == ".csv":
            dataset = load_dataset("csv", data_files=str(train_path), split="train")
        elif train_path.is_dir():
            dataset = load_dataset(str(train_path), split="train")
        else:
            dataset = load_dataset(str(train_path), split="train")

        log(rank, f"  Loaded {len(dataset)} training samples")

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples[args.text_field],
                truncation=True,
                max_length=args.max_seq_len,
                padding=False,
            )

        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

        # Load or split validation data
        if args.val_data:
            val_path = Path(args.val_data)
            if not val_path.exists():
                print(f"ERROR: Validation data not found: {args.val_data}", flush=True)
                dist.barrier()
                sys.exit(1)

            if val_path.suffix == ".jsonl":
                val_raw = load_dataset("json", data_files=str(val_path), split="train")
            else:
                val_raw = load_dataset(str(val_path), split="train")

            val_dataset = val_raw.map(tokenize_fn, batched=True, remove_columns=val_raw.column_names)
            train_dataset = dataset
            log(rank, f"  Loaded {len(val_dataset)} validation samples")
        else:
            split = dataset.train_test_split(test_size=args.val_split, seed=42)
            train_dataset = split["train"]
            val_dataset = split["test"]
            log(rank, f"  Split: {len(train_dataset)} train / {len(val_dataset)} val")

        # Save metadata
        metadata = {
            "model": args.model,
            "train_data": args.train_data,
            "val_data": args.val_data,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "max_seq_len": args.max_seq_len,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(Path(args.output_dir) / "pipeline_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    # Barrier — all ranks wait for rank 0 to finish data prep
    dist.barrier()
    log(rank, "  Stage 1 complete — all ranks synchronized\n")

    return train_dataset, val_dataset


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 & 3: Train + Eval (Both Nodes via HF Trainer)
# ═══════════════════════════════════════════════════════════════════════════

def stage_train_and_eval(args, rank, world_size, local_rank, tokenizer,
                          train_dataset, val_dataset):
    banner(rank, "2+3", "Training & Evaluation (both nodes)")

    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log(rank, "  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(rank, f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    effective_batch = args.batch_size * args.grad_accum * world_size
    log(rank, f"  Effective batch size: {effective_batch}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
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
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
        seed=42,
    )

    # Build trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train (Stage 2) with eval during training (Stage 3)
    log(rank, "  Starting training...\n")
    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start

    # Log results
    log(rank, f"\n  Training complete in {train_time:.1f}s")
    log(rank, f"  Final train loss: {train_result.training_loss:.4f}")

    # Save adapter (rank 0 only, handled by Trainer)
    adapter_path = Path(args.output_dir) / "adapter"
    if rank == 0:
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        log(rank, f"  Adapter saved to {adapter_path}")

    dist.barrier()
    log(rank, "  Stages 2+3 complete\n")

    return train_result, train_time


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: Merge LoRA → Full Model (Rank 0 only)
# ═══════════════════════════════════════════════════════════════════════════

def stage_merge(args, rank):
    banner(rank, 4, "Merge LoRA Adapter (rank 0)")

    if rank == 0:
        adapter_path = Path(args.output_dir) / "adapter"
        merged_path = Path(args.output_dir) / "merged"

        log(rank, f"  Loading base model: {args.model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        log(rank, f"  Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))

        log(rank, "  Merging weights...")
        model = model.merge_and_unload()

        log(rank, f"  Saving merged model to {merged_path}")
        merged_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged_path))

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(merged_path))

        log(rank, f"  Merged model saved to {merged_path}")

    dist.barrier()
    log(rank, "  Stage 4 complete\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    rank, world_size, local_rank = setup()

    log(rank, "\n" + "=" * 60)
    log(rank, "  DGX Spark Training Pipeline")
    log(rank, f"  Model:  {args.model}")
    log(rank, f"  Nodes:  {world_size}")
    log(rank, f"  NCCL:   {os.environ.get('NCCL_NET_GDR_LEVEL', '?')}/{os.environ.get('NCCL_DMABUF_ENABLE', '?')}")
    log(rank, "=" * 60)

    pipeline_start = time.time()

    # Load tokenizer (all ranks need this)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Stage 1: Data Prep (rank 0) ──────────────────────────────────────
    train_dataset, val_dataset = stage_data_prep(args, rank, tokenizer)

    # Broadcast datasets to all ranks (rank 0 did the loading)
    if rank != 0:
        # Non-zero ranks need to load the data too for their shard
        # Trainer handles DistributedSampler internally
        train_path = Path(args.train_data)
        if train_path.suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(train_path), split="train")
        elif train_path.suffix == ".csv":
            dataset = load_dataset("csv", data_files=str(train_path), split="train")
        else:
            dataset = load_dataset(str(train_path), split="train")

        def tokenize_fn(examples):
            return tokenizer(
                examples[args.text_field],
                truncation=True,
                max_length=args.max_seq_len,
                padding=False,
            )

        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

        if args.val_data:
            val_path = Path(args.val_data)
            if val_path.suffix == ".jsonl":
                val_raw = load_dataset("json", data_files=str(val_path), split="train")
            else:
                val_raw = load_dataset(str(val_path), split="train")
            val_dataset = val_raw.map(tokenize_fn, batched=True, remove_columns=val_raw.column_names)
            train_dataset = dataset
        else:
            split = dataset.train_test_split(test_size=args.val_split, seed=42)
            train_dataset = split["train"]
            val_dataset = split["test"]

    dist.barrier()

    # ── Stage 2+3: Train + Eval (both nodes) ─────────────────────────────
    train_result, train_time = stage_train_and_eval(
        args, rank, world_size, local_rank, tokenizer,
        train_dataset, val_dataset
    )

    # ── Stage 4: Merge (rank 0) ──────────────────────────────────────────
    if not args.no_merge:
        stage_merge(args, rank)

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    log(rank, "\n" + "=" * 60)
    log(rank, "  PIPELINE COMPLETE")
    log(rank, f"  Total time:  {total_time:.1f}s ({total_time/60:.1f} min)")
    log(rank, f"  Train time:  {train_time:.1f}s")
    log(rank, f"  Train loss:  {train_result.training_loss:.4f}")
    log(rank, f"  Output:      {args.output_dir}")
    if not args.no_merge:
        log(rank, f"  Merged:      {args.output_dir}/merged/")
    log(rank, f"  Adapter:     {args.output_dir}/adapter/")
    log(rank, "=" * 60 + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
