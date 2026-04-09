#!/usr/bin/env python3
"""
DDP Training Template for DGX Spark 2-Node Cluster

A production-ready template with:
  - DistributedSampler for correct data sharding
  - Rank-aware checkpointing (only rank 0 saves)
  - Proper model saving/loading across nodes
  - Gradient accumulation support
  - Mixed precision (bf16) training

Usage:
  # Single command (from spark-dgx-1):
  ./scripts/launch_distributed.sh training/ddp_training_template.py \\
      --model meta-llama/Llama-3.2-3B-Instruct \\
      --dataset /mnt/rosa-storage/my_dataset \\
      --output /mnt/rosa-models/my_finetuned_model \\
      --epochs 3

  # Or manual two-node launch:
  ./scripts/distributed_train.sh 0 training/ddp_training_template.py --epochs 3
  ./scripts/distributed_train.sh 1 training/ddp_training_template.py --epochs 3
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    parser = argparse.ArgumentParser(description="DDP Training Template")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca",
                        help="HuggingFace dataset name or path to local dataset")
    parser.add_argument("--output", type=str, default="/mnt/rosa-models/ddp-output",
                        help="Output directory for model and checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--checkpoint_every", type=int, default=500,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Name of the text field in the dataset")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed process group."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def log(rank, msg):
    """Print only from rank 0."""
    if rank == 0:
        print(msg, flush=True)


def load_and_prepare_dataset(args, tokenizer, rank, world_size):
    """Load dataset and create distributed data loader."""
    # Load dataset
    if os.path.isdir(args.dataset):
        dataset = load_from_disk(args.dataset)
        if "train" in dataset:
            dataset = dataset["train"]
    else:
        dataset = load_dataset(args.dataset, split="train")

    log(rank, f"Dataset loaded: {len(dataset)} samples")

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples[args.text_field],
            truncation=True,
            max_length=args.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    tokenized.set_format("torch")

    # DistributedSampler — each rank gets a different shard
    sampler = DistributedSampler(
        tokenized,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    return loader, sampler


def load_model(args, local_rank):
    """Load model with 4-bit quantization and LoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

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

    return model


def save_checkpoint(model, optimizer, scheduler, epoch, step, args, rank):
    """Save checkpoint — only rank 0 saves to avoid corruption."""
    if rank != 0:
        return

    checkpoint_dir = Path(args.output) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint-epoch{epoch}-step{step}"

    # Save LoRA adapter weights
    model.module.save_pretrained(str(checkpoint_path))

    # Save optimizer and scheduler state
    torch.save({
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, checkpoint_path / "training_state.pt")

    print(f"[checkpoint] Saved to {checkpoint_path}", flush=True)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, local_rank):
    """Load checkpoint — all ranks load model, only need one copy of optimizer state."""
    from peft import PeftModel

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, checkpoint_path)

    # Load optimizer/scheduler state
    state_path = Path(checkpoint_path) / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=f"cuda:{local_rank}")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        return state["epoch"], state["step"]

    return 0, 0


def train(args):
    rank, world_size, local_rank = setup_distributed()

    log(rank, f"\n{'='*60}")
    log(rank, f"  DDP Training — {world_size} nodes")
    log(rank, f"  Model: {args.model}")
    log(rank, f"  Effective batch size: {args.batch_size * args.grad_accum * world_size}")
    log(rank, f"  Output: {args.output}")
    log(rank, f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    log(rank, "Loading model...")
    model = load_model(args, local_rank)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(rank, f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Load data
    log(rank, "Loading dataset...")
    loader, sampler = load_and_prepare_dataset(args, tokenizer, rank, world_size)
    log(rank, f"Steps per epoch: {len(loader)}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(loader) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    # Resume from checkpoint if specified
    start_epoch, start_step = 0, 0
    if args.resume_from:
        log(rank, f"Resuming from {args.resume_from}...")
        start_epoch, start_step = load_checkpoint(
            model.module, optimizer, scheduler, args.resume_from, local_rank
        )
        log(rank, f"Resumed at epoch {start_epoch}, step {start_step}")

    # Training loop
    log(rank, "\nStarting training...\n")
    global_step = start_step
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        # IMPORTANT: set epoch for DistributedSampler to shuffle differently each epoch
        sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(local_rank)
            attention_mask = batch["attention_mask"].to(local_rank)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += loss.item() * args.grad_accum
                epoch_steps += 1

                if global_step % 50 == 0:
                    avg_loss = epoch_loss / epoch_steps
                    elapsed = time.time() - train_start
                    log(rank, f"  Epoch {epoch+1}/{args.epochs} | "
                         f"Step {global_step} | "
                         f"Loss: {avg_loss:.4f} | "
                         f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                         f"Time: {elapsed:.0f}s")

                # Checkpoint
                if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                    dist.barrier()
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, rank)
                    dist.barrier()

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        log(rank, f"\n  Epoch {epoch+1} complete — Avg loss: {avg_epoch_loss:.4f}\n")

    # Save final model — rank 0 only
    dist.barrier()
    if rank == 0:
        final_path = Path(args.output) / "final"
        final_path.mkdir(parents=True, exist_ok=True)

        model.module.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        # Save training metadata
        metadata = {
            "model": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum * world_size,
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "total_steps": global_step,
            "training_time_seconds": time.time() - train_start,
            "world_size": world_size,
        }
        with open(final_path / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModel saved to {final_path}", flush=True)
        print(f"Training completed in {time.time() - train_start:.1f}s", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    train(args)
