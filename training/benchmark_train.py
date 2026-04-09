#!/usr/bin/env python3
"""
Simplified training script for benchmarking purposes.
Runs a fixed number of steps and outputs detailed metrics.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Training Script")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training if applicable."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def format_prompt(example):
    """Format the training examples."""
    return example


def main():
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Track metrics
    metrics = {
        "scenario": os.path.basename(args.output_dir),
        "start_time": datetime.now().isoformat(),
        "model_name": args.model_name,
        "world_size": world_size,
        "rank": rank,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation * world_size,
        "train_data": args.train_data,
        "val_data": args.val_data,
    }
    
    if is_main:
        print("=" * 60)
        print("BENCHMARK TRAINING")
        print("=" * 60)
        print(f"Scenario: {metrics['scenario']}")
        print(f"World Size: {world_size}")
        print(f"Max Steps: {args.max_steps}")
        print(f"Effective Batch Size: {metrics['effective_batch_size']}")
        print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load datasets
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    val_dataset = load_dataset("json", data_files=args.val_data, split="train")
    
    train_dataset = train_dataset.map(format_prompt)
    val_dataset = val_dataset.map(format_prompt)
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, peft_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="no",  # Don't save during benchmark
        eval_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        report_to="none",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    
    # Train and measure time
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    steps_per_sec = args.max_steps / elapsed_time
    samples_per_sec = (args.max_steps * args.batch_size * args.gradient_accumulation * world_size) / elapsed_time
    
    metrics.update({
        "end_time": datetime.now().isoformat(),
        "elapsed_seconds": elapsed_time,
        "steps_per_second": steps_per_sec,
        "samples_per_second": samples_per_sec,
        "final_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
    })
    
    # Save metrics
    if is_main:
        metrics_file = Path(args.output_dir) / "benchmark_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total Time: {elapsed_time:.2f}s")
        print(f"Steps/sec: {steps_per_sec:.4f}")
        print(f"Samples/sec: {samples_per_sec:.2f}")
        print(f"Results saved to: {metrics_file}")
        print("=" * 60)
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
