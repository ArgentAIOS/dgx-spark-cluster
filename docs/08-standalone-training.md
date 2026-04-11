# Standalone Training on a Single DGX Spark

**You don't need two Sparks to train.** This guide covers single-node fine-tuning on one DGX Spark — no cluster, no NCCL, no distributed setup. Just one machine training a model.

Start here if you have one DGX Spark, or if you want to validate your training script before going distributed.

---

## When to Use Single-Node vs. Distributed

| Scenario | Use | Why |
|---|---|---|
| Model fits in 128 GB | **Single node** | No network overhead, simpler setup |
| Learning / first time | **Single node** | Get it working, then scale |
| Model > 128 GB | **Distributed (2 nodes)** | Must shard across memory pools |
| Large dataset, want faster epochs | **Distributed (2 nodes)** | 1.87x speedup with 93.5% efficiency |
| Quick LoRA fine-tune (3B-8B) | **Single node** | Finishes in minutes, not worth the setup |
| Production training run (70B+) | **Distributed (2 nodes)** | Saves hours per epoch |

---

## Prerequisites

```bash
# Verify GPU
nvidia-smi

# Verify PyTorch + CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Install training dependencies
pip3 install transformers datasets peft trl accelerate bitsandbytes
```

---

## Quick Start: Fine-Tune Llama 3.2 3B with LoRA

This is the simplest possible training run — a LoRA fine-tune on a small model using the Hugging Face SFTTrainer.

```python
#!/usr/bin/env python3
"""Single-node LoRA fine-tuning on DGX Spark. No distributed setup needed."""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./models/llama3-lora-test"
DATASET = "tatsu-lab/alpaca"  # Or path to your local JSONL

# --- Load model with 4-bit quantization (saves memory) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# --- LoRA configuration ---
lora_config = LoraConfig(
    r=32,                   # Rank — higher = more capacity, more memory
    lora_alpha=64,          # Scaling factor (usually 2x rank)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should be ~1-2% of total

# --- Load dataset ---
dataset = load_dataset(DATASET, split="train")

def format_instruction(example):
    """Format into instruction-response pairs."""
    if example.get("input"):
        return {"text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"}
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

dataset = dataset.map(format_instruction)

# --- Train ---
training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # Effective batch = 2 * 8 = 16
    learning_rate=2e-4,
    bf16=True,                       # Use bfloat16 (native on Grace Blackwell)
    logging_steps=10,
    save_strategy="epoch",
    warmup_steps=50,
    max_seq_length=2048,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_config,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Training complete. Model saved to {OUTPUT_DIR}")
```

### Run It

```bash
python3 train_single_node.py
```

That's it. No environment variables, no launchers, no NCCL. Just run it.

---

## Using Shared Storage (ROSA)

If your DGX Spark is connected to the ROSA NVMe-TCP storage, use it for datasets and model output. This keeps local disk clean and makes models accessible from other nodes.

```bash
# Use ROSA for output
python3 train_single_node.py --output_dir /mnt/rosa-models/my-finetune-v1

# Use ROSA for datasets
python3 train_single_node.py --train_data /mnt/rosa-storage/datasets/my-data.jsonl
```

See [ROSA Initiator Setup](rosa/02-spark-initiator-setup.md) for mount configuration.

---

## Launch Script (With Timing and Config Display)

For repeatable training runs, use a launch script that shows your configuration and times the run:

```bash
#!/bin/bash
# launch_local_training.sh — single-node training with timing
set -e

# --- Configuration ---
OUTPUT_DIR="/mnt/rosa-models/llama3-lora-v1"
TRAIN_DATA="/mnt/rosa-storage/datasets/train.jsonl"
VAL_DATA="/mnt/rosa-storage/datasets/val.jsonl"

EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=2e-4
LORA_RANK=32
LORA_ALPHA=64
MAX_SEQ_LEN=2048

# --- Environment ---
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "Single-Node Training — $(hostname)"
echo "============================================================"
echo "  Output:     $OUTPUT_DIR"
echo "  Data:       $TRAIN_DATA"
echo "  Epochs:     $EPOCHS"
echo "  Batch:      $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  LR:         $LR"
echo "  LoRA:       rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "  Max Seq:    $MAX_SEQ_LEN"
echo "============================================================"

START_TIME=$(date +%s)

python3 train_single_node.py \
    --output_dir "$OUTPUT_DIR" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation $GRAD_ACCUM \
    --learning_rate $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --max_seq_length $MAX_SEQ_LEN

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "Training completed in $((DURATION / 60))m $((DURATION % 60))s"
echo "Model saved to: $OUTPUT_DIR"
```

```bash
chmod +x launch_local_training.sh
./launch_local_training.sh
```

---

## After Training: Merge LoRA and Test

LoRA training saves only the adapter weights (~50-200 MB). To get a standalone model, merge the adapter back into the base model.

### Merge

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER = "/mnt/rosa-models/llama3-lora-v1"
MERGED = "/mnt/rosa-models/llama3-lora-v1/merged"

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()

model.save_pretrained(MERGED)
AutoTokenizer.from_pretrained(BASE).save_pretrained(MERGED)
print(f"Merged model saved to {MERGED}")
```

### Test

```python
from transformers import pipeline
import torch

pipe = pipeline("text-generation", model="/mnt/rosa-models/llama3-lora-v1/merged",
                torch_dtype=torch.bfloat16, device_map="auto")

result = pipe("### Instruction:\nExplain what a GPU does.\n\n### Response:\n",
              max_new_tokens=256, do_sample=True, temperature=0.7)
print(result[0]["generated_text"])
```

---

## Memory Guide for DGX Spark

The DGX Spark has 128 GB of unified memory shared between CPU and GPU. Here's what fits:

| Model | Precision | Memory Needed | Fits on 1 Spark? |
|---|---|---|---|
| Llama 3.2 1B | bf16 | ~2 GB | Yes |
| Llama 3.2 3B | bf16 | ~6 GB | Yes |
| Llama 3.1 8B | 4-bit (LoRA) | ~8 GB | Yes |
| Llama 3.1 70B | 4-bit (LoRA) | ~40 GB | Yes |
| Llama 3.1 70B | bf16 (full) | ~140 GB | No — use distributed |
| Llama 3.1 405B | 4-bit (LoRA) | ~200 GB | No — use distributed |

**Tips for fitting larger models:**
- Use 4-bit quantization (`BitsAndBytesConfig(load_in_4bit=True)`)
- Use LoRA instead of full fine-tuning (trains 1-2% of parameters)
- Reduce `max_seq_length` (2048 → 1024 halves sequence memory)
- Reduce `batch_size` and increase `gradient_accumulation` (same effective batch, less memory)

---

## Ready to Scale?

Once your training script works on a single node, scaling to two nodes is straightforward:

1. Wrap your model with `DistributedDataParallel`
2. Add a `DistributedSampler` to your dataloader
3. Use the cluster launcher script

See [Distributed Training](02-distributed-training.md) for the full guide.
