# Training Pipeline Guide

**End-to-end workflow:** data prep → distributed training → evaluation

This guide shows which steps run on **one node** vs. **both nodes**, and the exact commands for each stage.

---

## Pipeline Overview

```
 ┌─────────────────────────────────────────────────────────┐
 │  Stage 1: Data Preparation                    ONE NODE  │
 │  Download, clean, tokenize, save to shared storage      │
 └─────────────────────────┬───────────────────────────────┘
                           │ /mnt/rosa-storage/datasets/
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Stage 2: Distributed Training               BOTH NODES │
 │  LoRA fine-tuning with DDP across 2 DGX Sparks          │
 └─────────────────────────┬───────────────────────────────┘
                           │ /mnt/rosa-models/<run>/final/
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Stage 3: Evaluation                          ONE NODE  │
 │  Merge LoRA, evaluate, push to hub                      │
 └─────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Preparation (One Node)

Run on **spark-dgx-1 only**. Output goes to shared storage so both nodes can access it.

### 1a. Download and Process Dataset

```python
#!/usr/bin/env python3
# scripts/prepare_dataset.py — run on ONE node
from datasets import load_dataset

# Load from HuggingFace
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Or load from local JSONL
# dataset = load_dataset("json", data_files="/mnt/rosa-storage/raw/data.jsonl", split="train")

# Format for instruction tuning
def format_instruction(example):
    if example.get("input"):
        text = (f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}")
    else:
        text = (f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}")
    return {"text": text}

dataset = dataset.map(format_instruction)

# Save to shared storage (both nodes will read from here)
dataset.save_to_disk("/mnt/rosa-storage/datasets/alpaca-formatted")
print(f"Saved {len(dataset)} samples to /mnt/rosa-storage/datasets/alpaca-formatted")
```

```bash
# Run it
python3 scripts/prepare_dataset.py
```

### 1b. Verify Both Nodes Can See the Data

```bash
# On spark-dgx-1
ls /mnt/rosa-storage/datasets/alpaca-formatted/

# On spark-dgx-2 (should see the same files)
ssh sem@10.0.0.2 "ls /mnt/rosa-storage/datasets/alpaca-formatted/"
```

---

## Stage 2: Distributed Training (Both Nodes)

Uses the DDP training template with DistributedSampler. Each node gets a different shard of the data — no duplication.

### 2a. Source NCCL Environment

```bash
# On spark-dgx-1 (the launcher handles spark-dgx-2 automatically)
source configs/nccl-env.sh safe    # or: source configs/nccl-env.sh dmabuf
```

### 2b. Run Smoke Test First

```bash
# Single command — fires both nodes
./scripts/launch_distributed.sh training/validate_distributed.py
```

All 4 tests should pass. If bandwidth is below 5 GB/s, check NCCL settings.

### 2c. Launch Training

```bash
# Single command — both nodes
./scripts/launch_distributed.sh training/ddp_training_template.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset /mnt/rosa-storage/datasets/alpaca-formatted \
    --output /mnt/rosa-models/alpaca-llama3-run1 \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lora_rank 32 \
    --checkpoint_every 500
```

**What happens on each node:**
- `DistributedSampler` splits the dataset — rank 0 gets even indices, rank 1 gets odd
- Each node computes forward/backward on its shard
- NCCL ring all-reduce synchronizes gradients over the 200 Gb/s fabric
- Only rank 0 (spark-dgx-1) saves checkpoints and the final model

**Effective batch size:** `batch_size × grad_accum × world_size` = 2 × 4 × 2 = **16**

### 2d. Monitor Training

```bash
# Watch GPU utilization on both nodes
ssh sem@10.0.0.2 "nvidia-smi" &  # spark-dgx-2
nvidia-smi                        # spark-dgx-1

# Check NCCL is using the fabric (not TCP fallback)
# In NCCL_DEBUG=INFO output, look for:
#   NET/IB : Using [0]mlx5_0:1 [RoCE]    ← GOOD (RDMA)
#   NET/Socket : Using [0]enp1s0f0np0     ← BAD (TCP fallback)
```

### 2e. Resume from Checkpoint (if interrupted)

```bash
./scripts/launch_distributed.sh training/ddp_training_template.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset /mnt/rosa-storage/datasets/alpaca-formatted \
    --output /mnt/rosa-models/alpaca-llama3-run1 \
    --resume_from /mnt/rosa-models/alpaca-llama3-run1/checkpoints/checkpoint-epoch1-step500 \
    --epochs 3
```

---

## Stage 3: Evaluation (One Node)

Run on **spark-dgx-1 only**. The final model is already on shared storage.

### 3a. Merge LoRA Adapter into Base Model

```python
#!/usr/bin/env python3
# scripts/merge_and_save.py — run on ONE node
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/mnt/rosa-models/alpaca-llama3-run1/final"
MERGED_PATH = "/mnt/rosa-models/alpaca-llama3-run1/merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH}...")
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print("Done.")
```

### 3b. Quick Inference Test

```python
#!/usr/bin/env python3
# scripts/test_inference.py — run on ONE node
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_path = "/mnt/rosa-models/alpaca-llama3-run1/merged"

pipe = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompts = [
    "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to reverse a linked list.\n\n### Response:\n",
]

for prompt in prompts:
    result = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    print(result[0]["generated_text"])
    print("-" * 60)
```

### 3c. Benchmark Comparison

```bash
# Compare training metrics across runs
python3 -c "
import json
from pathlib import Path

for run_dir in sorted(Path('/mnt/rosa-models').glob('*/final/training_metadata.json')):
    meta = json.loads(run_dir.read_text())
    print(f\"{run_dir.parent.parent.name}:\")
    print(f\"  Steps: {meta['total_steps']}, Time: {meta['training_time_seconds']:.0f}s\")
    print(f\"  Effective batch: {meta['effective_batch_size']}, LR: {meta['lr']}\")
    print()
"
```

---

## Quick Reference — What Runs Where

| Step | Nodes | Storage | Command |
|------|-------|---------|---------|
| Data prep | **1** (spark-dgx-1) | Write to `/mnt/rosa-storage` | `python3 scripts/prepare_dataset.py` |
| Smoke test | **2** (both) | — | `./scripts/launch_distributed.sh training/validate_distributed.py` |
| Training | **2** (both) | Read from ROSA, write checkpoints to ROSA | `./scripts/launch_distributed.sh training/ddp_training_template.py ...` |
| Merge LoRA | **1** (spark-dgx-1) | Read/write `/mnt/rosa-models` | `python3 scripts/merge_and_save.py` |
| Evaluate | **1** (spark-dgx-1) | Read from `/mnt/rosa-models` | `python3 scripts/test_inference.py` |
