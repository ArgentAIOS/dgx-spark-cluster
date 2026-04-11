# Training AI Models on DGX Spark: A Beginner's Guide

**No prior experience required.** This guide explains what training is, why you'd do it, and how it works on your DGX Spark — written so anyone can follow along.

---

## What Is "Training" an AI Model?

Think of a large language model (like ChatGPT or Llama) as a student who has read millions of books. It knows a lot about everything, but it doesn't know anything specific about *your* business, *your* data, or *your* way of doing things.

**Training** (more precisely, **fine-tuning**) is like giving that student a crash course on your specific topic. You show it hundreds or thousands of examples of the kind of work you want it to do, and it learns the patterns.

### Before Training
```
You: "Draft a response to this personal injury intake call."
AI:  "I'd be happy to help! Here's a generic response..." (vague, generic)
```

### After Training on Your Data
```
You: "Draft a response to this personal injury intake call."
AI:  "Based on the caller's described injuries and the accident details,
      here's the intake summary following our firm's format..." (specific, useful)
```

The model doesn't forget everything it already knew — it just gets better at the specific things you taught it.

---

## Why Train Your Own Model?

| Reason | Example |
|---|---|
| **Privacy** | Your data never leaves your building. No cloud APIs, no third parties |
| **Specialization** | Train on legal docs, medical records, sales calls — whatever your domain |
| **Cost** | No per-token API fees. Run it as much as you want, forever |
| **Speed** | Local inference is fast — no network round trips to a cloud provider |
| **Control** | You own the model. Update it, share it, deploy it however you want |

---

## What You Need

You already have the hard part — the hardware:

| What | Your Setup | Why It Matters |
|---|---|---|
| **GPU** | DGX Spark (GB10, 128 GB) | Does the heavy math. More memory = bigger models |
| **Storage** | ROSA NVMe-TCP shared drive | Stores your training data and finished models |
| **Network** | 200 Gb/s direct link between Sparks | If using 2 nodes, this is how they talk to each other |
| **Software** | PyTorch + Hugging Face libraries | The tools that actually run the training |

---

## How Training Works (The Simple Version)

### Step 1: Prepare Your Data

Training data is just a list of examples. Each example shows the AI what a good answer looks like.

```json
{"instruction": "Summarize this call transcript.", "response": "The caller reported a rear-end collision on I-35..."}
{"instruction": "Rate the severity of these injuries.", "response": "Based on the described symptoms: moderate severity..."}
{"instruction": "Draft a demand letter.", "response": "Dear [Adjuster], On behalf of our client..."}
```

More examples = better training. A few hundred is the minimum. A few thousand is better.

### Step 2: Pick a Base Model

You don't train from scratch — you start with a model that already understands language and teach it your specifics. This is called **fine-tuning**.

Popular base models:

| Model | Size | Good For |
|---|---|---|
| Llama 3.2 3B | Small (3 billion parameters) | Fast experiments, testing your pipeline |
| Llama 3.1 8B | Medium | Good balance of quality and speed |
| Llama 3.1 70B | Large | High quality, needs more memory |
| Llama 3.1 405B | Huge | Best quality, needs both DGX Sparks |

**Start small.** Get your pipeline working with the 3B model first. Once everything works, scale up.

### Step 3: Train

The training process:

1. The AI reads one of your examples
2. It tries to predict the correct response
3. It compares its prediction to your actual answer
4. It adjusts its internal weights to be a little more correct
5. Repeat thousands of times

```
 ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
 │  Your Data   │ ──► │  Training     │ ──► │  Your Model  │
 │  (examples)  │     │  (learning)   │     │  (improved)  │
 └─────────────┘     └──────────────┘     └─────────────┘
       500+              minutes to           ready to
     examples             hours                use!
```

### Step 4: Use Your Model

Your trained model works just like any other AI model — you can chat with it, build apps with it, or serve it as an API.

---

## Key Concepts (Plain English)

### LoRA (Low-Rank Adaptation)

Instead of changing *all* the model's parameters (billions of numbers), LoRA only changes a small fraction — usually 1-2%. This is:
- **Faster** — minutes instead of hours
- **Uses less memory** — you can fine-tune big models on smaller hardware
- **Saves storage** — the LoRA adapter is only ~50-200 MB, not the full model size

Think of it like teaching someone a new skill without rewiring their entire brain. You're adding a small "plugin" of new knowledge.

### Quantization (4-bit / 8-bit)

Models are normally stored in high precision (16-bit or 32-bit numbers). Quantization shrinks them by using fewer bits per number:

| Precision | Memory for 70B Model | Quality Loss |
|---|---|---|
| 32-bit (full) | ~280 GB | None |
| 16-bit (bf16) | ~140 GB | Negligible |
| 8-bit | ~70 GB | Very small |
| **4-bit** | **~35 GB** | Small but acceptable |

4-bit quantization lets you fit models on your DGX Spark that would otherwise be too large. The quality loss is surprisingly small for most tasks.

### Epochs

One **epoch** = one complete pass through all your training data. Training usually runs for 1-5 epochs.

- **1 epoch:** The model has seen every example once. May underfit (not learned enough).
- **3 epochs:** The model has seen every example three times. Usually the sweet spot.
- **10+ epochs:** The model has memorized your data. May **overfit** (too specific, can't generalize).

### Batch Size and Gradient Accumulation

The model doesn't learn from one example at a time — it learns from a **batch** (group) of examples at once. Bigger batches = smoother learning, but use more memory.

If your batch doesn't fit in memory, **gradient accumulation** simulates a bigger batch by processing small chunks and adding them up:

```
Batch size: 2, Gradient accumulation: 8
→ Effective batch size: 2 × 8 = 16
→ Same learning quality, fits in memory
```

### Learning Rate

How big of a step the model takes when adjusting its weights. Too high = unstable, too low = slow.

- **2e-4** (0.0002) is a good starting point for LoRA fine-tuning
- If training loss jumps around wildly → lower the learning rate
- If training loss barely moves → raise the learning rate

---

## Single Node vs. Two Nodes

### One DGX Spark (Standalone)

```
 ┌─────────────────────────┐
 │  DGX Spark               │
 │  128 GB unified memory   │
 │                          │
 │  Your training script    │
 │  runs right here         │
 └─────────────────────────┘
```

- **Simpler:** No network config, no coordination
- **Fits most tasks:** Models up to ~70B at 4-bit quantization
- **Start here:** Get your script working before going distributed

See [Standalone Training Guide](08-standalone-training.md) for step-by-step instructions.

### Two DGX Sparks (Distributed)

```
 ┌─────────────────────────┐          ┌─────────────────────────┐
 │  DGX Spark 1             │          │  DGX Spark 2             │
 │  Processes half the data │ 200 Gb/s │  Processes other half    │
 │                          │ ◄──────► │                          │
 │  "Hey, here's what I     │  fabric  │  "Cool, here's what I   │
 │   learned this batch"    │  link    │   learned this batch"    │
 └─────────────────────────┘          └─────────────────────────┘
```

- **Faster:** ~1.87x speedup (almost twice as fast)
- **More memory:** 256 GB total — fits models up to 405B
- **How it works:** Each Spark processes half the data, then they compare notes over the high-speed link

See [Distributed Training Guide](02-distributed-training.md) for the full setup.

---

## Use Cases and Examples

### Use Case 1: Customer Service Bot

**Goal:** Train a model that answers questions about your specific products and policies.

**Data:** Export your support ticket history — questions and the best answers your team gave.

```json
{"instruction": "Customer asks: Do you offer same-day service?", "response": "Yes, we offer same-day service for emergency repairs within our coverage area. Standard appointments are typically scheduled within 48 hours."}
```

**Model:** Llama 3.2 3B (fast, good enough for FAQ-style responses)

### Use Case 2: Legal Document Drafting

**Goal:** Train a model to draft legal documents in your firm's style and format.

**Data:** Redacted examples of demand letters, intake summaries, case evaluations.

**Model:** Llama 3.1 8B (better language quality for professional documents)

### Use Case 3: Sales Call Analysis

**Goal:** Train a model to analyze sales call transcripts and score them.

**Data:** Transcripts with your scoring criteria and example analyses.

**Model:** Llama 3.1 8B or 70B (complex reasoning benefits from larger models)

### Use Case 4: Internal Knowledge Base

**Goal:** Train a model on your company's internal documentation, SOPs, and runbooks.

**Data:** Export your wiki, Confluence, or SharePoint docs as instruction-response pairs.

**Model:** Llama 3.1 8B (good balance of knowledge capacity and speed)

---

## Your First Training Run (Step by Step)

Here's exactly what to do, in order:

### 1. SSH into Your DGX Spark

```bash
ssh sem@100.122.26.9   # spark-dgx-1 via Tailscale
# or
ssh sem@192.168.0.188  # spark-dgx-1 via LAN
```

### 2. Check That Everything Is Working

```bash
nvidia-smi                    # Should show your GPU
python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### 3. Install Training Libraries (One Time)

```bash
pip3 install transformers datasets peft trl accelerate bitsandbytes
```

### 4. Create a Test Dataset

```bash
cat > test_data.jsonl << 'EOF'
{"text": "### Instruction:\nWhat is 2+2?\n\n### Response:\nThe answer is 4."}
{"text": "### Instruction:\nWhat color is the sky?\n\n### Response:\nThe sky is blue on a clear day."}
{"text": "### Instruction:\nName three fruits.\n\n### Response:\nApple, banana, and orange."}
EOF
```

### 5. Run Training

```bash
python3 -c "
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('json', data_files='test_data.jsonl', split='train')

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'),
    args=SFTConfig(output_dir='./test-model', num_train_epochs=3, per_device_train_batch_size=1,
                   logging_steps=1, max_seq_length=512, dataset_text_field='text'),
    tokenizer=tokenizer,
)
trainer.train()
print('Done! Your first model is trained.')
"
```

If you see "Done! Your first model is trained." — congratulations, you just fine-tuned an AI model on your own hardware.

---

## Basic Troubleshooting

### "CUDA out of memory"

Your model + data doesn't fit in the GPU's 128 GB.

**Fixes:**
- Reduce `batch_size` (try 1)
- Increase `gradient_accumulation` to compensate
- Use 4-bit quantization (`load_in_4bit=True`)
- Use a smaller model (3B instead of 8B)
- Reduce `max_seq_length` (2048 → 1024)

### "No module named 'transformers'"

Training libraries aren't installed.

```bash
pip3 install transformers datasets peft trl accelerate bitsandbytes
```

### "torch.cuda.is_available() returns False"

PyTorch can't see the GPU.

```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA
pip3 install torch --index-url https://download.pytorch.org/whl/cu124
```

### Training loss isn't going down

- **Learning rate too low:** Try `5e-4` instead of `2e-4`
- **Too few examples:** Need at least 100-200 diverse examples
- **Bad data format:** Make sure your examples are properly formatted
- **Too few epochs:** Try 5 epochs instead of 3

### Training loss goes to 0 (too perfect)

- **Overfitting:** Model memorized your data instead of learning patterns
- **Fix:** Reduce epochs (3 instead of 10), add more diverse data, increase `lora_dropout`

### "Connection refused" (distributed training only)

Both nodes need to be running within 60 seconds of each other. See [Distributed Training Troubleshooting](02-distributed-training.md#troubleshooting).

---

## What's Next?

Once you're comfortable with the basics:

1. **[Standalone Training](08-standalone-training.md)** — Full single-node setup with launch scripts and ROSA storage
2. **[Distributed Training](02-distributed-training.md)** — Scale to two DGX Sparks for bigger models and faster training
3. **[Training Pipeline](05-training-pipeline.md)** — End-to-end workflow: data prep → train → merge LoRA → evaluate
4. **[EXO Inference](07-exo-cuda-tinygrad.md)** — Serve your trained model as an API using EXO

---

## Glossary

| Term | What It Means |
|---|---|
| **Fine-tuning** | Teaching an existing model new skills using your data |
| **LoRA** | A technique that trains only 1-2% of the model, saving time and memory |
| **Quantization** | Compressing a model to use less memory (4-bit, 8-bit) |
| **Epoch** | One complete pass through all training data |
| **Batch size** | How many examples the model processes at once |
| **Gradient accumulation** | Simulating a bigger batch by processing small chunks |
| **Learning rate** | How aggressively the model adjusts during training |
| **Overfitting** | When the model memorizes your data instead of learning patterns |
| **DDP** | Distributed Data Parallel — splitting training across multiple GPUs |
| **NCCL** | NVIDIA's library for GPU-to-GPU communication |
| **bfloat16** | A number format that's efficient on modern GPUs |
| **Inference** | Using a trained model to generate responses (the opposite of training) |
