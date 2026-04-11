# Tested Model Catalog

**Which models work on DGX Spark, how much memory they need, and what performance to expect.**

This catalog covers models tested on our 2-node DGX Spark cluster (256 GB total unified memory, tinygrad CUDA backend via EXO v0.0.9-alpha).

---

## Quick Reference

### Models That Fit on One Node (≤128 GB)

| Model | Quantization | Memory | Inference | Status |
|---|---|---|---|---|
| Llama 3.2 1B Instruct | bf16 | ~2 GB | Fast | Tested |
| Llama 3.2 3B Instruct | bf16 | ~6 GB | Fast | Tested |
| Llama 3.1 8B Instruct | bf16 | ~16 GB | Fast | Tested |
| Llama 3.1 8B Instruct | 4-bit | ~5 GB | Fast | Tested |
| Llama 3.1 70B Instruct | 4-bit | ~35 GB | Good | Tested |
| Qwen 2.5 7B Instruct | bf16 | ~14 GB | Fast | Tested |
| Qwen 2.5 72B Instruct | 4-bit | ~40 GB | Good | Tested |
| Mistral 7B Instruct | bf16 | ~14 GB | Fast | Tested |
| Gemma 2 9B | bf16 | ~18 GB | Fast | Tested |
| DeepSeek R1 Distill 7B | bf16 | ~14 GB | Fast | Tested |

### Models That Require Both Nodes (>128 GB)

| Model | Quantization | Memory | Nodes | Status |
|---|---|---|---|---|
| Llama 3.1 405B | 4-bit | ~200 GB | 2 | Requires sharding |
| Llama 3.1 405B | 8-bit | ~400 GB | 2 (tight) | May not fit |
| DeepSeek V3 (671B MoE) | 4-bit | ~350 GB | 2 (tight) | Experimental |
| Mixtral 8x22B | bf16 | ~280 GB | 2 | Requires sharding |

---

## EXO Model Catalog

EXO v0.0.9-alpha includes 40+ models in its built-in catalog. Query them:

```bash
curl http://10.0.0.1:52415/v1/models | python3 -m json.tool
```

### Available Model Families

| Family | Sizes Available | Notes |
|---|---|---|
| **Llama 3.x** | 1B, 3B, 8B, 70B, 405B | Best general-purpose. 405B needs both nodes |
| **Qwen 2.5** | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | Strong multilingual and coding |
| **Mistral** | 7B, Nemo 12B | Good instruction following |
| **Gemma 2** | 2B, 9B, 27B | Google's open models |
| **DeepSeek** | V2, V2.5, V3, R1 distills | MoE architecture, strong reasoning |
| **Phi** | 3.5 Mini | Microsoft, good for small tasks |

---

## For Training (Fine-Tuning)

Different models have different memory requirements during training vs. inference. Training uses more memory due to gradients, optimizer states, and activations.

### LoRA Fine-Tuning Memory (Single Node, 4-bit Base)

| Model | Base Memory | + LoRA (r=32) | + Batch=2 | Total |
|---|---|---|---|---|
| Llama 3.2 3B | ~4 GB | +0.1 GB | +2 GB | ~6 GB |
| Llama 3.1 8B | ~5 GB | +0.2 GB | +4 GB | ~10 GB |
| Llama 3.1 70B | ~35 GB | +0.5 GB | +8 GB | ~44 GB |
| Qwen 2.5 72B | ~40 GB | +0.5 GB | +8 GB | ~49 GB |

All of these fit comfortably on a single DGX Spark (128 GB).

### Full Fine-Tuning Memory (bf16, Single Node)

| Model | Memory Needed | Fits on 1 Node? |
|---|---|---|
| Llama 3.2 3B | ~24 GB | Yes |
| Llama 3.1 8B | ~64 GB | Yes |
| Llama 3.1 70B | ~560 GB | No (use LoRA or distributed) |

> **Recommendation:** Always use LoRA + 4-bit quantization unless you have a specific reason for full fine-tuning. The quality difference is minimal for most tasks.

---

## Model Download Locations

Models are stored on ROSA shared storage so both nodes can access them:

```
/mnt/rosa-models/
├── exo/                          # EXO inference models (auto-downloaded)
│   ├── llama-3.2-3b/
│   ├── llama-3.1-70b-4bit/
│   └── ...
└── training/                     # Fine-tuned models
    ├── my-finetune-v1/
    │   ├── final/                # LoRA adapter
    │   └── merged/               # Merged full model
    └── my-finetune-v2/
```

### Downloading Models Manually

```bash
# Via Hugging Face CLI
pip install huggingface-hub
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir /mnt/rosa-models/llama-3.1-8b

# Via EXO (auto-downloads on first use)
curl http://10.0.0.1:52415/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "hello"}]}'
```

---

## Choosing the Right Model

```
Start here:
│
├── Just experimenting?
│   └── Llama 3.2 3B — fast, small, good for testing pipelines
│
├── General assistant / chatbot?
│   └── Llama 3.1 8B — best quality-to-speed ratio
│
├── Coding / technical tasks?
│   └── Qwen 2.5 72B — excellent at code generation
│
├── Complex reasoning?
│   └── DeepSeek R1 or Llama 3.1 70B
│
├── Need the best quality possible?
│   └── Llama 3.1 405B (requires both DGX Sparks)
│
└── Multilingual?
    └── Qwen 2.5 (any size) — strong across languages
```

---

## Contributing to This Catalog

Tested a model on DGX Spark? Add it here:

1. Record: model name, quantization, memory usage, tok/s (if measured)
2. Note any issues (crashes, slow performance, compatibility problems)
3. Submit a PR or open an issue
