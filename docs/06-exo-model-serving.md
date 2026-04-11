# Serving Large Models Across Both DGX Sparks with Exo

**Goal:** Run models too large for a single DGX Spark (>128 GB) by splitting them across both nodes using [Exo](https://github.com/exo-explore/exo).

**Example:** Llama 405B at 4-bit quantization requires ~200 GB — too large for one Spark's 128 GB unified memory, but fits comfortably across two (256 GB total).

> **Looking for CUDA/GPU inference?** See [EXO on Linux with CUDA (Tinygrad Backend)](07-exo-cuda-tinygrad.md) — the tested, working GPU-accelerated setup using EXO v0.0.9-alpha + tinygrad.

---

## How Exo Works

Exo uses **pipeline parallelism** to shard a model across multiple devices. Each node holds a slice of the model layers, and inference requests flow through the pipeline. Nodes discover each other automatically via libp2p — no manual topology config needed.

```
 ┌──────────────────────────┐          ┌──────────────────────────┐
 │  spark-dgx-1             │          │  spark-dgx-2             │
 │  Layers 0-39             │  ──────► │  Layers 40-79            │
 │  ~100 GB                 │ 200 Gb/s │  ~100 GB                 │
 │  Coordinator + API       │  fabric  │  Worker                  │
 └──────────────────────────┘          └──────────────────────────┘
          │
          ▼
    http://192.168.0.188:52415
    OpenAI-compatible API
```

---

## Current Exo Status on DGX Spark

| Feature | Status | Notes |
|---|---|---|
| Auto-discovery | Works | libp2p over LAN or fabric (10.0.0.x) |
| CPU inference (ARM64) | Works | Grace CPU, ~5-10 tok/s on large models |
| NVIDIA GPU inference (CUDA) | **Working** | EXO v0.0.9-alpha + tinygrad 0.10.0. See [CUDA/Tinygrad Setup](07-exo-cuda-tinygrad.md) |
| MLX backend | N/A | Apple Silicon only |
| Model sharding | Works | Pipeline parallel across nodes |
| OpenAI API compat | Works | Chat completions at port 52415 |

> **Bottom line:** GPU-accelerated inference is working on DGX Spark using EXO v0.0.9-alpha with the tinygrad CUDA backend. See [07-exo-cuda-tinygrad.md](07-exo-cuda-tinygrad.md) for the full setup guide. This doc covers the CPU/general setup path.

---

## Setup

### Step 1 — Install Exo on Both Nodes

```bash
# On BOTH spark-dgx-1 and spark-dgx-2

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install Node.js 18+ (for dashboard)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Rust nightly (for tokenizer compilation)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y
source "$HOME/.cargo/env"

# Clone exo
cd /opt
sudo git clone https://github.com/exo-explore/exo.git
sudo chown -R sem:sem exo
cd exo

# Build dashboard
cd exo/tinychat
npm install && npm run build
cd ../..
```

### Step 2 — Configure Shared Model Storage

Point Exo at the ROSE NVMe-TCP storage so models are downloaded once and shared:

```bash
# On BOTH nodes — add to ~/.bashrc
export EXO_DEFAULT_MODELS_DIR=/mnt/rosa-models/exo
export EXO_MODELS_READ_ONLY_DIRS=/mnt/rosa-models/exo

# Create the directory (on one node — it's shared storage)
mkdir -p /mnt/rosa-models/exo
```

### Step 3 — Isolate the Cluster (Optional)

If you have other Exo nodes on the network (Mac fleet, etc.), isolate the DGX cluster:

```bash
# On BOTH nodes
export EXO_LIBP2P_NAMESPACE=dgx-spark-cluster
```

### Step 4 — Start Exo on Both Nodes

**On spark-dgx-1 (coordinator):**

```bash
cd /opt/exo
uv run exo
```

**On spark-dgx-2 (worker):**

```bash
cd /opt/exo
uv run exo
```

Both nodes will auto-discover each other. spark-dgx-1 serves the API at `http://192.168.0.188:52415`.

### Step 5 — Verify Cluster

Open the dashboard at `http://192.168.0.188:52415` or check via API:

```bash
# From any machine on the LAN
curl http://192.168.0.188:52415/v1/models
```

---

## Running Large Models

### Llama 405B (4-bit)

```bash
# Download will go to /mnt/rosa-models/exo (shared NVMe-TCP storage)
# Exo auto-shards across both nodes

curl http://192.168.0.188:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-405b-4bit",
    "messages": [{"role": "user", "content": "Explain RDMA in one paragraph."}],
    "max_tokens": 256
  }'
```

### DeepSeek V3 (671B MoE)

```bash
curl http://192.168.0.188:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-0324",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }'
```

### Using with Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.188:52415/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="llama-3.1-405b-4bit",
    messages=[{"role": "user", "content": "What is NVMe-TCP?"}],
)
print(response.choices[0].message.content)
```

---

## GPU-Accelerated Inference (CUDA/Tinygrad)

GPU inference is working on DGX Spark. For the complete setup guide, see:

**[07-exo-cuda-tinygrad.md](07-exo-cuda-tinygrad.md)** — EXO v0.0.9-alpha + tinygrad 0.10.0, venv setup, device_capabilities.py patch, launch commands, systemd service, and troubleshooting.

---

## Performance Expectations

| Model | Size (4-bit) | Nodes | Mode | Expected tok/s |
|---|---|---|---|---|
| Llama-3.2-3B | ~2 GB | 1 | CPU | ~30-50 |
| Llama-3.1-70B | ~35 GB | 1 | CPU | ~5-10 |
| Llama-3.1-405B | ~200 GB | 2 | CPU | ~2-5 |
| Llama-3.1-405B | ~200 GB | 2 | GPU (if tinygrad works) | ~15-30 |

> **Note:** For single-request latency, one node is faster than two (no pipeline overhead). Use two nodes only when the model doesn't fit in one node's 128 GB, or to increase multi-request throughput.

---

## Alternative: vLLM for GPU-Native Serving

If Exo's GPU support doesn't work on Blackwell, [vLLM](https://github.com/vllm-project/vllm) is the production alternative for NVIDIA GPUs with tensor parallel:

```bash
pip install vllm

# Single node (models ≤128 GB)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --dtype bfloat16 --quantization awq \
    --host 0.0.0.0 --port 8000

# Multi-node tensor parallel (models >128 GB)
# Requires Ray cluster setup between both Sparks
ray start --head  # on spark-dgx-1
ray start --address=10.0.0.1:6379  # on spark-dgx-2

vllm serve meta-llama/Llama-3.1-405B-Instruct-AWQ \
    --dtype bfloat16 --quantization awq \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 --port 8000
```

---

## Systemd Service (Production)

To run Exo as a persistent service:

```bash
sudo tee /etc/systemd/system/exo.service << 'EOF'
[Unit]
Description=Exo Distributed AI Inference
After=network-online.target nvme-rosa-connect.service
Wants=network-online.target

[Service]
Type=simple
User=sem
WorkingDirectory=/opt/exo
Environment="EXO_DEFAULT_MODELS_DIR=/mnt/rosa-models/exo"
Environment="EXO_LIBP2P_NAMESPACE=dgx-spark-cluster"
ExecStart=/home/sem/.local/bin/uv run exo
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable exo
sudo systemctl start exo
```

Install on **both nodes**. They'll auto-discover each other on boot.

---

## References

- [Exo GitHub](https://github.com/exo-explore/exo)
- [Exo CUDA Fork (tinygrad)](https://github.com/Scottcjn/exo-cuda)
- [vLLM (alternative)](https://github.com/vllm-project/vllm)
- [Exo NVIDIA GPU discussion](https://github.com/exo-explore/exo/issues/1039)
