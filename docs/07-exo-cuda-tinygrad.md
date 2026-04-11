# EXO on Linux with CUDA (Tinygrad Backend)

**Platform:** Linux (NVIDIA GPU) — DGX Spark (Grace Blackwell GB10)
**Status: Working** — GPU-accelerated distributed inference across both DGX Spark nodes using EXO v0.0.9-alpha with the tinygrad backend on CUDA.

> **Platform guide:** EXO supports different backends per platform:
>
> | Platform | Backend | EXO Version | Guide |
> |---|---|---|---|
> | **macOS (Apple Silicon)** | MLX | v0.3.69+ (main branch) | [06-exo-model-serving.md](06-exo-model-serving.md) |
> | **Linux (NVIDIA CUDA)** | Tinygrad | v0.0.9-alpha | **This document** |
> | **Linux (CPU only)** | Tinygrad (CPU) | v0.0.9-alpha | [06-exo-model-serving.md](06-exo-model-serving.md) |
>
> The main EXO branch (v0.3.69+) uses MLX, which only works on Apple Silicon. For Linux with NVIDIA GPUs, use v0.0.9-alpha which defaults to the tinygrad inference engine with full CUDA support.

---

## Architecture

```
 ┌──────────────────────────┐              ┌──────────────────────────┐
 │  spark-dgx-1             │              │  spark-dgx-2             │
 │  GB10 · 128 GB unified   │              │  GB10 · 128 GB unified   │
 │                          │              │                          │
 │  EXO v0.0.9-alpha        │  10.0.0.x    │  EXO v0.0.9-alpha        │
 │  tinygrad 0.10.0 (CUDA)  │  ─────────►  │  tinygrad 0.10.0 (CUDA)  │
 │  API port: 52415         │  200 Gb/s    │  API port: 52416         │
 │  Coordinator             │  fabric      │  Worker                  │
 └──────────────────────────┘              └──────────────────────────┘
          │
          ▼
    http://10.0.0.1:52415
    TinyChat Web UI + OpenAI-compatible API
    40+ models available
```

Both nodes share model weights from ROSA NVMe-TCP storage — download once, both nodes read.

---

## Why Tinygrad, Not MLX

| | MLX | Tinygrad |
|---|---|---|
| Platform | Apple Silicon only | CUDA, ROCm, Metal, CPU |
| Quantization | QMM not implemented on CUDA (`RuntimeError: QMM NYI`) | Full CUDA quantization support (4-bit, 8-bit) |
| EXO version | v0.3.69+ (current main branch) | v0.0.9-alpha (tagged release) |
| DGX Spark | Fails on quantized models | Works |

The main EXO branch defaults to MLX on all platforms. On Linux with NVIDIA GPUs, v0.0.9-alpha correctly defaults to the tinygrad inference engine.

---

## Setup

### Step 1 — Create a Tinygrad Venv (Both Nodes)

EXO v0.0.9-alpha requires its own virtual environment separate from any existing EXO or PyTorch installs.

```bash
# On BOTH spark-dgx-1 and spark-dgx-2

python3 -m venv /home/sem/exo-tinygrad-venv
source /home/sem/exo-tinygrad-venv/bin/activate

# Install EXO v0.0.9-alpha (includes tinygrad 0.10.0)
pip install exo-explore==0.0.9a0

# Verify tinygrad sees the GPU
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
# Should print: CUDA
```

### Step 2 — Patch device_capabilities.py (Both Nodes)

The DGX Spark's unified memory architecture causes `pynvml.nvmlDeviceGetMemoryInfo()` to throw `NVMLError_NotSupported`. This patch falls back to system RAM reporting.

Find the file:
```bash
CAPS_FILE=$(find /home/sem/exo-tinygrad-venv -name "device_capabilities.py" -path "*/topology/*")
echo $CAPS_FILE
```

Apply the patch — replace the `nvmlDeviceGetMemoryInfo` call (~line 195):

```python
# BEFORE:
gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
gpu_memory_mb = gpu_memory_info.total // 2**20

# AFTER:
try:
    gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_memory_mb = gpu_memory_info.total // 2**20
except pynvml.NVMLError:
    import psutil
    gpu_memory_mb = psutil.virtual_memory().total // 2**20
```

This is required on both nodes. Without it, EXO crashes on startup when probing GPU memory.

### Step 3 — Shared Model Storage via ROSA NVMe-TCP

By default, each EXO node downloads models to `~/.local/share/exo/models/` — meaning every model gets downloaded twice (once per node) and eats local disk. The solution: point both nodes at a shared directory on the ROSA NVMe-TCP volume.

#### How it works

```
 ┌─────────────┐     NVMe-TCP      ┌──────────────────────────────┐
 │ spark-dgx-1 │────────────────►  │  MikroTik RDS2216            │
 │ reads from   │    192.168.0.100  │  ROSA Data Server            │
 │ /mnt/rosa-   │                   │                              │
 │  models/exo  │                   │  Subsystem: llm-models       │
 └─────────────┘                   │  /mnt/rosa-models/exo/       │
                                    │                              │
 ┌─────────────┐     NVMe-TCP      │  ┌────────────────────────┐  │
 │ spark-dgx-2 │────────────────►  │  │ llama-3.2-3b/          │  │
 │ reads from   │    192.168.0.100  │  │ llama-3.1-70b-4bit/    │  │
 │ /mnt/rosa-   │                   │  │ qwen-2.5-72b/          │  │
 │  models/exo  │                   │  │ deepseek-v3/           │  │
 └─────────────┘                   │  └────────────────────────┘  │
                                    └──────────────────────────────┘
```

Both DGX Spark nodes connect to the ROSA Data Server (RDS2216) over NVMe-TCP. The `llm-models` subsystem is mounted at `/mnt/rosa-models` on both nodes. This mount is handled automatically at boot by the `nvme-rosa-connect.service` systemd unit (see [ROSA Initiator Setup](rosa/02-spark-initiator-setup.md)).

#### Configure EXO to use ROSA

```bash
# Add to ~/.bashrc on BOTH nodes
export EXO_DEFAULT_MODELS_DIR=/mnt/rosa-models/exo
```

Or pass it as a flag at launch time:

```bash
exo --models-seed-dir /mnt/rosa-models/exo
```

#### The result

- **Download once:** When EXO on spark-dgx-1 downloads a model, spark-dgx-2 sees it immediately
- **No local disk usage:** Models live on the ROSA's 20x U.2 NVMe array, not on local storage
- **NVMe-TCP performance:** The ROSA volume is mounted as a block device over TCP — reads perform at near-local NVMe speeds (tested at ~3.2 GB/s sequential read)
- **Survives reboots:** The `nvme-rosa-connect.service` auto-connects and `fstab` auto-mounts on every boot

#### Verify the mount

```bash
# Check ROSA volumes are mounted
df -h | grep rosa
# Should show: /dev/nvmeXnX  ... /mnt/rosa-models

# Check EXO models directory exists
ls /mnt/rosa-models/exo/
```

> **Prerequisite:** ROSA NVMe-TCP initiator must be configured on both nodes. See [ROSA Initiator Setup](rosa/02-spark-initiator-setup.md) for the full setup including kernel module, discovery, connect, fstab, and systemd auto-connect.

### Step 4 — Launch the Cluster

**On spark-dgx-1 (coordinator):**

```bash
TERM=xterm /home/sem/exo-tinygrad-venv/bin/exo \
    --inference-engine tinygrad \
    --chatgpt-api-port 52415 \
    --disable-tui \
    --models-seed-dir /mnt/rosa-models/exo \
    --node-host 10.0.0.1 \
    --broadcast-port 5678
```

**On spark-dgx-2 (worker):**

```bash
TERM=xterm /home/sem/exo-tinygrad-venv/bin/exo \
    --inference-engine tinygrad \
    --chatgpt-api-port 52416 \
    --disable-tui \
    --models-seed-dir /mnt/rosa-models/exo \
    --node-host 10.0.0.2 \
    --broadcast-port 5678
```

Both nodes discover each other via UDP broadcast on port 5678 over the 10.0.0.x fabric link.

### Flags Explained

| Flag | Purpose |
|---|---|
| `--inference-engine tinygrad` | Use tinygrad CUDA backend instead of MLX |
| `--chatgpt-api-port` | API/TinyChat port (52415 on node 1, 52416 on node 2) |
| `--disable-tui` | No terminal UI — required when running headless or via SSH |
| `--models-seed-dir` | Path to shared model storage on ROSA NVMe |
| `--node-host` | Bind to fabric IP for inter-node communication |
| `--broadcast-port 5678` | UDP discovery port (must match on both nodes) |
| `TERM=xterm` | Prevents terminal capability errors over SSH |

---

## Verify It Works

### Check the API

```bash
# List available models (40+ supported)
curl http://10.0.0.1:52415/v1/models | python3 -m json.tool

# Run inference
curl http://10.0.0.1:52415/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-3.2-3b",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 64
    }'
```

### TinyChat Web UI

Open in a browser:
- **From LAN:** `http://192.168.0.188:52415`
- **From fabric:** `http://10.0.0.1:52415`
- **Via Tailscale:** `http://100.122.26.9:52415`

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://10.0.0.1:52415/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="llama-3.2-3b",
    messages=[{"role": "user", "content": "Explain RDMA in one paragraph."}],
)
print(response.choices[0].message.content)
```

---

## Systemd Service (Production)

To run as persistent services that start on boot:

**On spark-dgx-1:**

```bash
sudo tee /etc/systemd/system/exo-cuda.service << 'EOF'
[Unit]
Description=EXO Distributed Inference (CUDA/Tinygrad)
After=network-online.target nvme-rosa-connect.service
Wants=network-online.target

[Service]
Type=simple
User=sem
Environment="TERM=xterm"
Environment="EXO_DEFAULT_MODELS_DIR=/mnt/rosa-models/exo"
ExecStart=/home/sem/exo-tinygrad-venv/bin/exo \
    --inference-engine tinygrad \
    --chatgpt-api-port 52415 \
    --disable-tui \
    --models-seed-dir /mnt/rosa-models/exo \
    --node-host 10.0.0.1 \
    --broadcast-port 5678
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

**On spark-dgx-2:** Same file but change `--chatgpt-api-port 52416` and `--node-host 10.0.0.2`.

```bash
sudo systemctl daemon-reload
sudo systemctl enable exo-cuda
sudo systemctl start exo-cuda
```

---

## Troubleshooting

### `RuntimeError: QMM NYI`

You're running a quantized model on the MLX backend. Switch to tinygrad:
- Use `--inference-engine tinygrad`
- Make sure you're running EXO v0.0.9-alpha, not v0.3.69+

### `NVMLError_NotSupported` on startup

The `device_capabilities.py` patch hasn't been applied. See [Step 2](#step-2--patch-device_capabilitiespy-both-nodes).

### Nodes don't discover each other

1. Verify both nodes use the same `--broadcast-port`
2. Verify `--node-host` is set to each node's fabric IP (10.0.0.1 / 10.0.0.2)
3. Check that UDP broadcast works: `ping -b 10.0.0.255`
4. Kill any stale EXO processes: `pkill -9 -f exo`

### Port already in use

```bash
# Find and kill whatever holds the port
fuser -k 52415/tcp
# Or kill all EXO processes
pkill -9 -f exo
```

### Slow inter-node transfer / poor bandwidth

If you're using a MikroTik switch (CRS812, etc.) or any managed switch, the switch ports may default to a lower MTU (1500). You must set **MTU 9000 (jumbo frames)** on both the switch ports AND the DGX Spark NIC interfaces. Mismatched MTU causes silent packet fragmentation or drops, significantly degrading throughput.

**On MikroTik RouterOS:**
```
/interface ethernet set [find name=sfp-sfpplus1] mtu=9000
/interface ethernet set [find name=sfp-sfpplus2] mtu=9000
```

**On each DGX Spark node:** Already handled by the netplan configs in `configs/netplan-node0.yaml` / `configs/netplan-node1.yaml` (both set `mtu: 9000`).

Verify end-to-end MTU from one node:
```bash
ping -M do -s 8972 10.0.0.2
# If this fails, the MTU is not 9000 somewhere in the path
```

### `tinygrad` uses CPU instead of CUDA

```bash
# Check CUDA visibility
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
# If it prints CPU, CUDA toolkit may not be in PATH:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## Available Models

EXO v0.0.9-alpha includes 40+ models in its catalog. Some highlights:

| Model | Size | Fits on 1 Node? |
|---|---|---|
| Llama 3.2 1B / 3B | ~2-6 GB | Yes |
| Llama 3.1 8B | ~16 GB | Yes |
| Llama 3.1 70B (4-bit) | ~35 GB | Yes |
| Llama 3.1 405B (4-bit) | ~200 GB | **No — requires both nodes** |
| Qwen 2.5 72B | ~40 GB | Yes |
| DeepSeek V3 (671B MoE) | ~350 GB | **No — requires both nodes** |
| Mistral, Gemma, etc. | Varies | Check catalog |

Models >128 GB are automatically sharded across both nodes via pipeline parallelism.

---

## References

- [EXO GitHub](https://github.com/exo-explore/exo) — Main repo (v0.3.69+, MLX-focused)
- [EXO v0.0.9-alpha Release](https://github.com/exo-explore/exo/releases/tag/v0.0.9-alpha) — The release with working tinygrad/CUDA support
- [Tinygrad](https://github.com/tinygrad/tinygrad) — CUDA inference engine used by EXO
- [CPU/MLX Setup](06-exo-model-serving.md) — Alternative: CPU inference or Apple Silicon MLX
