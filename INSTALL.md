# Setup Guide — Getting This Repo onto GitHub

## Step 1: Pull the tarball to your Mac

```bash
scp sem@100.122.26.9:/home/sem/dgx-spark-cluster-full.tar.gz ~/Downloads/
cd ~/Downloads
tar -xzf dgx-spark-cluster-full.tar.gz
cd dgx-spark-direct-fabric
```

## Step 2: Push to GitHub

```bash
git branch -m master main          # rename branch if needed
git remote add origin https://github.com/ArgentAIOS/dgx-spark-cluster.git
git push -u origin main
```

## Step 3: Add the ROSA folder

```bash
# From inside the repo:
tar -xzf ~/Downloads/rosa-nvme-setup.tar.gz
git add rosa-nvme-setup/
git commit -m "Add ROSA NVMe-TCP storage setup (MikroTik RDS2216)

Co-Authored-By: Oz <oz-agent@warp.dev>"
git push
```

---

# LLM Context — What This Cluster Is

> Copy everything below this line and paste it as system context into your local LLM
> (Ollama, LM Studio, etc.) to give it full awareness of your setup.

---

## SYSTEM CONTEXT: ArgentAI DGX Spark Cluster

You are assisting with a 2-node NVIDIA DGX Spark AI training cluster.
Below is the complete reference for the hardware, network, storage, and software stack.

### Hardware

- **2× NVIDIA DGX Spark** — each has an NVIDIA GB10 Grace Blackwell Superchip
  - ARM64 (aarch64) CPU + Blackwell GPU in a unified memory architecture
  - OS: DGX Spark OS 7.5.0, Ubuntu 22.04 LTS, kernel 6.17.0-1014-nvidia
  - NVIDIA open driver 580, CUDA 13.x, NCCL 2.27.7, MLNX OFED 24.07
- **MikroTik RDS2216 ("ROSA")** — 20× U.2 NVMe storage server, RouterOS
- **Dell PowerEdge R750** — shared NFS storage, 50G bonded network
- **TrueNAS** — archive storage, 10G
- **MikroTik CRS812** — 10G management switch

### Network

| Link | Interfaces | IPs | Speed | Use |
|---|---|---|---|---|
| Direct fabric | enp1s0f0np0 / mlx5_0 | 10.0.0.1 ↔ 10.0.0.2 | 200 Gb/s | NCCL gradient sync |
| LAN | enp1s0f1np1 / mlx5_1 | 192.168.0.188 / .218 | 200 Gb/s | Storage, internet |
| Management | enP7s7 | 192.168.0.110 | 10 Gb/s | Out-of-band |
| Remote | tailscale0 | 100.122.26.9 | — | SSH from Mac |

- spark-dgx-1: master, rank 0, 10.0.0.1, 192.168.0.188
- spark-dgx-2: worker, rank 1, 10.0.0.2, 192.168.0.218
- ROSA storage: 192.168.0.100
- Dell NFS: 192.168.0.98
- Gateway: 192.168.0.1

### Storage (mounted on each Spark)

| Mount | Source | Protocol | Speed |
|---|---|---|---|
| /mnt/rosa-storage | ROSA nvme1n1 (raid5-data) | NVMe-TCP | ~2 GB/s |
| /mnt/rosa-models | ROSA nvme3n1 (llm-models) | NVMe-TCP | ~2 GB/s |
| /mnt/dell-shared | Dell 192.168.0.98:/data/shared | NFS v4.2 | ~580 MB/s |
| /mnt/nas | TrueNAS 192.168.0.180 | NFS | ~90 MB/s |
| /home/sem | Local Samsung PM9E1 4TB | PCIe NVMe | 6+ GB/s |

ROSA volumes connect via NVMe-TCP using systemd service `nvme-rosa-connect.service`.
Subsystems: `raid5-data` and `llm-models` on port 4420.

### NCCL / Distributed Training

The direct fabric uses RoCE v2 (Ethernet link layer, not native InfiniBand).
NCCL is configured with two modes:

**Safe mode** (default, always works):
```bash
source configs/nccl-env.sh safe
# NCCL_NET_GDR_LEVEL=0, NCCL_DMABUF_ENABLE=0
# Bandwidth: ~18-20 GB/s
```

**DMA-BUF GPU Direct mode** (max performance):
```bash
source configs/nccl-env.sh dmabuf
# NCCL_NET_GDR_LEVEL=5, NCCL_DMABUF_ENABLE=1
# Bandwidth: ~22-23 GB/s
# Requires: open NVIDIA driver 580+, kernel 6.x (both present on this system)
```

Key NCCL settings (both modes):
- `NCCL_SOCKET_IFNAME=enp1s0f0np0`
- `NCCL_IB_HCA=mlx5_0`
- `NCCL_IB_GID_INDEX=3` (RoCEv2)
- `NCCL_ALGO=Ring`
- `NCCL_IB_TIMEOUT=22`

**Important:** `nvidia-peermem` does NOT work on this hardware (ARM64, kernel 6.x — missing kernel symbols). Use DMA-BUF instead.

### Launch Distributed Training

```bash
# Generic launcher — works with any PyTorch script
# On spark-dgx-1:
./scripts/distributed_train.sh 0 your_script.py [args]

# On spark-dgx-2 (within 60 seconds):
./scripts/distributed_train.sh 1 your_script.py [args]

# Smoke test:
./scripts/distributed_train.sh 0 training/validate_distributed.py
./scripts/distributed_train.sh 1 training/validate_distributed.py
```

### Python Environment

Conda environment: `personal-injury-llm`
```bash
conda activate personal-injury-llm
```
Key packages: torch, transformers, peft, bitsandbytes, trl, datasets, accelerate
See `training/requirements.txt` for full list.

### Repo Structure

```
dgx-spark-cluster/
├── INSTALL.md                    ← This file
├── README.md                     ← Quick start
├── configs/
│   ├── nccl-env.sh               ← NCCL env (safe/dmabuf modes)
│   ├── netplan-node0.yaml        ← spark-dgx-1 network config
│   └── netplan-node1.yaml        ← spark-dgx-2 network config
├── docs/
│   ├── 00-cluster-overview.md   ← Full topology & IP reference
│   ├── 01-network-setup.md      ← Physical cabling & netplan HOW-TO
│   ├── 02-distributed-training.md ← NCCL deep-dive & training guide
│   ├── 03-troubleshooting.md    ← Common issues & fixes
│   └── 04-gpu-direct-rdma.md    ← GPU Direct investigation & DMA-BUF solution
├── scripts/
│   ├── distributed_train.sh     ← Generic 2-node torchrun launcher
│   ├── verify_network_setup.sh  ← Pre-flight check
│   └── launch_local_training.sh ← Single-node launcher
├── training/
│   ├── validate_distributed.py  ← NCCL smoke test (allreduce + bandwidth)
│   ├── benchmark_train.py       ← LoRA fine-tuning benchmark
│   ├── run_benchmark.py         ← Benchmark orchestration
│   ├── configs.yaml             ← Benchmark scenarios
│   ├── test_gpudirect_dmabuf.py ← GPU Direct preflight check
│   └── requirements.txt         ← Python dependencies
└── rosa-nvme-setup/
    ├── README.md
    ├── 01-rosa-server-setup.md  ← MikroTik RDS2216 RouterOS config
    ├── 02-spark-initiator-setup.md ← NVMe-TCP connect on Sparks
    ├── 03-performance-testing.md
    └── configs/
        ├── nvme-rosa-connect.service ← Systemd unit
        └── fstab-entries.txt
```
