# DGX Spark Direct Fabric — 2-Node Cluster Setup & Distributed Training

**Author:** Sem  
**Hardware:** 2× NVIDIA DGX Spark (Grace Blackwell GB10 / NVL32)  
**OS:** DGX Spark OS 7.5.0 — Ubuntu 22.04 LTS (aarch64)  
**Fabric:** 200 Gb/s direct ConnectX-7 RoCE v2 link (no switch)

---

## Overview

This repo documents everything needed to:
1. Cable and configure a direct 200 Gb/s fabric link between two DGX Sparks
2. Verify the link and run bandwidth/latency benchmarks
3. Launch distributed PyTorch training across both nodes using NCCL over RoCE

The setup uses a **direct attach cable** between the CX7 NICs — no InfiniBand switch required for a 2-node cluster.

---

## Cluster Topology

```
┌─────────────────────────┐         200 Gb/s Direct Fabric          ┌─────────────────────────┐
│      spark-dgx-1        │◄───────────────────────────────────────►│      spark-dgx-2        │
│   (Master / Rank 0)     │   enp1s0f0np0 ←→ enp1s0f0np0 (mlx5_0)  │   (Worker / Rank 1)     │
│      10.0.0.1/24        │             MTU 9000 / RoCEv2            │      10.0.0.2/24        │
└─────────────────────────┘                                          └─────────────────────────┘
         │                                                                       │
         └──────────────── LAN (192.168.0.0/24) ─────────────────────────────────┘
              enp1s0f1np1 (mlx5_1)                              192.168.0.218
              192.168.0.188
```

### Interface Map (per node)

| netdev           | mlx5   | Speed    | Subnet            | Use                   |
|------------------|--------|----------|-------------------|-----------------------|
| enp1s0f0np0      | mlx5_0 | 200 Gb/s | 10.0.0.x/24       | Direct fabric (NCCL)  |
| enp1s0f1np1      | mlx5_1 | 200 Gb/s | 192.168.0.x/24    | LAN / default route   |
| enP2p1s0f0np0    | mlx5_2 | 200 Gb/s | —                 | Available             |
| enP2p1s0f1np1    | mlx5_3 | 200 Gb/s | 192.168.0.x/24    | Secondary LAN         |
| enP7s7           | —      | 10 Gb/s  | 192.168.0.x/24    | Management            |

---

## Quick Start

### 1. Verify the fabric link is up

```bash
./scripts/verify_network_setup.sh
```

### 2. Run distributed training

**On spark-dgx-1 (rank 0):**
```bash
./scripts/distributed_train.sh 0 your_train_script.py --epochs 3 --lr 2e-4
```

**On spark-dgx-2 (rank 1) — within ~60 seconds:**
```bash
./scripts/distributed_train.sh 1 your_train_script.py --epochs 3 --lr 2e-4
```

### 3. Run benchmarks

```bash
# List available scenarios
python3 training/run_benchmark.py --list

# Run all scenarios
python3 training/run_benchmark.py --scenario all

# Single scenario
python3 training/run_benchmark.py --scenario distributed_rosa
```

---

## Key NCCL Insight: No GPU Direct RDMA on Grace Blackwell

The GB10 unified memory architecture does **not** support GPU Direct RDMA (`GDR`).  
All settings that enable GDR (`NCCL_NET_GDR_LEVEL`, `NCCL_DMABUF_ENABLE`) must be **disabled** or
training will hang during `dist.init_process_group()`.

See [`docs/02-distributed-training.md`](docs/02-distributed-training.md) for full explanation.

---

## Repo Structure

```
.
├── README.md                        ← You are here
├── configs/
│   ├── netplan-node0.yaml           ← Netplan config for spark-dgx-1
│   ├── netplan-node1.yaml           ← Netplan config for spark-dgx-2
│   └── nccl-env.sh                  ← NCCL env vars to source before training
├── docs/
│   ├── 01-network-setup.md          ← Physical setup, netplan, IP, verification
│   ├── 02-distributed-training.md   ← NCCL tuning, launcher, performance
│   └── 03-troubleshooting.md        ← Common issues and fixes
├── scripts/
│   ├── distributed_train.sh         ← Generic 2-node torchrun launcher
│   ├── verify_network_setup.sh      ← Pre-flight network check
│   └── launch_local_training.sh     ← Single-node local storage launcher
└── training/
    ├── benchmark_train.py           ← PyTorch training script (LoRA fine-tuning)
    ├── run_benchmark.py             ← Benchmark orchestration (single + distributed)
    ├── configs.yaml                 ← Benchmark scenarios config
    └── run_manual_distributed.sh    ← Manual command generator for both nodes
```

---

## Hardware Specs (per node)

- **GPU:** NVIDIA GB10 (Grace Blackwell Superchip)
- **RDMA NICs:** 4× ConnectX-7 (mlx5_0–mlx5_3) @ 200 Gb/s each
- **Link layer:** Ethernet (RoCE v2) — not native InfiniBand
- **OFED:** MLNX OFED 24.07
- **UCX:** 1.17.0
- **Kernel:** Linux 6.17.0-1014-nvidia aarch64

---

## Measured Performance

| Metric                        | Value                |
|-------------------------------|----------------------|
| Fabric RTT (ping)             | ~0.12 ms             |
| NCCL bandwidth (ib_write_bw)  | ~185–195 Gb/s        |
| NCCL practical throughput     | ~23–24 GB/s          |
| Llama-2-7B epoch speedup      | 1.87× (single→dual)  |
| DDP scaling efficiency        | ~93.5%               |

---

## References

- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch Distributed Training Guide](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [NVIDIA DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/)
- [MLNX OFED Documentation](https://docs.nvidia.com/networking/display/ofedv24070612)
