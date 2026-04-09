# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Distributed PyTorch training setup for two NVIDIA DGX Spark systems (Grace Blackwell GB10) connected via a direct 200 Gb/s InfiniBand fabric link. The critical constraint: Grace Blackwell does **not support GPU Direct RDMA (GDR)**, requiring all RDMA data to flow through CPU memory with specific NCCL configuration (`NCCL_NET_GDR_LEVEL=0`, `NCCL_DMABUF_ENABLE=0`).

## Architecture

- **2-node cluster** connected point-to-point (no switch), using RoCEv2 protocol over ConnectX-7 NICs
- **Fabric network**: 10.0.0.1 (node0) ↔ 10.0.0.2 (node1) on `enp1s0f0np0`, MTU 9000
- **LAN network**: 192.168.0.188 (node0), 192.168.0.218 (node1) on `enp1s0f1np1`
- **Single GPU rank per node** (`nproc_per_node=1`), ring all-reduce algorithm (optimal for 2-node)
- Training uses Hugging Face SFTTrainer with 4-bit quantization + LoRA (PEFT)

## Key Commands

```bash
# Verify network and hardware readiness
./scripts/verify_network_setup.sh

# Distributed training (run node 0 first, node 1 within 60s)
./scripts/distributed_train.sh 0 your_script.py --epochs 3
./scripts/distributed_train.sh 1 your_script.py --epochs 3

# Single-node local training
./scripts/launch_local_training.sh

# Benchmarks
python3 training/run_benchmark.py --list
python3 training/run_benchmark.py --scenario all
python3 training/run_benchmark.py --scenario distributed_rosa
```

## NCCL Modes (configs/nccl-env.sh)

Two modes via `source configs/nccl-env.sh <mode>`:
- **`safe`** (default) — `NCCL_NET_GDR_LEVEL=0`, `NCCL_DMABUF_ENABLE=0`. CPU-staged RDMA, ~18-20 GB/s. Always works.
- **`dmabuf`** — `NCCL_NET_GDR_LEVEL=5`, `NCCL_DMABUF_ENABLE=1`. GPU Direct via DMA-BUF, ~22-23 GB/s. Requires open NVIDIA driver 580+.

Legacy `nvidia-peermem` does NOT work on this ARM64 hardware — use DMA-BUF instead. See `docs/04-gpu-direct-rdma.md`.

Shared settings (both modes): `NCCL_IB_HCA=mlx5_0`, `NCCL_SOCKET_IFNAME=enp1s0f0np0`, `NCCL_IB_GID_INDEX=3`, `NCCL_ALGO=Ring`.

## Code Structure

- `configs/` — NCCL env vars (two-mode), netplan YAML for each node, NVMe-TCP systemd service, fstab entries
- `docs/` — Guides: cluster overview (00), network setup (01), distributed training (02), troubleshooting (03), GPU Direct RDMA (04), plus `rosa/` subdir for NVMe-TCP storage
- `scripts/` — Shell launchers: `distributed_train.sh` (generic 2-node torchrun wrapper), `verify_network_setup.sh` (pre-flight checks), `launch_local_training.sh`
- `training/` — `validate_distributed.py` (smoke test), `test_gpudirect_dmabuf.py` (GPU Direct preflight), `benchmark_train.py`, `run_benchmark.py`, `requirements.txt`

## Dependencies

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install -r training/requirements.txt
```

Requires: Ubuntu 22.04 (aarch64), CUDA 12.4+, Mellanox OFED 24.07, Python 3.8+

## When Modifying

- Two NCCL modes exist: `safe` (GDR=0, always works) and `dmabuf` (GDR=5, max perf). Don't mix settings from different modes.
- The master node is always node 0 (10.0.0.1:29500)
- Benchmark scripts use `--max_steps 50` for consistent comparison; don't change for benchmarking
- `run_benchmark.py` uses SSH to launch remote node processes — passwordless SSH between nodes is assumed
- Default model is `meta-llama/Llama-3.2-3B-Instruct` with LoRA rank 32
