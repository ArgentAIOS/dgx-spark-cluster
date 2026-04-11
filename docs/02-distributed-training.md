# Distributed Training on NVIDIA DGX Spark: A Complete Guide

**Author:** Sem  
**Environment:** 2x NVIDIA DGX Spark (Grace Blackwell GB200 NVL32)  
**Date:** November 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Hardware Overview](#hardware-overview)
3. [Network Architecture](#network-architecture)
4. [The Challenge](#the-challenge)
5. [The Solution](#the-solution)
6. [Configuration Details](#configuration-details)
7. [The Generic Launcher Script](#the-generic-launcher-script)
8. [How to Use It](#how-to-use-it)
9. [Troubleshooting](#troubleshooting)
10. [Performance Characteristics](#performance-characteristics)
11. [Conclusion](#conclusion)

---

## Introduction

This guide documents our journey setting up distributed PyTorch training across two NVIDIA DGX Spark systems. These machines represent NVIDIA's latest Grace Blackwell architecture, combining ARM-based Grace CPUs with Blackwell GPUs in a tightly integrated NVL32 configuration.

What makes this setup unique—and challenging—is that the Grace Blackwell architecture intentionally **does not support GPU Direct RDMA**. This architectural decision required us to carefully tune NCCL (NVIDIA Collective Communications Library) to work optimally over standard InfiniBand RDMA without direct GPU memory access.

## Hardware Overview

### DGX Spark Specifications

Each DGX Spark node features:
- **CPU:** NVIDIA Grace ARM processors (10x Cortex-X925 + 10x Cortex-A725)
- **GPU:** GB10 Grace Blackwell Superchip
- **Memory:** 128 GB unified LPDDR5x (273 GB/s bandwidth)
- **Network:** 200 Gbps RoCEv2 (ConnectX-7 / mlx5_0)
- **OS:** DGX Spark OS 7.5.0 (Ubuntu 22.04 aarch64)

### Our Cluster

- **Node 1 (spark-dgx-1):** 10.0.0.1 (Master) / 192.168.0.188 (LAN) / 100.122.26.9 (Tailscale)
- **Node 2 (spark-dgx-2):** 10.0.0.2 (Worker) / 192.168.0.218 (LAN) / 100.81.184.19 (Tailscale)
- **Connection:** Direct 200 Gbps QSFP112 DAC, point-to-point (no switch)

## Network Architecture

The two nodes are connected via a direct InfiniBand connection:

```
┌─────────────────────┐         200 Gbps IB         ┌─────────────────────┐
│   spark-dgx-1       │◄──────────────────────────►│   spark-dgx-2       │
│   (10.0.0.1)        │     mlx5_0 (enp1s0f0np0)   │   (10.0.0.2)        │
│   Master (rank 0)   │                             │   Worker (rank 1)   │
└─────────────────────┘                             └─────────────────────┘
```

Key network details:
- **Interface:** `enp1s0f0np0` (standard naming on both nodes)
- **InfiniBand Device:** `mlx5_0`
- **Bandwidth:** 200 Gbps theoretical, ~20-25 GB/s practical
- **Latency:** ~100-200 microseconds

### Verifying Network Configuration

```bash
# Check InfiniBand status
ibdev2netdev

# Expected output:
# mlx5_0 port 1 ==> enp1s0f0np0 (Up)

# Test connectivity
ping -c 3 10.0.0.2  # From node 1
ping -c 3 10.0.0.1  # From node 2
```

## Do You Need Docker?

**Short answer: No.** Docker is not required for distributed training on DGX Spark.

You can run everything directly on the host OS if both Sparks have the same OS version, CUDA, and PyTorch installed — which they do out of the box. The native approach is simpler, faster to set up, and eliminates container networking complexity.

Consider Docker only if you need environment isolation for multiple projects with conflicting dependencies, or you're deploying with NGC containers. **Start without Docker** and only add it if you hit a specific environment management problem.

---

## Official NVIDIA Resources

Before diving into our configuration, be aware of NVIDIA's own guides:

| Resource | URL |
|---|---|
| **Spark Playbooks** (30-min guides) | https://build.nvidia.com/spark |
| **GitHub: dgx-spark-playbooks** | https://github.com/NVIDIA/dgx-spark-playbooks |
| **Clustering User Guide** | https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html |
| **Developer Forum** | https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10 |

Our guide goes beyond the official docs by documenting the specific NCCL tuning required when GPU Direct RDMA is not available — a situation the official docs don't address in depth.

---

## Software Installation

### Prerequisites (Both Nodes)

Verify your stack before starting:

```bash
# CUDA
nvcc --version

# PyTorch with CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# GPU visibility
nvidia-smi

# torchrun (modern distributed launcher)
which torchrun
```

### Install NCCL (Both Nodes)

```bash
# Add CUDA repo for ARM64
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install NCCL
sudo apt install -y libnccl2 libnccl-dev

# Verify
dpkg -l | grep nccl
```

### Build NCCL Test Suite (Recommended)

```bash
# On both nodes
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests
make MPI=1

# Creates test binaries in ./build/
ls build/all_reduce_perf
```

### Install OpenMPI (Optional)

Only needed if you want MPI-based launching. We recommend torchrun instead (see [Launch Methods](#torchrun-vs-mpi)).

```bash
sudo apt install -y openmpi-bin libopenmpi-dev
mpirun --version
```

---

## The Challenge

Getting distributed training working on Grace Blackwell presented several challenges:

### 1. No GPU Direct RDMA Support

Unlike traditional NVIDIA architectures (A100, H100), Grace Blackwell's unified memory architecture **does not support GPU Direct RDMA** (`GDR`). This means:
- GPU memory cannot be directly accessed by InfiniBand NICs
- Data must be staged through CPU memory
- Different NCCL tuning is required

### 2. NCCL Configuration Complexity

NCCL has dozens of environment variables, and the optimal settings for Grace Blackwell differ significantly from other architectures:

```bash
# ❌ What DOESN'T work:
export NCCL_NET_GDR_LEVEL=2        # Causes hangs
export NCCL_IB_GID_INDEX=0         # Wrong GID index
export NCCL_ALGO=Tree              # Suboptimal for 2 nodes

# ✅ What DOES work:
export NCCL_NET_GDR_LEVEL=0        # Disable GDR
export NCCL_IB_GID_INDEX=3         # Correct for our setup
export NCCL_ALGO=Ring              # Best for 2-node clusters
```

### 3. Process Group Initialization Failures

Initial attempts resulted in cryptic errors:
```
RuntimeError: [1] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store failed
```

This was caused by:
- Incorrect `MASTER_ADDR` resolution
- Wrong network interface selection
- GID index mismatches
- Timeouts due to GPU Direct RDMA attempts

### 4. Interface Name Variations

The InfiniBand interface name `enp1s0f0np0` is non-standard and varies by system. If NCCL selects the wrong interface, training will fail or use a slow path.

## The Solution

After extensive testing and collaboration with NVIDIA support documentation, we developed a working configuration:

### Key Insights

1. **Disable GPU Direct completely** via `NCCL_NET_GDR_LEVEL=0` and `NCCL_DMABUF_ENABLE=0`
2. **Use the correct GID index** (3 in our case) via `NCCL_IB_GID_INDEX=3`
3. **Explicitly specify the InfiniBand interface** via `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA`
4. **Use Ring algorithm** for optimal 2-node performance
5. **Increase timeouts** to account for initialization overhead

### Working Environment Variables

```bash
export NCCL_SOCKET_IFNAME=enp1s0f0np0    # Network interface
export GLOO_SOCKET_IFNAME=enp1s0f0np0    # For process group init
export NCCL_IB_DISABLE=0                  # Enable InfiniBand
export NCCL_IB_HCA=mlx5_0                 # Specify IB device
export NCCL_NET=IB                        # Use InfiniBand transport
export NCCL_IB_GID_INDEX=3                # Correct GID for our setup
export NCCL_NET_GDR_LEVEL=0               # Disable GPU Direct RDMA
export NCCL_DMABUF_ENABLE=0               # Disable DMA-BUF
export NCCL_IB_TIMEOUT=22                 # Increase timeout
export NCCL_IB_RETRY_CNT=7                # Retry on errors
export NCCL_IB_QPS_PER_CONNECTION=4       # Queue pairs
export NCCL_IB_TC=106                     # Traffic class
export NCCL_ALGO=Ring                     # Algorithm
export NCCL_PROTO=Simple                  # Protocol
export NCCL_DEBUG=WARN                    # Logging level
export NCCL_TIMEOUT=600                   # Overall timeout (10 min)
```

## Configuration Details

### Why These Settings Matter

#### `NCCL_NET_GDR_LEVEL=0`
GPU Direct RDMA is not supported on Grace Blackwell. Setting this to anything other than 0 causes hangs during initialization.

#### `NCCL_IB_GID_INDEX=3`
The Global Identifier (GID) index determines which InfiniBand path is used. Index 3 corresponds to RoCEv2, which is what our setup uses. Wrong GID indices cause communication failures.

#### `NCCL_SOCKET_IFNAME=enp1s0f0np0`
Explicitly tells NCCL which network interface to use. Without this, NCCL might select `eth0` or another interface, resulting in slow or failed communication.

#### `NCCL_ALGO=Ring`
For 2-node setups, Ring algorithm is optimal. Tree algorithms are better for larger clusters but add overhead for small clusters.

#### `NCCL_IB_TIMEOUT=22` and `NCCL_IB_RETRY_CNT=7`
Grace Blackwell initialization can take longer due to the unified memory architecture. These settings prevent premature timeouts.

### Finding Your GID Index

If you're setting this up on different hardware, find your GID index:

```bash
# Method 1: Check with ibstat
ibstat mlx5_0 | grep GID

# Method 2: Test different indices
for i in {0..3}; do
  echo "Testing GID index $i"
  NCCL_IB_GID_INDEX=$i python3 -c "import torch.distributed as dist; dist.init_process_group('nccl')"
done
```

## The Generic Launcher Script

Instead of writing custom launch scripts for each training job, we created a universal launcher:

```bash
#!/bin/bash
# Generic distributed training launcher for 2-node DGX Spark cluster
# Usage: ./distributed_train.sh <node_rank> <python_script> [script args...]

set -e

# Validate arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <node_rank> <python_script> [script args...]"
    echo ""
    echo "Examples:"
    echo "  Node 0: ./distributed_train.sh 0 train.py --epochs 10 --lr 2e-4"
    echo "  Node 1: ./distributed_train.sh 1 train.py --epochs 10 --lr 2e-4"
    echo ""
    echo "Arguments:"
    echo "  node_rank     : 0 for first node, 1 for second node"
    echo "  python_script : Path to your training script"
    echo "  [script args] : Any additional arguments to pass to your script"
    exit 1
fi

NODE_RANK=$1
shift
TRAINING_SCRIPT=$1
shift
SCRIPT_ARGS="$@"

# Cluster configuration
MASTER_ADDR="10.0.0.1"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=1

# Network configuration for DGX Spark
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_NET=IB
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=0  # GPU Direct not supported on Grace Blackwell
export NCCL_DMABUF_ENABLE=0
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=106
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=600

echo "=========================================="
echo "DGX Spark Distributed Training Launcher"
echo "=========================================="
echo "Node rank: $NODE_RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Training script: $TRAINING_SCRIPT"
echo "Script arguments: $SCRIPT_ARGS"
echo "Network: 200Gbps via $NCCL_SOCKET_IFNAME (mlx5_0)"
echo "=========================================="
echo ""

# Validate training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

echo "Starting training in 3 seconds..."
sleep 3

# Launch distributed training
python3 -m torch.distributed.run \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $TRAINING_SCRIPT $SCRIPT_ARGS

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully on node $NODE_RANK"
else
    echo "✗ Training failed on node $NODE_RANK (exit code $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
```

### Script Features

1. **Simple interface:** Just specify node rank and your training script
2. **Automatic configuration:** All NCCL settings handled automatically
3. **Error checking:** Validates script exists before launching
4. **Clear output:** Shows configuration and status
5. **Reusable:** Works with any PyTorch distributed training script

## How to Use It

### Step 1: Prepare Your Training Script

Your Python script must initialize distributed training:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}/{world_size} initialized")
    
    # Create model and move to correct GPU
    model = YourModel()
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    # Training loop
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer)
        
    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

### Step 2: Launch on Both Nodes

**Terminal on Node 1 (spark-dgx-1):**
```bash
./distributed_train.sh 0 my_train.py --epochs 10 --batch-size 32 --lr 2e-4
```

**Terminal on Node 2 (spark-dgx-2):**
```bash
./distributed_train.sh 1 my_train.py --epochs 10 --batch-size 32 --lr 2e-4
```

**Important:** 
- Launch on **both nodes within ~60 seconds** to avoid initialization timeout
- Use **identical arguments** on both nodes
- Node ranks must be **0 and 1** (not interchangeable)

### Step 3: Monitor Progress

The launcher will display:
```
==========================================
DGX Spark Distributed Training Launcher
==========================================
Node rank: 0
Master: 10.0.0.1:29500
Training script: my_train.py
Script arguments: --epochs 10 --batch-size 32 --lr 2e-4
Network: 200Gbps via enp1s0f0np0 (mlx5_0)
==========================================

Starting training in 3 seconds...
```

You'll see NCCL initialization logs:
```
NCCL version 2.21.5+cuda12.4
[0] mlx5_0:1 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE
[0] NCCL INFO comm 0x... rank 0 nranks 2
```

### Real-World Example: LLM Fine-tuning

```bash
# Node 0
./distributed_train.sh 0 fine_tune_llama.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset ./data/instructions.json \
  --output ./models/llama-finetuned \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --warmup-steps 100

# Node 1 (same command, different rank)
./distributed_train.sh 1 fine_tune_llama.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset ./data/instructions.json \
  --output ./models/llama-finetuned \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --warmup-steps 100
```

## Torchrun vs. MPI

There are two ways to launch distributed training. **Use torchrun** — it's simpler and handles everything automatically.

### torchrun (Recommended)

```bash
# On spark-dgx-1
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 train.py

# On spark-dgx-2 (within 60 seconds)
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 train.py
```

torchrun automatically sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` as environment variables. Your script just calls `dist.init_process_group('nccl')` and it works.

### MPI (Alternative)

```bash
mpirun -np 2 \
    -H 10.0.0.1:1,10.0.0.2:1 \
    -bind-to none -map-by slot \
    -x NCCL_SOCKET_IFNAME=enp1s0f0np0 \
    -x LD_LIBRARY_PATH \
    python3 train.py
```

Requires passwordless SSH and your script must manually initialize the process group.

> **WARNING:** Never wrap torchrun inside MPI (`mpirun torchrun ...`). Use one or the other, not both. This is a common mistake that causes cryptic failures.

---

## DDP Training Template

A complete, working PyTorch DDP training script for DGX Spark. Copy this as a starting point.

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup():
    """Initialize distributed environment. torchrun sets env vars automatically."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def train():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size} running on GPU {local_rank}")

    # Create model and move to GPU
    model = YourModel().to(local_rank)

    # Wrap with DDP — this handles gradient synchronization
    model = DDP(model, device_ids=[local_rank])

    # IMPORTANT: DistributedSampler partitions data across GPUs
    dataset = YourDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(10):
        # IMPORTANT: Set epoch for proper shuffling each epoch
        sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(local_rank)
            target = target.to(local_rank)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()  # DDP auto-syncs gradients here
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if rank == 0:
            print(f"Epoch {epoch} done.")

    # Only save from rank 0 — use model.module to unwrap DDP
    if rank == 0:
        torch.save(model.module.state_dict(), "model_final.pt")
        print("Model saved.")

    cleanup()

if __name__ == "__main__":
    train()
```

### Key Rules for DDP Scripts

1. **Always use `DistributedSampler`** — ensures each GPU gets different data
2. **Call `sampler.set_epoch(epoch)`** — ensures proper shuffling across epochs
3. **Only save/print from rank 0** — avoids duplicate outputs and corrupted saves
4. **Use `model.module`** to access the unwrapped model when saving
5. **Move data to the correct GPU** — use `.to(local_rank)`, not `.cuda()`

---

## Testing Your Setup

Before running real training, verify the cluster works end-to-end.

### Test 1: NCCL Bandwidth

```bash
# Requires nccl-tests built with MPI (see Software Installation)
mpirun -np 2 -H 10.0.0.1:1,10.0.0.2:1 \
    -x NCCL_SOCKET_IFNAME=enp1s0f0np0 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_NET_GDR_LEVEL=0 \
    ./nccl-tests/build/all_reduce_perf -b 8 -e 512M -f 2 -g 1
```

Look for bandwidth reaching ~20-24 GB/s at large message sizes.

### Test 2: Simple DDP Validation

Create `test_ddp.py`:

```python
import os
import torch
import torch.distributed as dist

def test():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    # Each rank creates a tensor with its rank+1
    tensor = torch.ones(2, 2, device=device) * (rank + 1)
    print(f"[Rank {rank}] Before all_reduce: {tensor[0,0].item()}")

    # Sum across all ranks
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] After all_reduce: {tensor[0,0].item()}")

    # Should be 3.0 (rank0=1 + rank1=2)
    expected = 3.0
    status = "PASS" if tensor[0,0].item() == expected else "FAIL"
    print(f"[Rank {rank}] {status}")

    dist.destroy_process_group()

if __name__ == "__main__":
    test()
```

```bash
# spark-dgx-1
NCCL_SOCKET_IFNAME=enp1s0f0np0 torchrun --nnodes=2 --nproc_per_node=1 \
    --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 test_ddp.py

# spark-dgx-2 (within 60 seconds)
NCCL_SOCKET_IFNAME=enp1s0f0np0 torchrun --nnodes=2 --nproc_per_node=1 \
    --node_rank=1 --master_addr=10.0.0.1 --master_port=29500 test_ddp.py
```

### Test 3: Monitor GPUs During Training

```bash
# On both nodes
watch -n 1 nvidia-smi
```

You should see GPU utilization > 0% and memory being consumed on both nodes.

---

## Troubleshooting

### Problem: "Connection refused" or "Timeout"

**Symptoms:**
```
RuntimeError: [Rank 1] Watchdog caught collective operation timeout
```

**Solutions:**
1. Verify both nodes can ping each other:
   ```bash
   ping -c 3 10.0.0.1  # From node 2
   ping -c 3 10.0.0.2  # From node 1
   ```

2. Check firewall isn't blocking port 29500:
   ```bash
   sudo ufw status
   sudo ufw allow 29500/tcp
   ```

3. Ensure SSH works without password:
   ```bash
   ssh 10.0.0.2 hostname  # Should return immediately
   ```

### Problem: "No such device" or "mlx5_0 not found"

**Symptoms:**
```
NCCL WARN NET/IB : No device found
```

**Solutions:**
1. Check InfiniBand status:
   ```bash
   ibdev2netdev
   ibstat
   ```

2. Verify NVIDIA drivers:
   ```bash
   nvidia-smi
   ```

3. Check if `mlx5_core` module is loaded:
   ```bash
   lsmod | grep mlx5
   ```

### Problem: Training hangs at initialization

**Symptoms:**
- Script runs but never prints "Epoch 1"
- CPU usage high but no GPU activity
- Hangs at `dist.init_process_group()`

**Solutions:**
1. Set `NCCL_DEBUG=INFO` to see where it hangs:
   ```bash
   export NCCL_DEBUG=INFO
   ```

2. Check if it's trying to use GPU Direct:
   ```bash
   # Should see "NET_GDR_LEVEL 0" in logs
   grep GDR /tmp/nccl_debug.log
   ```

3. Verify GID index is correct:
   ```bash
   # Test with different GID indices
   export NCCL_IB_GID_INDEX=0  # Then try 1, 2, 3
   ```

### Problem: Slow training (not using InfiniBand)

**Symptoms:**
- Training works but is unexpectedly slow
- Network usage low (check with `iftop`)

**Solutions:**
1. Check NCCL is using InfiniBand:
   ```bash
   # Should see "NET/IB" in logs, not "NET/Socket"
   grep "NCCL INFO" /tmp/nccl_debug.log
   ```

2. Verify correct interface selected:
   ```bash
   # Should show enp1s0f0np0, not eth0
   export NCCL_DEBUG=INFO
   # Look for "Using [0]mlx5_0:1/RoCE" in output
   ```

### Problem: "GID index out of range"

**Symptoms:**
```
NCCL WARN NET/IB : Invalid GID index 3
```

**Solution:**
Find available GIDs:
```bash
show_gids | grep mlx5_0
```

Update `NCCL_IB_GID_INDEX` in the launcher script to a valid index.

### Debugging Checklist

When things go wrong:

```bash
# 1. Network connectivity
ping -c 3 10.0.0.2
ssh 10.0.0.2 hostname

# 2. InfiniBand status
ibdev2netdev
ibstat mlx5_0

# 3. GPU visibility
nvidia-smi

# 4. Python environment
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch.distributed; print(torch.distributed.is_nccl_available())"

# 5. Port availability
netstat -tuln | grep 29500

# 6. NCCL test (run on both nodes simultaneously)
./distributed_train.sh 0 validate_distributed.py  # Node 0
./distributed_train.sh 1 validate_distributed.py  # Node 1
```

## Performance Characteristics

### Network Bandwidth

Measured with NCCL bandwidth test:

```
Size (B)    Bandwidth (GB/s)    Algorithm
1KB         0.5                 Ring
1MB         18.2                Ring
1GB         23.4                Ring
4GB         24.1                Ring
```

**Key takeaway:** We achieve ~24 GB/s out of 25 GB/s theoretical (200 Gbps), which is excellent for non-GPU-Direct configuration.

### Latency

```
Message Size    Latency (μs)
8 bytes         110
256 bytes       125
4 KB            180
```

**Note:** Latency is ~2-3x higher than GPU Direct RDMA would achieve, but this is expected and acceptable given the architecture.

### Training Performance

Example: Llama-2-7B fine-tuning

| Configuration | Time per Epoch | GPU Utilization |
|---------------|----------------|-----------------|
| Single node   | 45 min         | 85%             |
| 2-node (ours) | 24 min         | 82%             |
| **Speedup**   | **1.87x**      | -               |

The scaling efficiency is ~93.5%, which is excellent for distributed training.

### When to Use Multi-Node

Multi-node training is beneficial when:
- ✅ Model is too large for a single GPU
- ✅ Dataset is large (>100GB)
- ✅ Training time is measured in hours/days
- ✅ Batch size benefits from distributed data parallelism

It may not be worth it when:
- ❌ Model fits comfortably on one GPU
- ❌ Training completes in <1 hour
- ❌ Network bottleneck exceeds computation time
- ❌ Small batch sizes (gradient accumulation might be better)

## Conclusion

Setting up distributed training on NVIDIA Grace Blackwell required understanding the architecture's unique characteristics, particularly the absence of GPU Direct RDMA support. Key lessons learned:

1. **Architecture matters:** Grace Blackwell's unified memory requires different tuning than traditional GPU architectures

2. **NCCL configuration is critical:** Wrong settings can cause hangs, failures, or degraded performance

3. **Start simple:** Test with minimal configurations before scaling up

4. **Document everything:** NCCL has many moving parts; keep notes on what works

5. **Generic tooling saves time:** A reusable launcher script eliminates repetitive setup

### Resources

- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [DGX Spark User Guide](https://docs.nvidia.com/dgx/)

### Our Setup Files

All configuration files are available in `/home/sem/`:
- `distributed_train.sh` - Generic launcher
- `validate_distributed.py` - Test script
- `README_DISTRIBUTED.md` - Quick reference
- `DGX_Spark_Distributed_Training_Guide.md` - This document

### Acknowledgments

Thanks to NVIDIA DevZone forums and the PyTorch community for guidance on NCCL tuning for Grace Blackwell architecture.

---

**Questions?** Feel free to reach out or open an issue. Happy distributed training! 🚀
