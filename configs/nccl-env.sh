#!/usr/bin/env bash
# NCCL environment variables for DGX Spark 2-node direct fabric cluster
# Source this file before training: source configs/nccl-env.sh
#
# IMPORTANT: GPU Direct RDMA is NOT supported on Grace Blackwell (GB10).
# GDR settings must be disabled or dist.init_process_group() will hang.

# ── Network interface selection ─────────────────────────────────────────
export NCCL_SOCKET_IFNAME=enp1s0f0np0   # Use direct fabric NIC
export GLOO_SOCKET_IFNAME=enp1s0f0np0   # Gloo backend (used for init)
export UCX_NET_DEVICES=mlx5_0:1

# ── InfiniBand / RoCE transport ──────────────────────────────────────────
export NCCL_IB_DISABLE=0                # Enable IB transport
export NCCL_IB_HCA=mlx5_0              # Pin to direct fabric CX7 port
export NCCL_NET=IB                      # Use InfiniBand/RoCE
export NCCL_IB_GID_INDEX=3             # RoCEv2 GID index (verify with show_gids)

# ── GPU Direct RDMA — MUST BE DISABLED on Grace Blackwell ───────────────
export NCCL_NET_GDR_LEVEL=0            # Disable GPU Direct RDMA
export NCCL_DMABUF_ENABLE=0            # Disable DMA-BUF

# ── Reliability tuning ───────────────────────────────────────────────────
export NCCL_IB_TIMEOUT=22              # Increase timeout (Grace Blackwell init is slower)
export NCCL_IB_RETRY_CNT=7            # Retry on transient errors
export NCCL_IB_QPS_PER_CONNECTION=4   # Queue pairs per connection
export NCCL_IB_TC=106                  # Traffic class
export NCCL_TIMEOUT=600                # Overall timeout (10 min)

# ── Algorithm selection ──────────────────────────────────────────────────
export NCCL_ALGO=Ring                  # Ring is optimal for 2-node clusters
export NCCL_PROTO=Simple               # Simple protocol

# ── Logging (set to INFO for debugging, WARN for production) ─────────────
export NCCL_DEBUG=WARN

echo "[nccl-env] NCCL configured for DGX Spark direct fabric (mlx5_0 / enp1s0f0np0)"
