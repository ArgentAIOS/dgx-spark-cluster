#!/usr/bin/env bash
# NCCL environment for DGX Spark 2-node direct fabric cluster
#
# Usage:
#   source configs/nccl-env.sh          # defaults to 'safe' mode
#   source configs/nccl-env.sh safe     # CPU-path RDMA (always works)
#   source configs/nccl-env.sh dmabuf   # DMA-BUF GPU Direct (max perf)
#
# Which mode to use?
#   - Start with 'safe' to confirm training works
#   - Switch to 'dmabuf' for production — requires open NVIDIA driver (580+)
#     and kernel 6.x with DMA-BUF support (both present on DGX Spark OS 7.5+)

MODE="${1:-safe}"

# ── Network interface selection (same for both modes) ────────────────────
export NCCL_SOCKET_IFNAME=enp1s0f0np0   # Direct fabric NIC
export GLOO_SOCKET_IFNAME=enp1s0f0np0   # Gloo (used for process group init)
export UCX_NET_DEVICES=mlx5_0:1

# ── InfiniBand / RoCE transport (same for both modes) ────────────────────
export NCCL_IB_DISABLE=0                # Enable IB/RoCE transport
export NCCL_IB_HCA=mlx5_0              # Direct fabric CX7 port
export NCCL_NET=IB                      # InfiniBand/RoCE transport
export NCCL_IB_GID_INDEX=3             # RoCEv2 GID (verify with: show_gids)

# ── Reliability (same for both modes) ────────────────────────────────────
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=106
export NCCL_TIMEOUT=600
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

if [[ "$MODE" == "dmabuf" ]]; then
    # ── MODE: DMA-BUF GPU Direct RDMA ────────────────────────────────────
    # Requires: NVIDIA open driver 580+, kernel 6.x, DGX Spark OS 7.5+
    # GPU data transfers directly to/from RDMA NIC without CPU bounce buffer
    # Expected bandwidth: ~22-23 GB/s (near line rate)
    export NCCL_NET_GDR_LEVEL=5        # Enable GPU Direct at max level
    export NCCL_NET_GDR_READ=1         # Enable GPUDirect for reads too
    export NCCL_DMABUF_ENABLE=1        # Use DMA-BUF (modern GPU Direct path)
    export NCCL_DEBUG=WARN
    echo "[nccl-env] Mode: DMA-BUF GPU Direct (NCCL_NET_GDR_LEVEL=5, NCCL_DMABUF_ENABLE=1)"
    echo "[nccl-env] Verify with NCCL_DEBUG=INFO — look for 'GPU Direct RDMA Enabled'"
else
    # ── MODE: Safe / CPU-path RDMA (default) ─────────────────────────────
    # GPU data staged through CPU memory before RDMA transfer
    # Always works regardless of driver/kernel version
    # Expected bandwidth: ~18-20 GB/s (still fast, ~10-15% slower than dmabuf)
    export NCCL_NET_GDR_LEVEL=0        # Disable GPU Direct RDMA
    export NCCL_DMABUF_ENABLE=0        # Disable DMA-BUF
    export NCCL_DEBUG=WARN
    echo "[nccl-env] Mode: safe (CPU-path RDMA, NCCL_NET_GDR_LEVEL=0)"
fi
