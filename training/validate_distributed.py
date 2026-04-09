#!/usr/bin/env python3
"""
Distributed training validation script for 2-node DGX Spark cluster.
Tests NCCL connectivity, bandwidth, and GPU Direct status.

Usage:
  Node 0: ./scripts/distributed_train.sh 0 training/validate_distributed.py
  Node 1: ./scripts/distributed_train.sh 1 training/validate_distributed.py
"""

import os
import sys
import time
import torch
import torch.distributed as dist


def fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def check_nccl_mode():
    gdr = int(os.environ.get("NCCL_NET_GDR_LEVEL", 0))
    dmabuf = int(os.environ.get("NCCL_DMABUF_ENABLE", 0))
    if gdr >= 2 and dmabuf == 1:
        return "DMA-BUF GPU Direct (max performance)"
    elif gdr == 0:
        return "CPU path / safe fallback (no GPU Direct)"
    else:
        return f"Mixed (GDR_LEVEL={gdr}, DMABUF={dmabuf})"


def main():
    # ── Init ─────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log("\n" + "=" * 60)
    log("  DGX Spark Distributed Validation")
    log("=" * 60)
    log(f"  World size : {world_size}")
    log(f"  Backend    : NCCL {torch.cuda.nccl.version()}")
    log(f"  Interface  : {os.environ.get('NCCL_SOCKET_IFNAME', 'not set')}")
    log(f"  IB device  : {os.environ.get('NCCL_IB_HCA', 'not set')}")
    log(f"  NCCL mode  : {check_nccl_mode()}")
    log("=" * 60)

    dist.barrier()
    print(f"  [rank {rank}] GPU: {torch.cuda.get_device_name(device)} | "
          f"Mem: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB",
          flush=True)
    dist.barrier()

    # ── Test 1: Basic allreduce ───────────────────────────────────────────
    log("\n[1/4] Basic allreduce ... ")
    t = torch.ones(1, device=device) * rank
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    assert t.item() == expected, f"allreduce mismatch: got {t.item()}, expected {expected}"
    log("      PASS")

    # ── Test 2: Latency (small tensor, 1 KB) ─────────────────────────────
    log("[2/4] Latency test (1 KB tensor, 100 iterations) ...")
    size = 256  # 256 float32 = 1 KB
    t = torch.ones(size, device=device)
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_us = (elapsed / 100) * 1e6
    log(f"      Avg latency: {avg_us:.1f} µs  ({'OK' if avg_us < 500 else 'HIGH — check interface'})")

    # ── Test 3: Bandwidth (large tensor sweep) ────────────────────────────
    log("[3/4] Bandwidth test ...")
    results = []
    for size_mb in [1, 64, 256, 1024]:
        n_floats = (size_mb * 1024 * 1024) // 4
        t = torch.ones(n_floats, device=device)
        dist.barrier()
        torch.cuda.synchronize()
        # Warmup
        for _ in range(3):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        # Timed
        start = time.perf_counter()
        iters = 10
        for _ in range(iters):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        # allreduce transfers 2*(N-1)/N * size across the fabric; for N=2 that's 1x size each way
        bw_gbs = (size_mb / 1024) / (elapsed / iters)
        results.append((size_mb, bw_gbs))
        log(f"      {size_mb:>5} MB tensor → {bw_gbs:.2f} GB/s")

    peak = max(r[1] for r in results)
    if peak > 15:
        bw_note = "✓ Excellent (RDMA active)"
    elif peak > 5:
        bw_note = "~ OK (verify RDMA interface)"
    else:
        bw_note = "✗ Low — likely using TCP socket fallback"
    log(f"      Peak: {peak:.2f} GB/s  {bw_note}")

    # ── Test 4: Broadcast ─────────────────────────────────────────────────
    log("[4/4] Broadcast test ...")
    t = torch.zeros(1024, device=device)
    if rank == 0:
        t.fill_(42.0)
    dist.broadcast(t, src=0)
    assert t[0].item() == 42.0, "Broadcast failed"
    log("      PASS")

    # ── Summary ───────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("  RESULT: All tests passed")
    log(f"  Peak allreduce bandwidth: {peak:.2f} GB/s")
    if peak < 5:
        log("  WARNING: Bandwidth is low. Check NCCL_SOCKET_IFNAME")
        log("           and NCCL_IB_HCA point to enp1s0f0np0 / mlx5_0")
    log("=" * 60 + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
