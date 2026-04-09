# GPU Direct RDMA on DGX Spark (Grace Blackwell)

**TL;DR:** Use `source configs/nccl-env.sh dmabuf` for max performance.  
The old `nvidia-peermem` approach does NOT work on this hardware — use DMA-BUF instead.

---

## Background: Two GPU Direct Approaches

### Legacy: nvidia-peermem (does NOT work here)
- Used on x86 systems (A100, H100)
- Requires proprietary NVIDIA driver
- Loads `nvidia-peermem.ko` kernel module
- **Status on DGX Spark aarch64: BROKEN** — see below

### Modern: DMA-BUF (works on DGX Spark)
- Kernel-native (Linux 5.12+)
- Works with **open-source NVIDIA driver** (580+)
- No extra kernel module needed
- NCCL uses it via `NCCL_DMABUF_ENABLE=1`
- **Status on DGX Spark OS 7.5+: WORKS ✅**

---

## What We Tried and Why nvidia-peermem Failed

```bash
# Attempt 1 — modprobe
sudo modprobe nvidia-peermem
# Result: ERROR: could not insert 'nvidia_peermem': Invalid argument

# Attempt 2 — install loader package
sudo apt install nvidia-peermem-loader
# Result: Installs but module still won't load

# Attempt 3 — build nv_peer_mem from Mellanox source
git clone https://github.com/Mellanox/nv_peer_memory.git
./build_module.sh
# Result: DKMS fails — missing symbols ib_register_peer_memory_client
#         These symbols are not exported on ARM64 kernel 6.x
```

**Root cause:** The `ib_register_peer_memory_client` / `ib_unregister_peer_memory_client`
symbols are not available in the upstream kernel on ARM64. MLNX OFED provides them on
x86, but not on the DGX Spark's aarch64 kernel. This is expected on new hardware —
DMA-BUF is the replacement.

---

## The Working Solution: DMA-BUF

The open NVIDIA driver (580+) and Linux kernel 6.x both support the DMA-BUF
peer memory interface natively. No additional modules needed.

### Enable in NCCL:

```bash
source configs/nccl-env.sh dmabuf
# Sets:
#   NCCL_NET_GDR_LEVEL=5
#   NCCL_NET_GDR_READ=1
#   NCCL_DMABUF_ENABLE=1
```

### How to verify it's working:

```bash
export NCCL_DEBUG=INFO
./scripts/distributed_train.sh 0 training/validate_distributed.py
```

**Look for in NCCL output:**

```
# ✅ GPU Direct via DMA-BUF active:
NCCL INFO NET/IB : Using [0]mlx5_0:1 [RoCE]
NCCL INFO NET/Plugin : Using GPU Direct RDMA
NCCL INFO NET/IB : GPU Direct RDMA Enabled for HCA 0 'mlx5_0' [Send: 1 Recv: 1]

# ⚠️ Fallback — RDMA but no GPU Direct (safe mode):
NCCL INFO NET/IB : GPU Direct RDMA Disabled for HCA 0 'mlx5_0'

# ❌ No RDMA at all (wrong interface or module not loaded):
NCCL INFO NET/Socket : Using [0]enp1s0f0np0
```

Also run the preflight check:

```bash
python3 training/test_gpudirect_dmabuf.py
```

---

## Performance Comparison

| Mode | Path | Expected BW | CPU load |
|---|---|---|---|
| `safe` (default) | GPU → CPU RAM → RoCE → CPU RAM → GPU | ~18–20 GB/s | High |
| `dmabuf` | GPU → RoCE → GPU (zero-copy) | ~22–23 GB/s | Near zero |
| No RDMA (socket fallback) | TCP/IP | ~10–12 GB/s | Very high |

The bandwidth difference between safe and dmabuf is ~10–15%. For most training jobs
the bottleneck is compute, not gradient sync — start with `safe` and switch to
`dmabuf` once training is verified working.

---

## System Requirements for DMA-BUF GPU Direct

| Component | Required | This System |
|---|---|---|
| NVIDIA driver | Open 520+ | 580-open ✅ |
| Kernel | Linux 5.12+ | 6.17.0 ✅ |
| DGX Spark OS | 7.x | 7.5.0 ✅ |
| NCCL | 2.12+ | 2.27.7 ✅ |
| MLNX OFED | 5.x+ | 24.07 ✅ |

---

## If DMA-BUF Doesn't Work

1. Check kernel DMA-BUF support:
   ```bash
   grep DMA_BUF /boot/config-$(uname -r) | head -5
   # Should show: CONFIG_DMA_BUF=y
   ```

2. Check open driver is loaded (not proprietary):
   ```bash
   cat /proc/driver/nvidia/version
   # Should say "open" in the filename
   nvidia-smi --query-gpu=driver_version --format=csv,noheader
   ```

3. Check NCCL version supports DMA-BUF:
   ```bash
   python3 -c "import torch; print(torch.cuda.nccl.version())"
   # Need 2.12+
   ```

4. Fall back to safe mode and file a bug with NVIDIA DGX support:
   ```bash
   source configs/nccl-env.sh safe
   ```
