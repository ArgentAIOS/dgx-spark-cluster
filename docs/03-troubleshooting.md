# Troubleshooting — DGX Spark Direct Fabric

## Quick Diagnostic Checklist

```bash
# 1. Network connectivity
ping -c 3 -I enp1s0f0np0 10.0.0.2

# 2. InfiniBand / RoCE status
rdma link show
ibstat mlx5_0

# 3. GPU visibility
nvidia-smi

# 4. Python environment
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch.distributed; print(torch.distributed.is_nccl_available())"

# 5. Port availability
ss -tuln | grep 29500

# 6. Run end-to-end validation (both nodes simultaneously)
./scripts/distributed_train.sh 0 validate_distributed.py   # spark-dgx-1
./scripts/distributed_train.sh 1 validate_distributed.py   # spark-dgx-2
```

---

## Problem: Connection refused / Timeout

**Symptom:**
```
RuntimeError: [Rank 1] Watchdog caught collective operation timeout
```

**Fixes:**
1. Verify both nodes can ping over the fabric:
   ```bash
   ping -c 3 -I enp1s0f0np0 10.0.0.2   # from node 1
   ping -c 3 -I enp1s0f0np0 10.0.0.1   # from node 2
   ```
2. Unblock port 29500:
   ```bash
   sudo ufw allow 29500/tcp
   sudo ufw allow from 10.0.0.0/24
   ```
3. Ensure passwordless SSH works over the fabric:
   ```bash
   ssh 10.0.0.2 hostname   # should return immediately
   ```
4. Check that both nodes launched within ~60 seconds of each other.

---

## Problem: Training hangs at `dist.init_process_group()`

**Symptom:** Script starts, prints launcher header, then stalls indefinitely.

**Root cause (most common):** GPU Direct RDMA is enabled but **not supported** on Grace Blackwell.

**Fix:** Confirm these are set in `scripts/distributed_train.sh` or sourced from `configs/nccl-env.sh`:
```bash
export NCCL_NET_GDR_LEVEL=0
export NCCL_DMABUF_ENABLE=0
```

**Debug:**
```bash
export NCCL_DEBUG=INFO
# Look for "NET_GDR_LEVEL 0" in output
# Should NOT see "GDRCopy" or "GPUDirect"
```

---

## Problem: No InfiniBand / RDMA devices found

**Symptom:**
```
NCCL WARN NET/IB : No device found
ibstat: No InfiniBand devices found
```

**Fixes:**
```bash
# Check drivers loaded
lsmod | grep mlx5_core

# Reload if needed
sudo modprobe -r mlx5_ib mlx5_core
sudo modprobe mlx5_core mlx5_ib

# Check RDMA link state
rdma link show
# All 4 ports should show: state ACTIVE physical_state LINK_UP
```

---

## Problem: Wrong network interface selected

**Symptom:** Training runs but bandwidth is unexpectedly low, or you see `NET/Socket` instead of `NET/IB` in NCCL logs.

**Verify:**
```bash
export NCCL_DEBUG=INFO
# Should see: "Using [0]mlx5_0:1/RoCE"
# Should NOT see: eth0, tailscale0, enP7s7
```

**Fix:** Confirm these exports are set:
```bash
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_HCA=mlx5_0
```

---

## Problem: "GID index out of range" or communication failures

**Symptom:**
```
NCCL WARN NET/IB : Invalid GID index 3
```

**Fix:** Find valid GID indices for your setup:
```bash
show_gids | grep mlx5_0
# or
ibv_devinfo -d mlx5_0 | grep -A2 "GID"
```

Update `NCCL_IB_GID_INDEX` in `configs/nccl-env.sh` to a valid index.  
For RoCEv2, this is typically index 3; for RoCEv1 it is typically 1.

---

## Problem: Port state is "Down" or "Polling"

**Symptom:**
```bash
ibstat mlx5_0
# Port 1: State: Down / Polling
```

**Fixes:**
1. Check the physical cable — re-seat both ends
2. Try a different cable (DAC cables can be DOA)
3. Verify the correct ports are cabled (see `docs/01-network-setup.md` §Physical Setup)
4. Check firmware:
   ```bash
   sudo mst start
   sudo mlxconfig -d /dev/mst/mt4129_pciconf0 query | grep LINK_TYPE
   ```

---

## Problem: SSH requires password every time

**Fixes:**
```bash
# Fix key permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
chmod 644 ~/.ssh/authorized_keys

# Verify authorized_keys on remote
ssh 10.0.0.2 "cat ~/.ssh/authorized_keys"
# Should contain your public key

# Ensure PubkeyAuthentication is enabled
grep PubkeyAuthentication /etc/ssh/sshd_config
# Should show: PubkeyAuthentication yes
```

---

## Problem: Slow throughput (not using fabric)

**Check with:**
```bash
# During training: watch network utilization
watch -n 1 'cat /sys/class/net/enp1s0f0np0/statistics/tx_bytes'

# Or with iftop (if installed)
sudo iftop -i enp1s0f0np0
```

If `enp1s0f0np0` shows no traffic, NCCL fell back to socket. Check NCCL_DEBUG=INFO for which interface it chose.

---

## Problem: `ibdev2netdev` command not found

```bash
sudo apt install infiniband-diags
# or use rdma CLI (already installed):
rdma link show
```

---

## Useful One-Liners

```bash
# Full NCCL debug log to file
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL ./scripts/distributed_train.sh 0 your_script.py 2>&1 | tee /tmp/nccl_debug.log

# Watch RoCE port counters live
watch -n 1 'ethtool -S enp1s0f0np0 | grep -E "rx_bytes|tx_bytes|discards|errors"'

# Quick bandwidth test (server on dgx-2, client on dgx-1)
# On dgx-2: ib_write_bw -d mlx5_0 --report_gbits
# On dgx-1: ib_write_bw -d mlx5_0 10.0.0.2 --report_gbits

# Kill stale distributed processes on a port
lsof -ti :29500 | xargs -r kill -9
```
