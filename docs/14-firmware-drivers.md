# Firmware and Driver Updates

**How to update CUDA, OFED, and DGX OS without breaking your cluster.**

Grace Blackwell is sensitive to driver version mismatches — a bad NCCL/OFED combination can silently break distributed training. This guide documents the compatibility matrix and safe update process.

---

## Current Stack (Known Working)

| Component | Version | Notes |
|---|---|---|
| **DGX Spark OS** | 7.5.0 | Based on Ubuntu 22.04 aarch64 |
| **CUDA Toolkit** | 12.4+ | Required for PyTorch and tinygrad |
| **NVIDIA Driver** | 580+ | Required for DMA-BUF GPU Direct mode |
| **Mellanox OFED** | 24.07 | ConnectX-7 drivers for 200 Gb/s fabric |
| **NCCL** | 2.21.5+ | GPU collective communications |
| **PyTorch** | 2.x | With CUDA 12.4 wheels |
| **tinygrad** | 0.10.0 | EXO inference engine |

> **Rule of thumb:** Don't update components unless you have a specific reason. If training and inference are working, leave it alone.

---

## What Can Break

| Update | Risk | Symptom |
|---|---|---|
| CUDA toolkit | Medium | PyTorch fails to find CUDA, `torch.cuda.is_available()` returns False |
| NVIDIA driver | Medium | `nvidia-smi` fails, GPU not visible |
| OFED / mlx5 driver | **High** | NCCL hangs, `ibdev2netdev` shows no devices, training freezes at init |
| DGX OS | **High** | Can reset CUDA, drivers, and network config simultaneously |
| NCCL | Medium | Training hangs, wrong algorithm selected, bandwidth regression |
| Python / pip packages | Low | Import errors, version conflicts |

---

## Safe Update Process

### Before Any Update

```bash
# 1. Record current versions (save this output!)
nvidia-smi
nvcc --version
ofed_info -s
dpkg -l | grep -E "nccl|mlx5|nvidia"
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

# 2. Save to a file
echo "=== Pre-update snapshot $(date) ===" >> ~/version_snapshots.log
nvidia-smi >> ~/version_snapshots.log 2>&1
nvcc --version >> ~/version_snapshots.log 2>&1
ofed_info -s >> ~/version_snapshots.log 2>&1
dpkg -l | grep -E "nccl|mlx5|nvidia" >> ~/version_snapshots.log 2>&1

# 3. Run the distributed validation test
./scripts/launch_distributed.sh training/validate_distributed.py
# Save the output — this is your "before" baseline
```

### CUDA Toolkit Update

```bash
# Check available versions
apt list --upgradable 2>/dev/null | grep cuda

# Install specific version (don't just apt upgrade)
sudo apt install cuda-toolkit-12-4

# Verify
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available())"
```

### OFED / Mellanox Driver Update

**This is the highest-risk update.** OFED includes the mlx5 kernel module that drives your 200 Gb/s fabric link.

```bash
# Check current version
ofed_info -s

# Download from NVIDIA (match your OS and arch)
# https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/

# Install (example — adjust filename)
sudo ./mlnxofedinstall --force

# Reboot required
sudo reboot

# After reboot — verify
ibdev2netdev
# Should show: mlx5_0 port 1 ==> enp1s0f0np0 (Up)

ibstat mlx5_0
# Should show: State: Active, Physical state: LinkUp

# Run bandwidth test
ib_write_bw -d mlx5_0 10.0.0.2  # From spark-dgx-1
ib_write_bw -d mlx5_0            # On spark-dgx-2 (server mode)
```

### DGX OS Update

NVIDIA releases DGX OS updates through their standard channel. These can change CUDA, drivers, and kernel simultaneously.

```bash
# Check for updates
sudo apt update
sudo apt list --upgradable 2>/dev/null | grep dgx

# BEFORE updating: snapshot everything (see above)
# Update
sudo apt upgrade

# Reboot
sudo reboot

# After reboot: verify EVERYTHING
nvidia-smi
nvcc --version
ibdev2netdev
python3 -c "import torch; print(torch.cuda.is_available())"
./scripts/verify_network_setup.sh
```

---

## Compatibility Matrix

Known working combinations for DGX Spark (Grace Blackwell GB10):

| DGX OS | CUDA | Driver | OFED | NCCL | Status |
|---|---|---|---|---|---|
| 7.5.0 | 12.4 | 580+ | 24.07 | 2.21.5 | **Working** — our current stack |

> **Help wanted:** If you've tested other version combinations, please submit a PR or issue with your results.

---

## Rollback

If an update breaks something:

### CUDA Rollback

```bash
# List installed CUDA versions
ls /usr/local/ | grep cuda

# Switch symlink to previous version
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.4 /usr/local/cuda
```

### OFED Rollback

```bash
# Uninstall current
sudo /usr/local/sbin/ofed_uninstall.sh

# Reinstall previous version from saved package
sudo ./mlnxofedinstall --force  # Point to old version
sudo reboot
```

### Nuclear Option

If everything is broken and you can't recover:

```bash
# Reinstall DGX OS from NVIDIA's recovery image
# This resets everything to factory defaults
# You'll need to reconfigure: netplan, SSH keys, mounts, NCCL env
```

Keep your config files backed up (they're in this repo under `configs/`).

---

## Maintenance Schedule

| Task | Frequency | Command |
|---|---|---|
| Check NVMe health | Monthly | `sudo nvme smart-log /dev/nvme1n1` |
| Check GPU health | Monthly | `nvidia-smi -q \| grep -A5 "Gpu\|ECC\|Retired"` |
| Check OFED status | After reboot | `ibdev2netdev && ibstat mlx5_0` |
| Security updates | Monthly | `sudo apt update && sudo apt upgrade` (non-CUDA packages) |
| CUDA/OFED updates | Only when needed | Follow safe update process above |
| Version snapshot | Before any update | Save to `~/version_snapshots.log` |
