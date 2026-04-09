# DGX Spark Network Setup Prerequisites

**Environment:** 2x NVIDIA DGX Spark (Grace Blackwell GB200 NVL32)  
**Purpose:** Establish 200 Gbps InfiniBand connectivity for distributed training

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Network Configuration Overview](#network-configuration-overview)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [Verification](#verification)
5. [Common Issues](#common-issues)
6. [Next Steps](#next-steps)

---

## Hardware Requirements

### Physical Setup

- **2x DGX Spark systems** with InfiniBand NICs (ConnectX-7)
- **InfiniBand cable** (QSFP112 200G, direct attach or optical)
- **Direct connection** between nodes (no switch required for 2-node setup)

### Software Requirements

- **OS:** Ubuntu 22.04 LTS (or later)
- **NVIDIA Driver:** Latest Grace Blackwell drivers
- **CUDA:** 12.4 or later
- **PyTorch:** 2.0+ with NCCL support
- **Mellanox OFED:** OpenFabrics drivers for InfiniBand

---

## Network Configuration Overview

### Architecture

```
┌─────────────────────────────────────┐
│         spark-dgx-1 (Node 0)        │
│                                     │
│  InfiniBand NIC (ConnectX-7)       │
│  mlx5_0 / enp1s0f0np0              │
│  IP: 10.0.0.1                      │
└──────────────┬──────────────────────┘
               │
               │ 200 Gbps InfiniBand Cable
               │
┌──────────────┴──────────────────────┐
│         spark-dgx-2 (Node 1)        │
│                                     │
│  InfiniBand NIC (ConnectX-7)       │
│  mlx5_0 / enp1s0f0np0              │
│  IP: 10.0.0.2                      │
└─────────────────────────────────────┘
```

### IP Addressing Scheme

| Node        | Hostname     | IB Interface    | IP Address | Role   |
|-------------|--------------|-----------------|------------|--------|
| Node 0      | spark-dgx-1  | enp1s0f0np0     | 10.0.0.1   | Master |
| Node 1      | spark-dgx-2  | enp1s0f0np0     | 10.0.0.2   | Worker |

---

## Step-by-Step Setup

### 1. Verify InfiniBand Hardware Detection

**On both nodes:**

```bash
# Check if InfiniBand devices are detected
lspci | grep -i mellanox

# Expected output (ConnectX-7):
# 01:00.0 Infiniband controller: Mellanox Technologies MT2910 Family [ConnectX-7]

# List InfiniBand devices
ibstat

# Expected output:
# CA 'mlx5_0'
#     CA type: MT4129
#     Number of ports: 1
#     Firmware version: XX.XX.XXXX
#     Hardware version: 0
#     Port 1:
#         State: Active
#         Physical state: LinkUp
#         Rate: 200
#         Base lid: 0
#         LMC: 0
#         SM lid: 0
#         Capability mask: 0x00010000
#         Port GUID: 0x...
```

**If InfiniBand devices are not detected:**

```bash
# Check if drivers are loaded
lsmod | grep mlx5

# If not loaded, install Mellanox OFED
wget https://content.mellanox.com/ofed/MLNX_OFED-latest/MLNX_OFED_LINUX-latest-ubuntu22.04-aarch64.tgz
tar -xzf MLNX_OFED_LINUX-latest-ubuntu22.04-aarch64.tgz
cd MLNX_OFED_LINUX-*
sudo ./mlnxofedinstall --force
sudo /etc/init.d/openibd restart
```

### 2. Configure Network Interfaces

**On Node 0 (spark-dgx-1):**

```bash
# Identify the InfiniBand interface name
ip link show | grep -E "enp|ib"

# Configure static IP (temporary)
sudo ip addr add 10.0.0.1/24 dev enp1s0f0np0
sudo ip link set enp1s0f0np0 up

# Make permanent using netplan
sudo nano /etc/netplan/02-infiniband.yaml
```

**Add the following to `/etc/netplan/02-infiniband.yaml` on Node 0:**

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      addresses:
        - 10.0.0.1/24
```

**On Node 1 (spark-dgx-2):**

```bash
# Configure static IP (temporary)
sudo ip addr add 10.0.0.2/24 dev enp1s0f0np0
sudo ip link set enp1s0f0np0 up

# Make permanent using netplan
sudo nano /etc/netplan/02-infiniband.yaml
```

**Add the following to `/etc/netplan/02-infiniband.yaml` on Node 1:**

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      addresses:
        - 10.0.0.2/24
```

**Apply netplan configuration on both nodes:**

```bash
sudo netplan apply
```

### 3. Verify Network Device Mapping

**On both nodes:**

```bash
# Map InfiniBand devices to network interfaces
ibdev2netdev

# Expected output:
# mlx5_0 port 1 ==> enp1s0f0np0 (Up)
```

### 4. Test Basic Connectivity

**From Node 0:**

```bash
# Ping Node 1
ping -c 5 10.0.0.2

# Expected output:
# 5 packets transmitted, 5 received, 0% packet loss
# rtt min/avg/max/mdev = 0.1/0.15/0.2/0.05 ms
```

**From Node 1:**

```bash
# Ping Node 0
ping -c 5 10.0.0.1
```

### 5. Configure Hostnames Resolution

**On both nodes, add entries to `/etc/hosts`:**

```bash
sudo nano /etc/hosts
```

**Add these lines:**

```
10.0.0.1    spark-dgx-1
10.0.0.2    spark-dgx-2
```

**Test hostname resolution:**

```bash
# From Node 0
ping -c 3 spark-dgx-2

# From Node 1
ping -c 3 spark-dgx-1
```

### 6. Set Up Passwordless SSH

**On Node 0:**

```bash
# Generate SSH key if not already present
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
fi

# Copy public key to Node 1
ssh-copy-id sem@10.0.0.2

# Test passwordless SSH
ssh 10.0.0.2 hostname
# Should print: spark-dgx-2
```

**On Node 1:**

```bash
# Generate SSH key if not already present
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
fi

# Copy public key to Node 0
ssh-copy-id sem@10.0.0.1

# Test passwordless SSH
ssh 10.0.0.1 hostname
# Should print: spark-dgx-1
```

### 7. Configure Firewall (if enabled)

**On both nodes:**

```bash
# Check if firewall is active
sudo ufw status

# If active, allow necessary ports
sudo ufw allow from 10.0.0.0/24  # Allow all traffic from cluster subnet
sudo ufw allow 29500/tcp          # PyTorch distributed master port
sudo ufw allow 22/tcp             # SSH
sudo ufw reload
```

### 8. Optimize InfiniBand Configuration

**On both nodes:**

```bash
# Set MTU to 4096 (optimal for InfiniBand)
sudo ip link set enp1s0f0np0 mtu 4096

# Make permanent in netplan
sudo nano /etc/netplan/02-infiniband.yaml
```

**Update netplan configuration to include MTU:**

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: no
      addresses:
        - 10.0.0.1/24  # or 10.0.0.2/24 for Node 1
      mtu: 4096
```

```bash
sudo netplan apply
```

### 9. Verify InfiniBand Performance

**Check link speed on both nodes:**

```bash
# Should show 200 Gbps
ibstat mlx5_0 | grep Rate

# Expected output:
# Rate: 200
```

**Test RDMA bandwidth (optional, requires perftest package):**

```bash
# Install perftest if not present
sudo apt-get update
sudo apt-get install -y perftest

# On Node 1 (server):
ib_write_bw -d mlx5_0 -a

# On Node 0 (client):
ib_write_bw -d mlx5_0 -a 10.0.0.2

# Expected bandwidth: ~20-25 GB/s
```

### 10. Install Python Environment and Dependencies

**On both nodes:**

```bash
# Verify Python 3 is installed
python3 --version

# Install pip if not present
sudo apt-get install -y python3-pip

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'NCCL available: {torch.distributed.is_nccl_available()}')"

# Expected output:
# PyTorch: 2.x.x
# CUDA available: True
# NCCL available: True
```

---

## Verification

### Complete Pre-Flight Checklist

Run this script on **both nodes** to verify setup:

```bash
#!/bin/bash
echo "=========================================="
echo "DGX Spark Network Setup Verification"
echo "=========================================="
echo ""

# 1. InfiniBand device check
echo "1. Checking InfiniBand devices..."
if ibstat mlx5_0 &>/dev/null; then
    echo "   ✓ mlx5_0 found"
    ibstat mlx5_0 | grep -E "State|Rate"
else
    echo "   ✗ mlx5_0 not found"
fi
echo ""

# 2. Network interface check
echo "2. Checking network interface..."
if ip link show enp1s0f0np0 &>/dev/null; then
    echo "   ✓ enp1s0f0np0 exists"
    ip addr show enp1s0f0np0 | grep "inet "
else
    echo "   ✗ enp1s0f0np0 not found"
fi
echo ""

# 3. Connectivity check
echo "3. Checking connectivity..."
if [ "$(hostname)" = "spark-dgx-1" ]; then
    if ping -c 1 -W 1 10.0.0.2 &>/dev/null; then
        echo "   ✓ Can reach Node 1 (10.0.0.2)"
    else
        echo "   ✗ Cannot reach Node 1 (10.0.0.2)"
    fi
else
    if ping -c 1 -W 1 10.0.0.1 &>/dev/null; then
        echo "   ✓ Can reach Node 0 (10.0.0.1)"
    else
        echo "   ✗ Cannot reach Node 0 (10.0.0.1)"
    fi
fi
echo ""

# 4. SSH check
echo "4. Checking passwordless SSH..."
if [ "$(hostname)" = "spark-dgx-1" ]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=2 10.0.0.2 hostname &>/dev/null; then
        echo "   ✓ Passwordless SSH to Node 1 works"
    else
        echo "   ✗ Passwordless SSH to Node 1 failed"
    fi
else
    if ssh -o BatchMode=yes -o ConnectTimeout=2 10.0.0.1 hostname &>/dev/null; then
        echo "   ✓ Passwordless SSH to Node 0 works"
    else
        echo "   ✗ Passwordless SSH to Node 0 failed"
    fi
fi
echo ""

# 5. PyTorch/CUDA check
echo "5. Checking PyTorch and CUDA..."
python3 -c "import torch; print(f'   ✓ PyTorch {torch.__version__}'); print(f'   ✓ CUDA available: {torch.cuda.is_available()}'); print(f'   ✓ NCCL available: {torch.distributed.is_nccl_available()}')" 2>/dev/null || echo "   ✗ PyTorch check failed"
echo ""

# 6. NCCL device check
echo "6. Checking NCCL device mapping..."
if ibdev2netdev | grep -q mlx5_0; then
    echo "   ✓ NCCL can detect InfiniBand device"
    ibdev2netdev
else
    echo "   ✗ NCCL cannot detect InfiniBand device"
fi
echo ""

echo "=========================================="
echo "Verification complete"
echo "=========================================="
```

**Save as `verify_network_setup.sh` and run:**

```bash
chmod +x verify_network_setup.sh
./verify_network_setup.sh
```

---

## Common Issues

### Issue 1: "No InfiniBand devices found"

**Symptoms:**
```bash
ibstat: No InfiniBand devices found
```

**Solutions:**

1. Check if drivers are loaded:
   ```bash
   lsmod | grep mlx5_core
   ```

2. Reload drivers:
   ```bash
   sudo modprobe -r mlx5_ib mlx5_core
   sudo modprobe mlx5_core
   sudo modprobe mlx5_ib
   ```

3. Reinstall Mellanox OFED (see Step 1 above)

### Issue 2: "Network interface not found"

**Symptoms:**
```bash
ip: link 'enp1s0f0np0' not found
```

**Solutions:**

1. List all network interfaces:
   ```bash
   ip link show
   ```

2. Your interface might have a different name. Look for patterns like:
   - `ibp*` (older naming)
   - `enp*` (predictable naming)
   - `ib*` (IPoIB)

3. Update scripts with correct interface name

### Issue 3: "Port is not active"

**Symptoms:**
```bash
ibstat mlx5_0
# Shows: State: Down
```

**Solutions:**

1. Check physical cable connection
2. Verify cable is properly seated
3. Try a different cable
4. Check if port is disabled in firmware:
   ```bash
   sudo mst start
   sudo mlxconfig -d /dev/mst/mt4129_pciconf0 query | grep LINK_TYPE
   ```

### Issue 4: "Cannot ping other node"

**Symptoms:**
```bash
ping 10.0.0.2
# Destination Host Unreachable
```

**Solutions:**

1. Verify IP is assigned:
   ```bash
   ip addr show enp1s0f0np0
   ```

2. Check routing table:
   ```bash
   ip route
   ```

3. Temporarily disable firewall:
   ```bash
   sudo ufw disable
   ping 10.0.0.2
   sudo ufw enable
   ```

4. Verify both nodes are on same subnet (10.0.0.0/24)

### Issue 5: "SSH requires password every time"

**Symptoms:**
```bash
ssh 10.0.0.2
# Prompts for password
```

**Solutions:**

1. Check SSH key permissions:
   ```bash
   chmod 700 ~/.ssh
   chmod 600 ~/.ssh/id_rsa
   chmod 644 ~/.ssh/id_rsa.pub
   chmod 644 ~/.ssh/authorized_keys
   ```

2. Verify public key is in authorized_keys on remote node:
   ```bash
   ssh 10.0.0.2 "cat ~/.ssh/authorized_keys"
   # Should contain your public key
   ```

3. Check SSH config allows key authentication:
   ```bash
   grep -E "PubkeyAuthentication|PasswordAuthentication" /etc/ssh/sshd_config
   # Should show: PubkeyAuthentication yes
   ```

---

## Next Steps

Once network setup is complete:

1. ✅ **Verify all checklist items pass** on both nodes
2. ✅ **Test basic distributed training** with validation script:
   ```bash
   # Node 0
   ./distributed_train.sh 0 validate_distributed.py
   
   # Node 1
   ./distributed_train.sh 1 validate_distributed.py
   ```

3. ✅ **Proceed to main guide:** `DGX_Spark_Distributed_Training_Guide.md`

---

## Quick Reference Commands

```bash
# Check InfiniBand status
ibstat mlx5_0
ibdev2netdev

# Check network configuration
ip addr show enp1s0f0np0
ip route

# Test connectivity
ping -c 3 10.0.0.2  # From Node 0
ping -c 3 10.0.0.1  # From Node 1

# Test SSH
ssh 10.0.0.2 hostname  # From Node 0
ssh 10.0.0.1 hostname  # From Node 1

# Check PyTorch/NCCL
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.distributed.is_nccl_available())"

# Monitor network traffic
watch -n 1 'ibstat mlx5_0 | grep -A 5 "Port 1"'
```

---

**Ready?** Once all prerequisites are met, proceed to the main distributed training guide!
