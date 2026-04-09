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
