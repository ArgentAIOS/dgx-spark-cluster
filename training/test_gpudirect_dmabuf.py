#!/usr/bin/env python3
"""
Test GPU Direct RDMA with DMA-BUF support
For use with NVIDIA open driver on DGX Spark
"""

import os
import sys
import torch
import torch.distributed as dist

def check_gpudirect_support():
    """Check if GPU Direct RDMA is available"""
    print("=" * 60)
    print("GPU Direct RDMA (DMA-BUF) Detection Test")
    print("=" * 60)
    print()
    
    # Check CUDA availability
    print("1. CUDA Status:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Check NCCL version
    print("2. NCCL Status:")
    try:
        nccl_version = torch.cuda.nccl.version()
        print(f"   NCCL version: {nccl_version[0]}.{nccl_version[1]}.{nccl_version[2]}")
    except:
        print("   NCCL version: Unable to determine")
    print()
    
    # Check environment variables
    print("3. RDMA Environment Variables:")
    rdma_vars = [
        'NCCL_IB_DISABLE',
        'NCCL_NET',
        'NCCL_IB_GID_INDEX',
        'NCCL_NET_GDR_LEVEL',
        'NCCL_SOCKET_IFNAME',
    ]
    for var in rdma_vars:
        value = os.environ.get(var, 'not set')
        print(f"   {var}: {value}")
    print()
    
    # Check RDMA devices
    print("4. RDMA Devices:")
    try:
        import subprocess
        result = subprocess.run(['ibv_devinfo'], capture_output=True, text=True)
        if result.returncode == 0:
            # Count active ports
            active_ports = result.stdout.count('PORT_ACTIVE')
            print(f"   Active RDMA ports: {active_ports}")
            
            # Check for RoCE
            if 'Ethernet' in result.stdout:
                print(f"   Transport: RoCE (RDMA over Ethernet)")
            elif 'InfiniBand' in result.stdout:
                print(f"   Transport: Native InfiniBand")
        else:
            print("   Unable to query RDMA devices")
    except:
        print("   ibv_devinfo not available")
    print()
    
    # Check kernel DMA-BUF support
    print("5. Kernel DMA-BUF Support:")
    try:
        import subprocess
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
        kernel = result.stdout.strip()
        print(f"   Kernel: {kernel}")
        
        # Check if DMA-BUF is enabled
        config_file = f'/boot/config-{kernel}'
        try:
            with open(config_file, 'r') as f:
                config = f.read()
                if 'CONFIG_DMA_SHARED_BUFFER=y' in config:
                    print(f"   DMA-BUF: ✓ Enabled")
                else:
                    print(f"   DMA-BUF: ? Unknown")
        except:
            print(f"   DMA-BUF: Unable to check")
    except:
        print("   Unable to check kernel config")
    print()
    
    print("6. NCCL GPU Direct Detection:")
    print("   Will be shown in NCCL debug output during distributed training.")
    print("   Look for messages like:")
    print("     - 'NET/IB : Using [0]mlx5_0:1 [RoCE]'")
    print("     - 'NCCL INFO NET/IB: Using GPU Direct RDMA'")
    print("     - 'Using DMA-BUF for GPU memory'")
    print()
    
    print("=" * 60)
    print("To test with actual training:")
    print("  1. Set NCCL_DEBUG=INFO")
    print("  2. Run distributed training")
    print("  3. Check NCCL output for GPU Direct messages")
    print("=" * 60)

if __name__ == "__main__":
    check_gpudirect_support()
