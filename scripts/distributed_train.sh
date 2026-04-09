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
