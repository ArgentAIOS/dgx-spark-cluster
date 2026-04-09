#!/bin/bash
# Single-command distributed launcher — fires both nodes from spark-dgx-1
# Usage: ./scripts/launch_distributed.sh <python_script> [script args...]
#
# Examples:
#   ./scripts/launch_distributed.sh training/validate_distributed.py
#   ./scripts/launch_distributed.sh training/benchmark_train.py --epochs 3 --lr 2e-4
#   NCCL_MODE=dmabuf ./scripts/launch_distributed.sh training/benchmark_train.py

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_script> [script args...]"
    echo ""
    echo "Launches distributed training on BOTH nodes from this machine."
    echo "Must be run from spark-dgx-1 (10.0.0.1)."
    echo ""
    echo "Options (via environment variables):"
    echo "  NCCL_MODE=safe|dmabuf   NCCL transport mode (default: safe)"
    echo "  REMOTE_USER=sem         SSH user for spark-dgx-2"
    echo "  REMOTE_HOST=10.0.0.2    spark-dgx-2 address"
    echo ""
    echo "Examples:"
    echo "  $0 training/validate_distributed.py"
    echo "  NCCL_MODE=dmabuf $0 training/benchmark_train.py --epochs 3"
    exit 1
fi

TRAINING_SCRIPT=$1
shift
SCRIPT_ARGS="$@"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NCCL_MODE="${NCCL_MODE:-safe}"
REMOTE_USER="${REMOTE_USER:-sem}"
REMOTE_HOST="${REMOTE_HOST:-10.0.0.2}"
MASTER_ADDR="10.0.0.1"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=1

# Verify we're on spark-dgx-1
LOCAL_IP=$(ip -4 addr show enp1s0f0np0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || true)
if [ "$LOCAL_IP" != "10.0.0.1" ]; then
    echo "ERROR: This script must be run from spark-dgx-1 (10.0.0.1)"
    echo "       Detected fabric IP: ${LOCAL_IP:-not found}"
    exit 1
fi

# Verify training script exists locally
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

# Verify SSH to spark-dgx-2
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" "true" 2>/dev/null; then
    echo "ERROR: Cannot SSH to ${REMOTE_USER}@${REMOTE_HOST}"
    echo "       Run: ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}"
    exit 1
fi

# Source NCCL environment
source "${REPO_DIR}/configs/nccl-env.sh" "$NCCL_MODE"

echo ""
echo "=========================================="
echo " DGX Spark Distributed Launcher"
echo "=========================================="
echo " Script:     $TRAINING_SCRIPT $SCRIPT_ARGS"
echo " NCCL mode:  $NCCL_MODE"
echo " Master:     $MASTER_ADDR:$MASTER_PORT"
echo " Nodes:      $NNODES"
echo "=========================================="
echo ""

# Build the torchrun command
TORCHRUN_CMD="python3 -m torch.distributed.run \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT"

# Build NCCL env export string for remote node
NCCL_EXPORTS=""
for var in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME UCX_NET_DEVICES \
           NCCL_IB_DISABLE NCCL_IB_HCA NCCL_NET NCCL_IB_GID_INDEX \
           NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ NCCL_DMABUF_ENABLE \
           NCCL_IB_TIMEOUT NCCL_IB_RETRY_CNT NCCL_IB_QPS_PER_CONNECTION \
           NCCL_IB_TC NCCL_TIMEOUT NCCL_ALGO NCCL_PROTO NCCL_DEBUG; do
    val="${!var}"
    if [ -n "$val" ]; then
        NCCL_EXPORTS+="export ${var}=${val}; "
    fi
done

# Launch Node 1 (spark-dgx-2) via SSH in background
echo "[launcher] Starting rank 1 on ${REMOTE_HOST}..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" \
    "cd ${REPO_DIR} && ${NCCL_EXPORTS} ${TORCHRUN_CMD} --node_rank=1 ${TRAINING_SCRIPT} ${SCRIPT_ARGS}" \
    2>&1 | sed 's/^/[rank 1] /' &
REMOTE_PID=$!

# Give remote node a head start for port binding
sleep 2

# Launch Node 0 (spark-dgx-1) locally
echo "[launcher] Starting rank 0 locally..."
${TORCHRUN_CMD} --node_rank=0 ${TRAINING_SCRIPT} ${SCRIPT_ARGS} \
    2>&1 | sed 's/^/[rank 0] /'
LOCAL_EXIT=$?

# Wait for remote node
wait $REMOTE_PID 2>/dev/null
REMOTE_EXIT=$?

echo ""
echo "=========================================="
if [ $LOCAL_EXIT -eq 0 ] && [ $REMOTE_EXIT -eq 0 ]; then
    echo " All nodes completed successfully"
else
    echo " Rank 0 exit: $LOCAL_EXIT"
    echo " Rank 1 exit: $REMOTE_EXIT"
fi
echo "=========================================="

exit $(( LOCAL_EXIT + REMOTE_EXIT ))
