#!/bin/bash
# ============================================================================
# DGX Spark Cluster Launcher
# ============================================================================
# Single command that handles everything:
#   1. SSHes to node 1 and verifies connectivity
#   2. Clears stale torchrun processes + ports on BOTH nodes
#   3. Sources NCCL environment (safe or dmabuf mode)
#   4. Starts node 1 in the background via SSH
#   5. Runs node 0 in the foreground
#   6. Waits for both to finish and reports status
#
# Usage:
#   ./scripts/launch_cluster.sh <python_script> [script args...]
#   NCCL_MODE=dmabuf ./scripts/launch_cluster.sh <python_script> [args]
#
# Examples:
#   ./scripts/launch_cluster.sh training/validate_distributed.py
#   ./scripts/launch_cluster.sh training/train_pipeline.py \
#       --model meta-llama/Llama-3.2-3B-Instruct \
#       --train-data /mnt/rosa-storage/data/train.jsonl \
#       --output-dir /mnt/rosa-models/my-model-v1 \
#       --epochs 3
#   NCCL_MODE=dmabuf ./scripts/launch_cluster.sh training/train_pipeline.py [args]
# ============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_script> [script args...]"
    echo ""
    echo "Single command to launch distributed training on both DGX Sparks."
    echo "Must be run from spark-dgx-1 (10.0.0.1)."
    echo ""
    echo "Environment:"
    echo "  NCCL_MODE=safe|dmabuf   NCCL transport mode (default: safe)"
    echo "  REMOTE_USER=sem         SSH user for spark-dgx-2"
    echo "  REMOTE_HOST=10.0.0.2    spark-dgx-2 fabric address"
    exit 1
fi

TRAINING_SCRIPT=$1
shift
SCRIPT_ARGS="$@"

# ── Configuration ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NCCL_MODE="${NCCL_MODE:-safe}"
REMOTE_USER="${REMOTE_USER:-sem}"
REMOTE_HOST="${REMOTE_HOST:-10.0.0.2}"
MASTER_ADDR="10.0.0.1"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=1

# ── Preflight checks ────────────────────────────────────────────────────
echo ""
echo "┌──────────────────────────────────────────────────────────┐"
echo "│  DGX Spark Cluster Launcher                              │"
echo "└──────────────────────────────────────────────────────────┘"

# Check we're on spark-dgx-1
LOCAL_IP=$(ip -4 addr show enp1s0f0np0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || true)
if [ "$LOCAL_IP" != "10.0.0.1" ]; then
    echo "  ERROR: Must run from spark-dgx-1 (10.0.0.1)"
    echo "         Detected: ${LOCAL_IP:-interface not found}"
    exit 1
fi
echo "  [ok] Running on spark-dgx-1 (10.0.0.1)"

# Check training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "  ERROR: Script not found: $TRAINING_SCRIPT"
    exit 1
fi
echo "  [ok] Script: $TRAINING_SCRIPT"

# Check SSH connectivity
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" "true" 2>/dev/null; then
    echo "  ERROR: Cannot SSH to ${REMOTE_USER}@${REMOTE_HOST}"
    echo "         Run: ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}"
    exit 1
fi
echo "  [ok] SSH to spark-dgx-2 (${REMOTE_HOST})"

# ── Clean up stale processes on BOTH nodes ───────────────────────────────
echo ""
echo "  Cleaning stale processes..."

cleanup_node() {
    local host=$1
    local label=$2
    local cmd="pkill -f 'torch.distributed.run' 2>/dev/null; \
               pkill -f 'torchrun' 2>/dev/null; \
               sleep 0.5; \
               if ss -tuln 2>/dev/null | grep -q ':${MASTER_PORT} '; then \
                   fuser -k ${MASTER_PORT}/tcp 2>/dev/null; \
               fi; \
               true"

    if [ "$host" = "local" ]; then
        eval "$cmd"
    else
        ssh -o ConnectTimeout=5 "${REMOTE_USER}@${host}" "$cmd" 2>/dev/null
    fi
    echo "  [ok] $label — cleared"
}

cleanup_node "local" "spark-dgx-1"
cleanup_node "$REMOTE_HOST" "spark-dgx-2"

# ── Source NCCL environment ──────────────────────────────────────────────
echo ""
source "${REPO_DIR}/configs/nccl-env.sh" "$NCCL_MODE"

# ── Build NCCL env export string for remote node ────────────────────────
NCCL_EXPORTS=""
for var in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME UCX_NET_DEVICES \
           NCCL_IB_DISABLE NCCL_IB_HCA NCCL_NET NCCL_IB_GID_INDEX \
           NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ NCCL_DMABUF_ENABLE \
           NCCL_IB_TIMEOUT NCCL_IB_RETRY_CNT NCCL_IB_QPS_PER_CONNECTION \
           NCCL_IB_TC NCCL_TIMEOUT NCCL_ALGO NCCL_PROTO NCCL_DEBUG; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        NCCL_EXPORTS+="export ${var}='${val}'; "
    fi
done

# ── Print launch summary ────────────────────────────────────────────────
echo ""
echo "  ┌────────────────────────────────────────────────────────┐"
echo "  │  Launching distributed training                        │"
echo "  │                                                        │"
echo "  │  Script:  $TRAINING_SCRIPT"
[ -n "$SCRIPT_ARGS" ] && \
echo "  │  Args:    $SCRIPT_ARGS"
echo "  │  Mode:    $NCCL_MODE"
echo "  │  Master:  $MASTER_ADDR:$MASTER_PORT"
echo "  │  Nodes:   $NNODES"
echo "  └────────────────────────────────────────────────────────┘"
echo ""

# ── Torchrun command ─────────────────────────────────────────────────────
TORCHRUN="python3 -m torch.distributed.run \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT"

# ── Launch Node 1 (spark-dgx-2) via SSH in background ───────────────────
echo "[cluster] Starting rank 1 on spark-dgx-2..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" \
    "cd ${REPO_DIR} && ${NCCL_EXPORTS} ${TORCHRUN} --node_rank=1 ${TRAINING_SCRIPT} ${SCRIPT_ARGS}" \
    2>&1 | sed 's/^/[rank 1] /' &
REMOTE_PID=$!

# Give remote node time to bind
sleep 3

# ── Launch Node 0 (spark-dgx-1) locally ─────────────────────────────────
echo "[cluster] Starting rank 0 on spark-dgx-1..."
echo ""
${TORCHRUN} --node_rank=0 ${TRAINING_SCRIPT} ${SCRIPT_ARGS} 2>&1 | sed 's/^/[rank 0] /'
LOCAL_EXIT=$?

# ── Wait for remote node ────────────────────────────────────────────────
wait $REMOTE_PID 2>/dev/null
REMOTE_EXIT=$?

# ── Report ───────────────────────────────────────────────────────────────
echo ""
echo "┌──────────────────────────────────────────────────────────┐"
if [ $LOCAL_EXIT -eq 0 ] && [ $REMOTE_EXIT -eq 0 ]; then
    echo "│  COMPLETED — Both nodes finished successfully           │"
else
    echo "│  FINISHED WITH ERRORS                                   │"
    echo "│  Rank 0 (spark-dgx-1): exit code $LOCAL_EXIT"
    echo "│  Rank 1 (spark-dgx-2): exit code $REMOTE_EXIT"
fi
echo "└──────────────────────────────────────────────────────────┘"

exit $(( LOCAL_EXIT + REMOTE_EXIT ))
