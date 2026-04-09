#!/bin/bash

# Generate timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/sem/training-benchmarks/results/distributed_rosa_${TIMESTAMP}"

echo "========================================================================"
echo "Manual Distributed Training - ROSA"
echo "========================================================================"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "INSTRUCTIONS:"
echo "1. Start Node 0 (master) on spark-dgx-1 FIRST"
echo "2. Wait 10 seconds for it to initialize"
echo "3. Then start Node 1 on spark-dgx-2"
echo ""
echo "========================================================================"
echo "NODE 0 COMMAND (run on spark-dgx-1):"
echo "========================================================================"
echo ""
cat << 'NODE0'
torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=spark-dgx-1 \
  --master_port=29500 \
  /home/sem/training-benchmarks/benchmark_train.py \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --train-data /mnt/rosa-storage/training-benchmarks/train_benchmark.jsonl \
  --val-data /mnt/rosa-storage/training-benchmarks/val_benchmark.jsonl \
  --output-dir OUTPUT_DIR_PLACEHOLDER \
  --max-steps 50 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-seq-length 2048
NODE0

echo ""
echo "========================================================================"
echo "NODE 1 COMMAND (run on spark-dgx-2):"
echo "========================================================================"
echo ""
cat << 'NODE1'
torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=spark-dgx-1 \
  --master_port=29500 \
  /home/sem/training-benchmarks/benchmark_train.py \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --train-data /mnt/rosa-storage/training-benchmarks/train_benchmark.jsonl \
  --val-data /mnt/rosa-storage/training-benchmarks/val_benchmark.jsonl \
  --output-dir OUTPUT_DIR_PLACEHOLDER \
  --max-steps 50 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-seq-length 2048
NODE1

# Replace placeholder with actual output dir
sed -i "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" /tmp/node0_cmd.sh 2>/dev/null || true
sed -i "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" /tmp/node1_cmd.sh 2>/dev/null || true

echo ""
echo "========================================================================"
echo ""
echo "After both nodes complete, check results at:"
echo "  ${OUTPUT_DIR}/benchmark_metrics.json"
echo ""
