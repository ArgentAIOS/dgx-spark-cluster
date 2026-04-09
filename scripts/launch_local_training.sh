#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "Local Storage Training (No Network Mount)"
echo "============================================================"

# Configuration - ALL LOCAL
LOCAL_DIR="/home/sem/personal-injury-llm-local"
OUTPUT_DIR="${LOCAL_DIR}/models/comprehensive-v3"
TRAIN_DATA="${LOCAL_DIR}/train_comprehensive.jsonl"
VAL_DATA="${LOCAL_DIR}/validation_comprehensive.jsonl"

# Source data from mounted share (for copying)
SOURCE_TRAIN="/mnt/dell-shared/training-data/pi-final/train_comprehensive.jsonl"
SOURCE_VAL="/mnt/dell-shared/training-data/pi-final/validation_comprehensive.jsonl"
SOURCE_SCRIPT="/mnt/dell-shared/personal-injury-llm/scripts/06_train_comprehensive.py"

# Hyperparameters (same as previous tests)
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=2e-4
LORA_RANK=64
LORA_ALPHA=128
MAX_SEQ_LEN=4096
WARMUP_STEPS=50
SAVE_STEPS=50

echo ""
echo "Setting up local directory structure..."
mkdir -p "${LOCAL_DIR}/models"
mkdir -p "${LOCAL_DIR}/scripts"
mkdir -p "${LOCAL_DIR}/.cache/huggingface"

# Copy training data locally if not already there
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Copying training data to local storage..."
    cp "${SOURCE_TRAIN}" "${TRAIN_DATA}"
    echo -e "${GREEN}✓${NC} Training data copied"
else
    echo -e "${GREEN}✓${NC} Training data already local"
fi

if [ ! -f "${VAL_DATA}" ]; then
    echo "Copying validation data to local storage..."
    cp "${SOURCE_VAL}" "${VAL_DATA}"
    echo -e "${GREEN}✓${NC} Validation data copied"
else
    echo -e "${GREEN}✓${NC} Validation data already local"
fi

# Copy training script
if [ ! -f "${LOCAL_DIR}/scripts/06_train_comprehensive.py" ]; then
    echo "Copying training script to local storage..."
    cp "${SOURCE_SCRIPT}" "${LOCAL_DIR}/scripts/"
    echo -e "${GREEN}✓${NC} Training script copied"
else
    echo -e "${GREEN}✓${NC} Training script already local"
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  Output: ${OUTPUT_DIR}"
echo "  Train Data: ${TRAIN_DATA}"
echo "  Val Data: ${VAL_DATA}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Gradient Accumulation: ${GRAD_ACCUM}"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Learning Rate: ${LR}"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  LoRA Alpha: ${LORA_ALPHA}"
echo "  Max Seq Length: ${MAX_SEQ_LEN}"
echo ""

# Environment variables - ALL LOCAL
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${LOCAL_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${LOCAL_DIR}/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="${LOCAL_DIR}/.cache/huggingface/transformers"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Starting Local Storage Training (100% Local)"
echo "============================================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Launch training
python3 "${LOCAL_DIR}/scripts/06_train_comprehensive.py" \
    --output_dir "${OUTPUT_DIR}" \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --warmup_steps ${WARMUP_STEPS} \
    --save_steps ${SAVE_STEPS}

EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Training completed successfully!"
    echo "  Duration: ${MINUTES}m ${SECONDS}s"
    echo "  Model saved to: ${OUTPUT_DIR}"
else
    echo -e "${RED}✗${NC} Training failed with exit code ${EXIT_CODE}"
fi
echo "============================================================"

exit ${EXIT_CODE}
