#!/bin/bash
# RL data splitting script: Merge multiple RL task datasets and split into training and test sets
# Borrowed from https://github.com/Kuaishou-OneRec/OpenOneRec/blob/main/data/prepare_rl.sh, with modifications

set -e

# ============== Configuration ==============
# onerec dataset output path, rl uses datasets starting with sft
REC_DATA_PATH="../output"

# Tasks that RL depends on
# VIDEO_REC=${REC_DATA_PATH}/sft_video_rec.parquet
# AD_REC=${REC_DATA_PATH}/sft_ad_rec.parquet
PRODUCT_REC=${REC_DATA_PATH}/sft_product_rec.parquet
# INTERACTIVE_REC=${REC_DATA_PATH}/sft_interactive_rec.parquet
# LABEL_COND_REC=${REC_DATA_PATH}/sft_label_cond_rec.parquet

# Output configuration
OUTPUT_DIR="../output/rl_data"
TEST_SIZE=1000
SEED=42
ENGINE="pyarrow"

# ============== Main ==============
# Define all task files to process
declare -a TASK_FILES=(
    "${VIDEO_REC}"
    "${AD_REC}"
    "${PRODUCT_REC}"
    "${INTERACTIVE_REC}"
    "${LABEL_COND_REC}"
)

# Check if input files exist
echo "Checking input files..."
MISSING_FILES=0
for file in "${TASK_FILES[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "Warning: File does not exist: ${file}"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ ${MISSING_FILES} -eq ${#TASK_FILES[@]} ]; then
    echo "Error: All input files do not exist"
    exit 1
fi

# Execute train_test_split, merge all files and process them together
echo ""
echo "Starting RL data splitting..."
echo "=========================================="
echo "Input files:"
for file in "${TASK_FILES[@]}"; do
    if [ -f "${file}" ]; then
        echo "  - ${file}"
    fi
done
echo "Output directory: ${OUTPUT_DIR}"
echo "Test set size: ${TEST_SIZE}"
echo "=========================================="

python3 train_test_split.py \
    --input_files "${TASK_FILES[@]}" \
    --test_size "${TEST_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --engine "${ENGINE}" \
    --test_filename "test.parquet" \
    --train_filename "train.parquet"

echo ""
echo "=========================================="
echo "RL data processing completed!"
echo "Output directory: ${OUTPUT_DIR}"
echo "  - train.parquet (training set)"
echo "  - test.parquet (test set)"
echo "=========================================="
