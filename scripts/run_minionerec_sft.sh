#!/bin/bash
# MiniOneRec SFT training via verl FSDP2 SFTTrainer.
#
# Usage:
#   bash scripts/run_minionerec_sft.sh [category] [base_model] [output_dir]
#
# Defaults:
#   category   = Industrial_and_Scientific
#   base_model = /path/to/Qwen2-0.5B
#   output_dir = ./outputs/minionerec_sft_<category>

set -euo pipefail

CATEGORY="${1:-Industrial_and_Scientific}"
BASE_MODEL="${2:-/path/to/Qwen2-0.5B}"
OUTPUT_DIR="${3:-./outputs/minionerec_sft_${CATEGORY}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/../MiniOneRec/data/Amazon"

TRAIN_FILE="${DATA_ROOT}/train/${CATEGORY}_5_*.csv"
EVAL_FILE="${DATA_ROOT}/valid/${CATEGORY}_5_*.csv"
SID_INDEX="${DATA_ROOT}/index/${CATEGORY}.index.json"
ITEM_META="${DATA_ROOT}/index/${CATEGORY}.item.json"

echo "============================================"
echo "MiniOneRec SFT Training"
echo "  category      = ${CATEGORY}"
echo "  train         = ${TRAIN_FILE}"
echo "  eval          = ${EVAL_FILE}"
echo "  base model    = ${BASE_MODEL}"
echo "  output        = ${OUTPUT_DIR}"
echo "============================================"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl_gr.trainers.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${EVAL_FILE}" \
    data.train_batch_size=256 \
    data.max_length=512 \
    data.micro_batch_size_per_gpu=4 \
    data.pad_mode=left_right \
    data.custom_cls.path=verl_gr.recipes.minionerec.data.sft_dataset \
    data.custom_cls.name=MiniOneRecSFTDataset \
    +data.sid_index_path="${SID_INDEX}" \
    +data.item_meta_path="${ITEM_META}" \
    +data.category="${CATEGORY}" \
    +data.include_alignment_tasks=true \
    model.path="${BASE_MODEL}" \
    model.use_remove_padding=true \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    engine.strategy=fsdp2 \
    engine.wrap_policy.min_num_params=0 \
    optim.lr=3e-4 \
    optim.lr_scheduler_type=cosine \
    optim.warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.0 \
    optim.weight_decay=0.01 \
    trainer.total_epochs=10 \
    trainer.project_name=MiniOneRec_SFT \
    trainer.experiment_name="minionerec_sft_${CATEGORY}_$(date +%Y%m%d_%H%M%S)" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.logger='[console,wandb]' \
    trainer.save_freq=500 \
    trainer.test_freq=100
