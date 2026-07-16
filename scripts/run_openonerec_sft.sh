#!/bin/bash
# OpenOneRec SFT training via verl FSDP2 SFTTrainer.
#
# Data should already be preprocessed into parquet with a ``messages``
# column by ``verl_gr.recipes.openonerec.data.sft.product_rec``.
#
# Usage:
#   bash scripts/run_onerec_sft.sh [train_parquet] [val_parquet] [model_path] [output_dir]

set -euo pipefail

TRAIN_FILE="${1:-/path/to/sft/train.parquet}"
EVAL_FILE="${2:-/path/to/sft/val.parquet}"
MODEL_PATH="${3:-/path/to/Qwen2-7B}"
OUTPUT_DIR="${4:-./outputs/onerec_sft}"

echo "============================================"
echo "OpenOneRec SFT Training"
echo "  train         = ${TRAIN_FILE}"
echo "  eval          = ${EVAL_FILE}"
echo "  model         = ${MODEL_PATH}"
echo "  output        = ${OUTPUT_DIR}"
echo "============================================"

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl_gr.trainers.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${EVAL_FILE}" \
    data.train_batch_size=256 \
    data.max_length=32768 \
    data.micro_batch_size_per_gpu=4 \
    data.pad_mode=no_padding \
    data.custom_cls.path=verl_gr.recipes.openonerec.data.sft_dataset \
    data.custom_cls.name=OneRecSFTDataset \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    model.enable_gradient_checkpointing=true \
    engine.strategy=fsdp2 \
    engine.ulysses_sequence_parallel_size=2 \
    optim.lr=2e-4 \
    optim.lr_scheduler_type=cosine \
    optim.min_lr_ratio=0.0 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    optim.warmup_steps_ratio=0.03 \
    trainer.total_epochs=4 \
    trainer.project_name=OpenOneRec_SFT \
    trainer.experiment_name="onerec_sft_$(date +%Y%m%d_%H%M%S)" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.logger='[console,wandb]' \
    trainer.save_freq=1000 \
    trainer.test_freq=200
