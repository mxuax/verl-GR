#!/bin/bash
# OpenOneRec SFT checkpoint evaluation: two-stage beam search + hit rate.
#
# Reuses verl-GR's RL two_stage rollout infrastructure for eval.
# Sets val_before_train=true so the trainer validates then exits.
#
# Usage:
#   bash scripts/misc/sft_eval/eval_sft_onerec.sh <sft_ckpt_path>

set -euo pipefail

SFT_CKPT="${1:?Usage: $0 <sft_ckpt_path>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_GR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_GR_ROOT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_OUTPUT="${VERL_GR_ROOT}/outputs/eval_sft_onerec_${TIMESTAMP}"

# Overfitting verification: same train split as .match_openoneonerec.sh GRPO run.
DATA_DIR="${DATA_DIR:-/home/dyvm6xra/dyvm6xrauser45/fred/openonerec_fredfork/data}"
TRAIN_PARQUET="${DATA_DIR}/train_1k.parquet"
VAL_FILES="[${TRAIN_PARQUET}]"

echo "============================================"
echo "OpenOneRec SFT Evaluation"
echo "  checkpoint    = ${SFT_CKPT}"
echo "  output        = ${EVAL_OUTPUT}"
echo "  val_files     = ${VAL_FILES} (train, overfitting eval)"
echo "============================================"

python -m verl_gr.trainers.main_ppo \
    ++task.name=openonerec \
    data.train_files="${VAL_FILES}" \
    data.val_files="${VAL_FILES}" \
    actor_rollout_ref.model.path="${SFT_CKPT}" \
    actor_rollout_ref.rollout.name=two_stage \
    ++actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40960 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='[Qwen2DecoderLayer]' \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=40960 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='[Qwen2DecoderLayer]' \
    custom_reward_function.name=compute_score \
    custom_reward_function.path=verl_gr.recipes.openonerec.onerec_recipe \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name=OpenOneRec_SFT_Eval \
    trainer.experiment_name="eval_sft_${TIMESTAMP}" \
    trainer.default_local_dir="${EVAL_OUTPUT}" \
    trainer.validation_data_dir="${EVAL_OUTPUT}/val_generations" \
    trainer.log_val_generations=8 \
    trainer.logger='[console,wandb]' \
    critic.enable=false \
    data.shuffle=false \
    data.seed=42
