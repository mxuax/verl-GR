#!/bin/bash
# OpenOneRec SFT checkpoint evaluation: two-stage beam search + hit rate.
#
# Reuses verl-GR's RL two_stage rollout infrastructure for eval.
# Sets val_before_train=true so the trainer validates then exits.
#
# Usage:
#   bash scripts/eval_sft_onerec.sh <sft_ckpt_path>

set -euo pipefail

SFT_CKPT="${1:?Usage: $0 <sft_ckpt_path>}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_OUTPUT="./outputs/eval_sft_onerec_${TIMESTAMP}"

echo "============================================"
echo "OpenOneRec SFT Evaluation"
echo "  checkpoint    = ${SFT_CKPT}"
echo "  output        = ${EVAL_OUTPUT}"
echo "============================================"

python -m verl_gr.trainers.main_ppo \
    ++task.name=openonerec \
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
