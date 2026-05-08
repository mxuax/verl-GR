#!/bin/bash
# MiniOneRec SFT checkpoint evaluation: constrained beam search + hit rate.
#
# Reuses verl-GR's RL constrained_beam rollout infrastructure for eval.
# Sets val_before_train=true so the trainer validates then exits.
#
# Usage:
#   bash scripts/eval_sft_minionerec.sh <sft_ckpt_path> [category]

set -euo pipefail

SFT_CKPT="${1:?Usage: $0 <sft_ckpt_path> [category]}"
CATEGORY="${2:-Industrial_and_Scientific}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/../MiniOneRec/data/Amazon"

TRAIN_FILE="${DATA_ROOT}/train/${CATEGORY}_5_*.csv"
EVAL_FILE="${DATA_ROOT}/valid/${CATEGORY}_5_*.csv"
SID_INDEX="${DATA_ROOT}/index/${CATEGORY}.index.json"
ITEM_META="${DATA_ROOT}/index/${CATEGORY}.item.json"
INFO_FILE="${DATA_ROOT}/info/${CATEGORY}_5_*.txt"
BASE_MODEL="${SFT_CKPT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_OUTPUT="./outputs/eval_sft_minionerec_${CATEGORY}_${TIMESTAMP}"

echo "============================================"
echo "MiniOneRec SFT Evaluation"
echo "  checkpoint    = ${SFT_CKPT}"
echo "  category      = ${CATEGORY}"
echo "  info_file     = ${INFO_FILE}"
echo "  output        = ${EVAL_OUTPUT}"
echo "============================================"

python -m verl_gr.trainers.main_ppo \
    ++task.name=minionerec \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${EVAL_FILE}" \
    data.custom_cls.name=MiniOneRecDataset \
    data.custom_cls.path=verl_gr.recipes.minionerec.minionerec_recipe \
    +data.category="${CATEGORY}" \
    +data.sid_index_path="${SID_INDEX}" \
    +data.item_meta_path="${ITEM_META}" \
    +data.include_alignment_tasks=false \
    +data.include_alignment_tasks_for_val=false \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.rollout.name=constrained_beam \
    ++actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.custom.beam_width=50 \
    ++actor_rollout_ref.rollout.custom.decode_mode_val=deterministic_beam \
    ++actor_rollout_ref.rollout.custom.disable_cache_in_train=false \
    actor_rollout_ref.rollout.custom.beam_search_params.max_tokens=16 \
    ++actor_rollout_ref.rollout.custom.beam_search_params.logprobs_multiplier=2 \
    ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.type=minionerec_prefix_trie \
    ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.info_file="${INFO_FILE}" \
    ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.base_model="${BASE_MODEL}" \
    ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.fallback_to_eos=true \
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
    custom_reward_function.path=verl_gr.recipes.minionerec.minionerec_reward \
    data.train_batch_size=64 \
    data.max_prompt_length=2560 \
    data.max_response_length=64 \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.project_name=MiniOneRec_SFT_Eval \
    trainer.experiment_name="eval_sft_${CATEGORY}_${TIMESTAMP}" \
    trainer.default_local_dir="${EVAL_OUTPUT}" \
    trainer.validation_data_dir="${EVAL_OUTPUT}/val_generations" \
    trainer.log_val_generations=8 \
    trainer.logger='[console,wandb]' \
    critic.enable=false \
    data.shuffle=false \
    data.seed=42
