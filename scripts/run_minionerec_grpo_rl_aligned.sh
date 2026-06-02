#!/usr/bin/env bash
# verl-GR GRPO launcher aligned with MiniOneRec/rl.sh (assumes 4 GPUs).
# Run from verl-GR root after setting BASE_MODEL:
#   bash scripts/run_minionerec_grpo_rl_aligned.sh
#
# Mapping to rl.sh (key points):
#   accelerate --num_processes 4     -> N_GPUS=4
#   --train_batch_size 64 (completions) -> TRAIN_BATCH_SIZE=32 prompts (x beam 16 = 512 completions/step)
#   --gradient_accumulation_steps 2-> PPO_MICRO_BATCH_PER_GPU=2 (see run_minionerec_grpo.sh)
#   --num_train_epochs 2           -> TOTAL_EPOCHS=2
#   --num_generations 16           -> BEAM_WIDTH=16
#   --temperature 1.0              -> ROLLOUT_TEMPERATURE=1.0
#   --learning_rate 1e-5           -> LEARNING_RATE=1e-5
#   --beta 1e-3                    -> +actor_rollout_ref.actor.kl_loss_coef=0.001
#   --test_during_training False   -> trainer.val_before_train=false
#   SidDataset seq_title sampling  -> SEQ_TITLE_SAMPLE=10000 (RLSeqTitle2SidDataset in rl.py)
#   seed (rl.py default 42)            -> +data.seed=42
#
# Not 1:1 mapped (different verl semantics): eval_batch_size, eval_step; override trainer.test_freq as needed.
#
# Dependency: paged_adamw_32bit requires bitsandbytes (pip install bitsandbytes)
# Performance: completion_only_logprob + logits_to_keep enabled by default in MiniOneRec recipe (aligned with original TRL)
# Memory: enable rmpad + chunked entropy to avoid OOM from full-vocab softmax in old_log_prob.
#       (Hydra overrides below; see configs/verl_gr/minionerec/grpo_trainer.yaml defaults)

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
MINIONEREC_ROOT="${MINIONEREC_ROOT:-${VERL_GR_ROOT}/../MiniOneRec}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

# Required: SFT/base checkpoint directory (same as rl.sh --model_path)
export BASE_MODEL="${BASE_MODEL:-${MINIONEREC_ROOT}/output_dir/xxx/checkpoint-390}"
export TRAIN_FILE="${TRAIN_FILE:-${MINIONEREC_ROOT}/data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv}"
export VAL_FILE="${VAL_FILE:-${MINIONEREC_ROOT}/data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv}"
export INFO_FILE="${INFO_FILE:-${MINIONEREC_ROOT}/data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt}"
export SID_INDEX_FILE="${SID_INDEX_FILE:-${MINIONEREC_ROOT}/data/Amazon/index/${CATEGORY}.index.json}"
export ITEM_META_FILE="${ITEM_META_FILE:-${MINIONEREC_ROOT}/data/Amazon/index/${CATEGORY}.item.json}"

# 4 GPUs (same as rl.sh accelerate --num_processes 4)
export N_NODES=1
export N_GPUS=4
# Optional tuning knobs (override via env):
# - AGENT_LOOP_NUM_WORKERS: 4 -> 6/8 (often helps constrained beam throughput)
# - CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS: 64 -> 96/128 (higher vLLM sub-request parallelism)
export AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-8}"
export CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS="${CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS:-128}"
export TASK_NAME="${TASK_NAME:-minionerec}"
export TASK_CLASS_PATH="${TASK_CLASS_PATH:-verl_gr.recipes.minionerec.minionerec_recipe.MiniOneRecTask}"
export REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-1}"
export CONFIG_NAME="${CONFIG_NAME:-minionerec/grpo_trainer_ddp}"

# Hyperparameters aligned with rl.sh
# Aligned with original MiniOneRec: 32 unique prompts x 16 beams = 512 completions/update
# Original: 64 completions/GPU x 4 GPUs x 2 grad_accum / 16 generations = 32 prompts/update
export TRAIN_BATCH_SIZE=32
export PPO_MICRO_BATCH_PER_GPU=2
export TOTAL_EPOCHS=2
export BEAM_WIDTH=16
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export SEQ_TITLE_SAMPLE=10000
# Defaults match run_minionerec_grpo.sh (not all listed explicitly in rl.sh)
export ITEM_MAX_TOKENS="${ITEM_MAX_TOKENS:-16}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2560}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
export ROLLOUT_MODE="${ROLLOUT_MODE:-async}"

export PROJECT_NAME="${PROJECT_NAME:-MiniOneRec_RL}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-minionerec_grpo_rlsh4gpu_$(date +%Y%m%d_%H%M%S)}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# vLLM CuMemAllocator is incompatible with expandable_segments; keep this commented.
# See https://github.com/pytorch/pytorch/issues/147851
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${VERL_GR_ROOT}"
bash scripts/run_minionerec_grpo.sh \
  ++data.shuffle=true \
  ++data.seed=42 \
  ++trainer.val_before_train=false \
  ++trainer.save_freq=165 \
  ++trainer.test_freq=165 \
  ++data.filter_overlong_prompts=false \
  ++data.truncation=left \
  ++actor_rollout_ref.actor.policy_loss.loss_mode=minionerec_reinforce \
  ++actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  ++actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  ++actor_rollout_ref.actor.optim.clip_grad=0.3 \
  ++actor_rollout_ref.actor.optim.weight_decay=0.0 \
  ++actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  ++actor_rollout_ref.actor.use_dynamic_bsz=true \
  ++actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  ++actor_rollout_ref.model.use_remove_padding=true \
  ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
  ++actor_rollout_ref.actor.entropy_checkpointing=true \
  "$@"
