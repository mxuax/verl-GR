#!/usr/bin/env bash
# verl-GR GRPO launcher aligned with MiniOneRec/rl.sh (assumes 4 GPUs).
# Run from verl-GR root after setting BASE_MODEL:
#   bash scripts/run_minionerec_grpo_align_rl_sh_4gpu.sh
#
# Mapping to rl.sh (key points):
#   accelerate --num_processes 4     -> N_GPUS=4, AGENT_LOOP_NUM_WORKERS=4
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
# Memory: enable rmpad + chunked entropy to avoid OOM from full-sequence vocab softmax in old_log_prob.
#       (Hydra overrides below; see configs/verl_gr/minionerec/grpo_trainer.yaml defaults)

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
MINIONEREC_ROOT="${MINIONEREC_ROOT:-${VERL_GR_ROOT}/../MiniOneRec}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

# Required: SFT/base checkpoint directory (same as rl.sh --model_path)
BASE_MODEL="${BASE_MODEL:-${MINIONEREC_ROOT}/output_dir/xxx/checkpoint-390}"

TRAIN_FILE="${TRAIN_FILE:-${MINIONEREC_ROOT}/data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv}"
VAL_FILE="${VAL_FILE:-${MINIONEREC_ROOT}/data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv}"
INFO_FILE="${INFO_FILE:-${MINIONEREC_ROOT}/data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt}"
SID_INDEX_FILE="${SID_INDEX_FILE:-${MINIONEREC_ROOT}/data/Amazon/index/${CATEGORY}.index.json}"
ITEM_META_FILE="${ITEM_META_FILE:-${MINIONEREC_ROOT}/data/Amazon/index/${CATEGORY}.item.json}"

# 4 GPUs (same as rl.sh accelerate --num_processes 4)
export N_NODES=1
export N_GPUS=4
export AGENT_LOOP_NUM_WORKERS=4

# Hyperparameters aligned with rl.sh
export TRAIN_BATCH_SIZE=64
export PPO_MICRO_BATCH_PER_GPU=2
export TOTAL_EPOCHS=2
export BEAM_WIDTH=16
export ROLLOUT_TEMPERATURE=1.0
export LEARNING_RATE=1e-5
export SEQ_TITLE_SAMPLE=10000
# Defaults match run_minionerec_grpo.sh (not all listed explicitly in rl.sh)
export ITEM_MAX_TOKENS="${ITEM_MAX_TOKENS:-16}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2560}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
export ROLLOUT_MODE="${ROLLOUT_MODE:-async}"

export PROJECT_NAME="${PROJECT_NAME:-MiniOneRec_RL}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-minionerec_grpo_rlsh4gpu_$(date +%Y%m%d_%H%M%S)}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Optional: reduce CUDA memory fragmentation
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${VERL_GR_ROOT}"
bash scripts/run_minionerec_grpo.sh \
  data.shuffle=true \
  data.seed=42 \
  trainer.val_before_train=false \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
  actor_rollout_ref.actor.entropy_checkpointing=true \
  "$@"
