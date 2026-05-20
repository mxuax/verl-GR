#!/usr/bin/env bash
# 与 MiniOneRec/rl.sh 超参对齐的 verl-GR GRPO 启动（假设 4 张 GPU）。
# 在 verl-GR 根目录执行: 先改好 BASE_MODEL，再:
#   bash scripts/run_minionerec_grpo_align_rl_sh_4gpu.sh
#
# 与 rl.sh 对应关系（要点）:
#   accelerate --num_processes 4     -> N_GPUS=4
#   --train_batch_size 64          -> TRAIN_BATCH_SIZE=64
#   --gradient_accumulation_steps 2-> PPO_MICRO_BATCH_PER_GPU=2（与 run_minionerec_grpo 注释一致）
#   --num_train_epochs 2           -> TOTAL_EPOCHS=2
#   --num_generations 16           -> BEAM_WIDTH=16
#   --temperature 1.0              -> ROLLOUT_TEMPERATURE=1.0
#   --learning_rate 1e-5           -> LEARNING_RATE=1e-5
#   --beta 1e-3                    -> +actor_rollout_ref.actor.kl_loss_coef=0.001
#   --test_during_training False   -> trainer.val_before_train=false
#   SidDataset 的 seq_title 采样    -> SEQ_TITLE_SAMPLE=10000（与 rl.py 中 RLSeqTitle2SidDataset 一致）
#   seed（rl.py 默认 42）            -> +data.seed=42
#
# 未做一一映射（verl 语义不同）: eval_batch_size、eval_step；可用 trainer.test_freq 等自行加覆盖。
#
# 显存: 须开启 rmpad + 分块熵，避免 old_log_prob 阶段对整段 padded logits 做全词表 softmax OOM。
#       （已写入下方 Hydra 覆盖；亦见 configs/verl_gr/minionerec/grpo_trainer.yaml 默认）

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
MINIONEREC_ROOT="${MINIONEREC_ROOT:-${VERL_GR_ROOT}/../MiniOneRec}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

# 必改: SFT/基座 checkpoint 目录（与 rl.sh --model_path 一致）
BASE_MODEL="${BASE_MODEL:-${MINIONEREC_ROOT}/output_dir/xxx/checkpoint-390}"

TRAIN_FILE="${TRAIN_FILE:-${MINIONEREC_ROOT}/data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv}"
VAL_FILE="${VAL_FILE:-${MINIONEREC_ROOT}/data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv}"
INFO_FILE="${INFO_FILE:-${MINIONEREC_ROOT}/data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt}"
SID_INDEX_FILE="${SID_INDEX_FILE:-${MINIONEREC_ROOT}/data/Amazon/index/${CATEGORY}.index.json}"
ITEM_META_FILE="${ITEM_META_FILE:-${MINIONEREC_ROOT}/data/Amazon/index/${CATEGORY}.item.json}"

# 4 卡（与 rl.sh accelerate --num_processes 4 一致）
export N_NODES=1
export N_GPUS=4
# 提速经验值（可覆盖）:
# - workers: 4 -> 6/8（常见在 constrained beam 下提升并发）
# - inflight: 64 -> 96/128（提升 vLLM 子请求并行度）
export AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-8}"
export CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS="${CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS:-128}"
export TASK_NAME="${TASK_NAME:-minionerec}"
export TASK_CLASS_PATH="${TASK_CLASS_PATH:-verl_gr.recipes.minionerec.minionerec_recipe.MiniOneRecTask}"
export REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-1}"
export CONFIG_NAME="${CONFIG_NAME:-minionerec/grpo_trainer_ddp}"

# 与 rl.sh 数值对齐
# 与原仓对齐：32 unique prompts × 16 beams = 512 completions/update
# 原仓: 64 completions/GPU × 4 GPUs × 2 grad_accum / 16 generations = 32 prompts/update
export TRAIN_BATCH_SIZE=32
export PPO_MICRO_BATCH_PER_GPU=2
export TOTAL_EPOCHS=2
export BEAM_WIDTH=16
export ROLLOUT_TEMPERATURE=1.0
export LEARNING_RATE=1e-5
export SEQ_TITLE_SAMPLE=10000
# 与 run_minionerec_grpo.sh 默认一致（未在 rl.sh 中单独写出）
export ITEM_MAX_TOKENS="${ITEM_MAX_TOKENS:-16}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2560}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
export ROLLOUT_MODE="${ROLLOUT_MODE:-async}"

export PROJECT_NAME="${PROJECT_NAME:-MiniOneRec_RL}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-minionerec_grpo_rlsh4gpu_$(date +%Y%m%d_%H%M%S)}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# vLLM CuMemAllocator 与 expandable_segments 不兼容，必须注释。
# 参见 https://github.com/pytorch/pytorch/issues/147851
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${VERL_GR_ROOT}"
bash scripts/run_minionerec_grpo.sh \
  ++data.shuffle=true \
  ++data.seed=42 \
  ++trainer.val_before_train=true \
  ++trainer.save_freq=165 \
  ++trainer.test_freq=165 \
  ++data.filter_overlong_prompts=false \
  ++data.truncation=left \
  ++actor_rollout_ref.actor.policy_loss.loss_mode=minionerec_reinforce \
  ++actor_rollout_ref.actor.kl_loss_coef=0.001 \
  ++actor_rollout_ref.actor.use_dynamic_bsz=true \
  ++actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  ++actor_rollout_ref.model.use_remove_padding=true \
  ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
  ++actor_rollout_ref.actor.entropy_checkpointing=true \
  "$@"
