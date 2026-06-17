#!/usr/bin/env bash
# Time-boxed MiniOneRec GRPO benchmark (verl-GR).
# Env: MiniOneRec | GPUs: 4,5,6,7 on hk01dgx036
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
cd "${VERL_GR_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MINIONEREC_ROOT="${MINIONEREC_ROOT:-/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec}"
export CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

export BASE_MODEL="${BASE_MODEL:-${MINIONEREC_ROOT}/output_dir/xxx/checkpoint-390}"
export TRAIN_FILE="${TRAIN_FILE:-${MINIONEREC_ROOT}/data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv}"
export VAL_FILE="${VAL_FILE:-${MINIONEREC_ROOT}/data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv}"
export INFO_FILE="${INFO_FILE:-${MINIONEREC_ROOT}/data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt}"
export SID_INDEX_FILE="${SID_INDEX_FILE:-${MINIONEREC_ROOT}/data/Amazon/index/${CATEGORY}.index.json}"
export ITEM_META_FILE="${ITEM_META_FILE:-${MINIONEREC_ROOT}/data/Amazon/index/${CATEGORY}.item.json}"

export N_NODES="${N_NODES:-1}"
export N_GPUS="${N_GPUS:-4}"
export AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export BEAM_WIDTH="${BEAM_WIDTH:-16}"
export ITEM_MAX_TOKENS="${ITEM_MAX_TOKENS:-128}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-2}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
export SEQ_TITLE_SAMPLE="${SEQ_TITLE_SAMPLE:-10000}"

export PROJECT_NAME="${PROJECT_NAME:-MiniOneRec_RL}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-minionerec_benchmark_${N_GPUS}gpu_$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}}"
export WANDB_MODE="${WANDB_MODE:-offline}"

TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"  # empty = full CSV

mkdir -p "${OUTPUT_DIR}"
WALLCLOCK_FILE="${OUTPUT_DIR}/benchmark_wallclock.txt"
{
  echo "host=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "base_model=${BASE_MODEL}"
  echo "train_file=${TRAIN_FILE}"
  echo "train_max_samples=${TRAIN_MAX_SAMPLES:-all}"
  echo "start_epoch=$(date -Iseconds)"
} > "${WALLCLOCK_FILE}"

export PYTHONUNBUFFERED=1

START_TS=$(date +%s)

EXTRA_SAMPLES=()
if [[ -n "${TRAIN_MAX_SAMPLES}" ]]; then
  EXTRA_SAMPLES+=( "data.train_max_samples=${TRAIN_MAX_SAMPLES}" )
fi

bash scripts/run_minionerec_grpo.sh \
  data.shuffle=true \
  data.seed=42 \
  trainer.val_before_train=false \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  "${EXTRA_SAMPLES[@]}" \
  "$@"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
{
  echo "end_epoch=$(date -Iseconds)"
  echo "elapsed_sec=${ELAPSED}"
} >> "${WALLCLOCK_FILE}"

echo "Benchmark done: ${ELAPSED}s — logs in ${OUTPUT_DIR}"
