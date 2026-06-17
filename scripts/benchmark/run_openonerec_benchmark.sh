#!/usr/bin/env bash
# Time-boxed OpenOneRec GRPO benchmark (verl-GR).
# Env: vllm-gr | GPUs: 4,5,6,7 on hk01dgx036
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
cd "${VERL_GR_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export N_NODES="${N_NODES:-1}"
export N_GPUS="${N_GPUS:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"

export BASE_MODEL="${BASE_MODEL:-/scratch/fq9hpsac/huggingface/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/5dc1b097ab8194f48f14730e5400a276a22f4ca1}"
export DATA_DIR="${DATA_DIR:-${VERL_GR_ROOT}/verl_gr/recipes/openonerec/output/rl_data}"
export TRAIN_FILES="${TRAIN_FILES:-[${DATA_DIR}/train.parquet]}"
export VAL_FILES="${VAL_FILES:-[${DATA_DIR}/test.parquet]}"

export PROJECT_NAME="${PROJECT_NAME:-OneRec_RL}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-openonerec_benchmark_${N_GPUS}gpu_$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-4}"

# Subset for controlled runtime (override via env)
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-2000}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-50}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
BEAM_WIDTH="${BEAM_WIDTH:-32}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-12288}"
TWO_STAGE_MAX_INFLIGHT="${TWO_STAGE_MAX_INFLIGHT:-16}"
export TEST_FREQ="${TEST_FREQ:-500}"
export SAVE_FREQ="${SAVE_FREQ:-500}"

mkdir -p "${OUTPUT_DIR}"
WALLCLOCK_FILE="${OUTPUT_DIR}/benchmark_wallclock.txt"
{
  echo "host=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "base_model=${BASE_MODEL}"
  echo "train_max_samples=${TRAIN_MAX_SAMPLES}"
  echo "val_max_samples=${VAL_MAX_SAMPLES}"
  echo "start_epoch=$(date -Iseconds)"
} > "${WALLCLOCK_FILE}"

export PYTHONUNBUFFERED=1

START_TS=$(date +%s)

bash scripts/run_openonerec_grpo.sh \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.val_before_train=true \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.log_val_generations=4 \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.custom.beam_width="${BEAM_WIDTH}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true \
  actor_rollout_ref.rollout.custom.two_stage_max_inflight_requests="${TWO_STAGE_MAX_INFLIGHT}" \
  "$@"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
{
  echo "end_epoch=$(date -Iseconds)"
  echo "elapsed_sec=${ELAPSED}"
} >> "${WALLCLOCK_FILE}"

echo "Benchmark done: ${ELAPSED}s — logs in ${OUTPUT_DIR}"
