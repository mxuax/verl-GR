#!/usr/bin/env bash
# Smoke-test main-branch training launchers on 4 GPUs (minimal steps/samples).
# Usage (from login node with an active 4-GPU allocation):
#   srun --overlap --jobid=<JOBID> bash scripts/misc/smoke/smoke_test_main_4gpu.sh
# Or directly on a GPU node:
#   bash scripts/misc/smoke/smoke_test_main_4gpu.sh

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
cd "${VERL_GR_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/fq9hpsac/fq9hpsacuser04/miniforge3/envs/vllm-gr/bin/python}"
export PYTHON_BIN
export PYTHONPATH="${VERL_GR_ROOT}:${VERL_GR_ROOT}/../verl:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export N_NODES=1
export N_GPUS=4

LOG_DIR="${VERL_GR_ROOT}/logs/smoke_main_4gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

stop_ray() {
  "${PYTHON_BIN}" -c "import ray; ray.shutdown()" 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  sleep 2
}

run_case() {
  local name="$1"
  shift
  echo "========== [${name}] start $(date -Is) =========="
  stop_ray
  if "$@" >"${LOG_DIR}/${name}.log" 2>&1; then
    echo "========== [${name}] PASS =========="
    return 0
  fi
  echo "========== [${name}] FAIL (see ${LOG_DIR}/${name}.log) =========="
  tail -40 "${LOG_DIR}/${name}.log" || true
  return 1
}

"${PYTHON_BIN}" -c "import torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()"

MINI_BASE="${BASE_MODEL:-/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390}"
OPEN_BASE="${OPENONEREC_BASE_MODEL:-/home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR/outputs/db455d0bdcf4b5e5e0b42f30c45d65260a49656a7f_20260424_142645/ckpt/global_step_400/actor/huggingface}"
# typo guard: fix path if the above does not exist
if [[ ! -f "${OPEN_BASE}/config.json" ]]; then
  OPEN_BASE="/home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR/outputs/db455d0bdcf4b5e0b42f30c45d65260a49656a7f_20260424_142645/ckpt/global_step_400/actor/huggingface"
fi

RANK_DATA="${RANKGRPO_DATA_DIR:-${VERL_GR_ROOT}/../rankgrpo_data_ckpts}"

FAIL=0

if [[ -f "${MINI_BASE}/config.json" ]]; then
  run_case minionerec_grpo \
    env BASE_MODEL="${MINI_BASE}" EXPERIMENT_NAME="smoke_mini_$(date +%H%M%S)" \
    bash scripts/run_minionerec_grpo_rl_aligned.sh \
      ++trainer.total_epochs=1 \
      ++data.train_max_samples=64 \
      ++data.val_max_samples=0 \
      ++trainer.test_freq=1000000 \
      ++trainer.save_freq=1000000 \
    || FAIL=$((FAIL + 1))
else
  echo "[minionerec_grpo] SKIP: missing BASE_MODEL ${MINI_BASE}"
  FAIL=$((FAIL + 1))
fi

if [[ -f "${OPEN_BASE}/config.json" && -f "${VERL_GR_ROOT}/verl_gr/recipes/openonerec/output/rl_data/train.parquet" ]]; then
  run_case openonerec_grpo \
    env N_GPUS=4 N_NODES=1 BASE_MODEL="${OPEN_BASE}" EXPERIMENT_NAME="smoke_open_$(date +%H%M%S)" \
      AGENT_LOOP_NUM_WORKERS=4 TOTAL_EPOCHS=1 TEST_FREQ=1000000 SAVE_FREQ=1000000 \
    bash scripts/run_openonerec_grpo.sh \
      data.train_max_samples=8 \
      data.val_max_samples=0 \
      trainer.val_before_train=false \
      trainer.total_epochs=1 \
    || FAIL=$((FAIL + 1))
else
  echo "[openonerec_grpo] SKIP: model=${OPEN_BASE} or train.parquet missing"
  FAIL=$((FAIL + 1))
fi

if [[ -f "${RANK_DATA}/Qwen2.5-0.5B-Instruct/checkpoint-1500/config.json" && -d "${RANK_DATA}/processed_datasets/grpo/grpo_dataset/train" ]]; then
  run_case rankgrpo \
    env N_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ROLLOUT_TENSOR_PARALLEL_SIZE=2 \
      DATA_DIR="${RANK_DATA}" VERL_GR_ENV=/home/fq9hpsac/fq9hpsacuser04/miniforge3/envs/vllm-gr \
      PYTHON_BIN="${PYTHON_BIN}" TOTAL_EPOCHS=1 TRAIN_MAX_SAMPLES=32 VAL_MAX_SAMPLES=0 \
      TEST_FREQ=1000000 SAVE_FREQ=1000000 VAL_BEFORE_TRAIN=False \
      EXPERIMENT_NAME="smoke_rank_$(date +%H%M%S)" \
    bash scripts/run_rankgrpo.sh \
    || FAIL=$((FAIL + 1))
else
  echo "[rankgrpo] SKIP: data dir not found at ${RANK_DATA}"
  echo "  (set RANKGRPO_DATA_DIR if data lives elsewhere)"
fi

stop_ray
echo "Logs: ${LOG_DIR}"
if (( FAIL > 0 )); then
  echo "Smoke summary: ${FAIL} required case(s) failed or skipped."
  exit 1
fi
echo "Smoke summary: all runnable cases passed."
