#!/usr/bin/env bash
# Chain MiniOneRec benchmark after OpenOneRec completes.
set -euo pipefail

VERL_GR_ROOT="/home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR"
WALL="${VERL_GR_ROOT}/outputs/openonerec_benchmark_2gpu_20260615_102322/benchmark_wallclock.txt"
LOG_DIR="${VERL_GR_ROOT}/logs/benchmark"

while ! grep -q 'elapsed_sec=' "${WALL}" 2>/dev/null; do
  pending=$(grep -o 'pending_requests=[0-9]*' "${LOG_DIR}/openonerec_20260615_102322.log" 2>/dev/null | tail -1 || true)
  step=$(grep -oE "global_step[s']?: [0-9]+" "${LOG_DIR}/openonerec_20260615_102322.log" 2>/dev/null | tail -1 || true)
  echo "$(date -Iseconds) waiting OpenOneRec pending=${pending:-?} ${step:-}"
  sleep 300
done

TS=$(date +%Y%m%d_%H%M%S)
MINI_LOG="${LOG_DIR}/minionerec_${TS}.log"
echo "$(date -Iseconds) OpenOneRec done, starting MiniOneRec -> ${MINI_LOG}"

srun --overlap --jobid=111945 bash -lc "
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate MiniOneRec
  cd ${VERL_GR_ROOT}
  export N_GPUS=2 N_NODES=1 CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1
  export TRAIN_BATCH_SIZE=16 AGENT_LOOP_NUM_WORKERS=2
  export EXPERIMENT_NAME=minionerec_benchmark_2gpu_${TS}
  bash scripts/benchmark/run_minionerec_benchmark.sh
" > "${MINI_LOG}" 2>&1

echo "$(date -Iseconds) MiniOneRec finished"
