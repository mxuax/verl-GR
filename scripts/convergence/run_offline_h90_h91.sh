#!/usr/bin/env bash
set -euo pipefail

cd /home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR

PYTHON_BIN=${PYTHON_BIN:-/home/fq9hpsac/fq9hpsacuser04/miniforge3/envs/vllm-gr/bin/python}
BASE_MODEL=${BASE_MODEL:-/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390}

COMMON_ENV=(
  N_GPUS=4
  N_NODES=1
  PYTHON_BIN="${PYTHON_BIN}"
  BASE_MODEL="${BASE_MODEL}"
  BEAM_WIDTH=16
  PPO_MICRO_BATCH_PER_GPU=2
  MAX_TOKENS_PER_GPU=40960
  ROLLOUT_MAX_NUM_SEQS=512
  TOTAL_EPOCHS=2
  WANDB_MODE=offline
)

mkdir -p logs/convergence/offline_runs

run_h90() {
  local exp=h90_reppenalty1_sync660
  rm -rf "outputs/${exp}"
  env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    TEST_FREQ=165 \
    EXPERIMENT_NAME="${exp}" \
    RAY_TMPDIR="/tmp/r90_$$" \
    bash scripts/run_minionerec_grpo.sh \
      ++trainer.total_training_steps=660 \
      ++trainer.save_freq=165 \
      ++trainer.test_freq=165 \
      ++trainer.val_before_train=false \
      ++actor_rollout_ref.actor.optim.scheduler_total_training_steps=3300
}

wait_for_gpus_free() {
  local gpus_csv="$1"
  local max_mem_mb="${2:-2000}"
  while true; do
    local busy
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v gpus="${gpus_csv}" -v max_mem="${max_mem_mb}" '
        BEGIN { split(gpus, a, ","); for (i in a) wanted[a[i]]=1; busy=0 }
        { gsub(/ /, "", $1); gsub(/ /, "", $2); if (($1 in wanted) && $2 > max_mem) busy=1 }
        END { print busy }
      ')
    if [[ "${busy}" == "0" ]]; then
      return 0
    fi
    sleep 120
  done
}

run_h91_after_h89() {
  local exp=h91_reppenalty1_step512_boundary
  wait_for_gpus_free "0,1,2,3" 2000
  rm -rf "outputs/${exp}"
  env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    TEST_FREQ=512 \
    EXPERIMENT_NAME="${exp}" \
    RAY_TMPDIR="/tmp/r91_$$" \
    bash scripts/run_minionerec_grpo.sh \
      ++trainer.total_training_steps=512 \
      ++trainer.save_freq=512 \
      ++trainer.test_freq=512 \
      ++trainer.val_before_train=false \
      ++actor_rollout_ref.actor.optim.scheduler_total_training_steps=3300
}

run_h90 > logs/convergence/offline_runs/h90_reppenalty1_sync660.log 2>&1 &
echo "h90_pid=$!" | tee logs/convergence/offline_runs/h90_reppenalty1_sync660.pid

run_h91_after_h89 > logs/convergence/offline_runs/h91_reppenalty1_step512_boundary.log 2>&1 &
echo "h91_queue_pid=$!" | tee logs/convergence/offline_runs/h91_reppenalty1_step512_boundary.pid

wait
