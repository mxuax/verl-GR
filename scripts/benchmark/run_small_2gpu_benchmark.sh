#!/bin/bash
#SBATCH --job-name=verl-gr-sm-bench
#SBATCH --partition=q-fq9hpsac
#SBATCH --nodes=1
#SBATCH --nodelist=hk01dgx006
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR/logs/benchmark/slurm_small_%j.log

# Small-subset benchmark on 2 GPUs (hk01dgx006 cards 4,5 when allocated).
# Launch manually:
#   srun --overlap --jobid=<JOBID> bash scripts/benchmark/run_small_2gpu_benchmark.sh
set -eo pipefail

VERL_GR_ROOT="/home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR"
cd "${VERL_GR_ROOT}"
mkdir -p logs/benchmark

source ~/miniforge3/etc/profile.d/conda.sh

export N_GPUS=2
export N_NODES=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export AGENT_LOOP_NUM_WORKERS=2

TS=$(date +%Y%m%d_%H%M%S)

echo "=== $(date -Iseconds) host=$(hostname) visible_gpus=$(nvidia-smi -L | wc -l) ==="
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv

# --- OpenOneRec: 200 train / 10 val, beam 8, no mid-run validation ---
echo "=== OpenOneRec (small) ==="
conda activate vllm-gr
export EXPERIMENT_NAME="openonerec_sm2gpu_${TS}"
export TRAIN_MAX_SAMPLES=200
export VAL_MAX_SAMPLES=10
export TRAIN_BATCH_SIZE=2
export VAL_BATCH_SIZE=8
export BEAM_WIDTH=8
export MAX_TOKENS_PER_GPU=8192
export TWO_STAGE_MAX_INFLIGHT=4
export TEST_FREQ=200
export SAVE_FREQ=200
export TOTAL_EPOCHS=1
export VAL_MAX_SAMPLES=10

bash scripts/benchmark/run_openonerec_benchmark.sh \
  trainer.val_before_train=false \
  trainer.test_freq=200 \
  trainer.save_freq=200 \
  data.val_max_samples=10 \
  actor_rollout_ref.rollout.custom.beam_width=8 \
  actor_rollout_ref.rollout.custom.two_stage_max_inflight_requests=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=512

# --- MiniOneRec: 256 samples, 1 epoch, beam 8 ---
echo "=== MiniOneRec (small) ==="
conda activate MiniOneRec
export EXPERIMENT_NAME="minionerec_sm2gpu_${TS}"
export TRAIN_MAX_SAMPLES=256
export TRAIN_BATCH_SIZE=16
export BEAM_WIDTH=8
export TOTAL_EPOCHS=1
export TEST_FREQ=128
export SAVE_FREQ=128

bash scripts/benchmark/run_minionerec_benchmark.sh \
  trainer.val_before_train=false \
  trainer.total_epochs=1 \
  data.train_max_samples=256 \
  actor_rollout_ref.rollout.custom.beam_width=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true

echo "=== All small benchmarks done $(date -Iseconds) ==="
