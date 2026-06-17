#!/bin/bash
#SBATCH --job-name=verl-gr-bench
#SBATCH --partition=q-fq9hpsac
#SBATCH --nodes=1
#SBATCH --nodelist=hk01dgx036
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR/logs/benchmark/slurm_%j.log

set -eo pipefail

VERL_GR_ROOT="/home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR"
cd "${VERL_GR_ROOT}"
mkdir -p logs/benchmark

source ~/miniforge3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export N_NODES=1
export PYTHONUNBUFFERED=1

TS=$(date +%Y%m%d_%H%M%S)

echo "=== $(date -Iseconds) host=$(hostname) GPUs=$(nvidia-smi -L | wc -l) ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

echo "=== OpenOneRec benchmark ==="
conda activate vllm-gr
export EXPERIMENT_NAME="openonerec_benchmark_4gpu_${TS}"
bash scripts/benchmark/run_openonerec_benchmark.sh \
  trainer.val_before_train=false \
  data.val_max_samples=50 \
  trainer.test_freq=500 \
  trainer.save_freq=500

echo "=== MiniOneRec benchmark ==="
conda activate MiniOneRec
export EXPERIMENT_NAME="minionerec_benchmark_4gpu_${TS}"
export TRAIN_MAX_SAMPLES=4096
bash scripts/benchmark/run_minionerec_benchmark.sh

echo "=== All benchmarks done $(date -Iseconds) ==="
