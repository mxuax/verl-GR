#!/usr/bin/env bash
#SBATCH --job-name=h95_kl_clamp
#SBATCH --partition=q-fq9hpsac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR/logs/convergence/h95_kl_clamp_grad.sbatch.%j.out
#SBATCH --error=/home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR/logs/convergence/h95_kl_clamp_grad.sbatch.%j.err

set -euo pipefail

cd /home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR

PYTHON_BIN=${PYTHON_BIN:-/home/fq9hpsac/fq9hpsacuser04/miniforge3/envs/vllm-gr/bin/python}

mkdir -p logs/convergence/h95_kl_clamp_grad

run_probe() {
  local run_name="$1"
  local exp_name="$2"
  local step="$3"
  local fixture="logs/convergence/${run_name}/mini_fixtures/step$(printf '%06d' "${step}")_post_mini.pt"
  local model="outputs/${exp_name}/ckpt/global_step_${step}/actor/huggingface"
  local out="logs/convergence/${run_name}/h95_kl_clamp_grad_step${step}_mini.json"

  echo "[h95] run=${run_name} step=${step} fixture=${fixture} model=${model} out=${out}"
  "${PYTHON_BIN}" scripts/convergence/h95_compare_kl_clamp_grad.py \
    --fixture "${fixture}" \
    --model "${model}" \
    --out "${out}" \
    --dtype bf16 \
    --attn-implementation sdpa \
    --disable-flash-sdp \
    --micro-batch-size 4
}

run_probe h93_dump330495 h93_dump330495_eval50 330
run_probe h93_dump330495 h93_dump330495_eval50 495
run_probe h94_unclampedkl h94_unclampedkl_eval50 330
run_probe h94_unclampedkl h94_unclampedkl_eval50 495

echo "[h95] done"
