#!/usr/bin/env bash
set -euo pipefail

cd /home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR

PYTHON_BIN=${PYTHON_BIN:-/home/fq9hpsac/fq9hpsacuser04/miniforge3/envs/vllm-gr/bin/python}
BASE_MODEL=${BASE_MODEL:-/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390}
SOURCE_EXP=${SOURCE_EXP:-h90_reppenalty1_sync660}
GPU_LIST=${GPU_LIST:-0,1,2,3}

mkdir -p logs/convergence/eval50

for step in 165 330 495 660; do
  ckpt="outputs/${SOURCE_EXP}/ckpt/global_step_${step}"
  exp="h92_eval50_${SOURCE_EXP}_s${step}"
  if [[ ! -d "${ckpt}" ]]; then
    echo "[skip] missing ${ckpt}"
    continue
  fi
  echo "[eval50] step=${step} ckpt=${ckpt} exp=${exp}"
  rm -rf "outputs/${exp}"
  env \
    CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    N_GPUS=4 \
    N_NODES=1 \
    PYTHON_BIN="${PYTHON_BIN}" \
    BASE_MODEL="${BASE_MODEL}" \
    BEAM_WIDTH=16 \
    VAL_BEAM_WIDTH=50 \
    PPO_MICRO_BATCH_PER_GPU=2 \
    MAX_TOKENS_PER_GPU=40960 \
    ROLLOUT_MAX_NUM_SEQS=512 \
    TEST_FREQ=1 \
    TOTAL_EPOCHS=2 \
    WANDB_MODE=offline \
    EXPERIMENT_NAME="${exp}" \
    RAY_TMPDIR="/tmp/r${step}_$$" \
    bash scripts/run_minionerec_grpo.sh \
      ++trainer.resume_mode=resume_path \
      ++trainer.resume_from_path="${ckpt}" \
      ++trainer.val_before_train=true \
      ++trainer.val_only=true \
      ++trainer.total_training_steps=1 \
      ++trainer.save_freq=-1 \
      ++trainer.test_freq=1 \
      ++actor_rollout_ref.actor.optim.scheduler_total_training_steps=3300 \
      > "logs/convergence/eval50/${exp}.log" 2>&1
done
