#!/usr/bin/env bash
# Minimal Rank-GRPO runtime launcher for verl-GR.

set -euo pipefail
clear
export CUDA_VISIBLE_DEVICES=4,5,6,7
N_GPUS=4

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
PROJECT_ROOT="$(dirname "${VERL_GR_ROOT}")"
RANKGRPO_RECIPE_PATH="${VERL_GR_ROOT}/verl_gr/recipes/rankgrpo/rankgrpo_recipe.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

RAY_INFO="$("${PYTHON_BIN}" -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes=[n for n in ray.nodes() if n.get('Alive')]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0),0); print(f'{len(nodes)} {gpus}')" 2>/dev/null || true)"
N_NODES="${N_NODES:-$(echo "${RAY_INFO}" | awk '{print $1}')}"
# N_GPUS="${N_GPUS:-$(echo "${RAY_INFO}" | awk '{print $2}')}"
if [[ -z "${N_NODES}" || -z "${N_GPUS}" || "${N_NODES}" == "0" ]]; then
  N_NODES=1
  # N_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  # if [[ -z "${N_GPUS}" || "${N_GPUS}" == "0" ]]; then
  #   N_GPUS=1
  # fi
fi

SFT_CHECKPOINT="${SFT_CHECKPOINT:-1500}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/rankgrpo_data_ckpts}"
BASE_MODEL="${BASE_MODEL:-${DATA_DIR}/Qwen2.5-0.5B-Instruct/checkpoint-${SFT_CHECKPOINT}}"
BASE_MODEL_DIRNAME="$(basename "${BASE_MODEL%/}")"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-${DATA_DIR}/processed_datasets/grpo/grpo_dataset/train}"
VAL_DATASET_DIR="${VAL_DATASET_DIR:-${DATA_DIR}/processed_datasets/sft_dataset/validation}"
TRAIN_FILES="${TRAIN_FILES:-[${TRAIN_DATASET_DIR}]}"
VAL_FILES="${VAL_FILES:-[${VAL_DATASET_DIR}]}"
ROLLOUT_N="${ROLLOUT_N:-8}"
REC_NUM="${REC_NUM:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_GPUS * N_NODES))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
# vLLM sleep-mode memory release can crash in CUDA/cumem after long runs on this
# stack. Keep rollout memory resident by default; override both to True if needed.
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-${ROLLOUT_FREE_CACHE_ENGINE}}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.99}"
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.06}"
PPO_CLIP_RATIO_HIGH="${PPO_CLIP_RATIO_HIGH:-0.08}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
PROJECT_NAME="${PROJECT_NAME:-RankGRPO}"
LAUNCH_TIMESTAMP="${LAUNCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${BASE_MODEL_DIRNAME}_${LAUNCH_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}}"
WANDB_MODE="${WANDB_MODE:-offline}"
RAY_TMPDIR="${RAY_TMPDIR:-${OUTPUT_DIR}/ray_tmp}"
RAY_TMPDIR_FALLBACK_ROOT="${RAY_TMPDIR_FALLBACK_ROOT:-${TMPDIR:-/tmp}}"
RAY_TMPDIR_MAX_LEN="${RAY_TMPDIR_MAX_LEN:-60}"
if (( ${#RAY_TMPDIR} > RAY_TMPDIR_MAX_LEN )); then
  # Ray creates deep session/socket paths under _temp_dir. Long roots can exceed
  # Linux AF_UNIX path limits, so use a short temp root for Ray only.
  SHORT_USER="${USER:-user}"
  RAY_TMPDIR="${RAY_TMPDIR_FALLBACK_ROOT}/vgr_ray_${SHORT_USER}"
  echo "Warning: RAY_TMPDIR path too long, fallback to ${RAY_TMPDIR}" >&2
fi
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${RAY_TMPDIR}/spill}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-200}"
TEST_FREQ="${TEST_FREQ:-200}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-2000}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[tensorboard]}"

mkdir -p "${VERL_GR_ROOT}/logs" "${OUTPUT_DIR}" "${RAY_TMPDIR}" "${RAY_SPILL_DIR}"

TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
export TENSORBOARD_DIR
export PYTHONPATH="${VERL_GR_ROOT}:${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND
export WANDB_MODE
export RAY_TMPDIR
export TMPDIR="${RAY_TMPDIR}"

echo "==================================="
echo "Rank-GRPO (verl-GR runtime)"
echo "==================================="
echo "Cluster: ${N_NODES} node(s) x ${N_GPUS} GPU(s)"
echo "Model: ${BASE_MODEL}"
echo "Train data: ${TRAIN_FILES}"
echo "Validation data: ${VAL_FILES}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Rec num: ${REC_NUM}"
echo "Rollout free cache engine: ${ROLLOUT_FREE_CACHE_ENGINE}"
echo "Rollout sleep mode: ${ROLLOUT_ENABLE_SLEEP_MODE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Output: ${OUTPUT_DIR}"
echo "==================================="

for arg in "$@"; do
  if [[ "$arg" == *"Rank-GRPO"* || "$arg" == *"trl"* ]]; then
    echo "Error: TRL/reference Rank-GRPO dependency detected in argument: $arg" >&2
    echo "Use only the verl_gr Rank-GRPO recipe path." >&2
    exit 2
  fi
done

"${PYTHON_BIN}" -u -m verl_gr.trainers.main_rankgrpo \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  ++data.train_max_samples=40000 \
  ++data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.custom_cls.path="${RANKGRPO_RECIPE_PATH}" \
  custom_reward_function.path="${RANKGRPO_RECIPE_PATH}" \
  data.rankgrpo.rec_num="${REC_NUM}" \
  algorithm.rank_grpo.rec_num="${REC_NUM}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.clip_ratio="${PPO_CLIP_RATIO}" \
  actor_rollout_ref.actor.clip_ratio_low="${PPO_CLIP_RATIO}" \
  actor_rollout_ref.actor.clip_ratio_high="${PPO_CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.betas="[${ADAM_BETA1},${ADAM_BETA2}]" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
  actor_rollout_ref.rollout.enable_sleep_mode="${ROLLOUT_ENABLE_SLEEP_MODE}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.logger="${LOGGER_BACKENDS}" \
  trainer.remove_previous_ckpt_in_save=True \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  +ray_kwargs.ray_init.object_spilling_directory="${RAY_SPILL_DIR}" \
  global_profiler.save_path="${GLOBAL_PROFILER_SAVE_PATH:-${OUTPUT_DIR}/profiles}" \
  actor_rollout_ref.ref.strategy="${FSDP_STRATEGY}" \
  actor_rollout_ref.actor.strategy="${FSDP_STRATEGY}" \
  critic.enable=False \
  "$@"

