#!/usr/bin/env bash
# Minimal Rank-GRPO runtime launcher for verl-GR.

set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
N_GPUS="${N_GPUS:-2}"

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
PROJECT_ROOT="$(dirname "${VERL_GR_ROOT}")"
WORKSPACE_ROOT="$(dirname "${PROJECT_ROOT}")"
RANKGRPO_RECIPE_PATH="${VERL_GR_ROOT}/verl_gr/recipes/rankgrpo/rankgrpo_recipe.py"
DEFAULT_VERL_LIB_PATH="${WORKSPACE_ROOT}/verl_080_dev"
if [[ ! -d "${DEFAULT_VERL_LIB_PATH}/verl" ]]; then
  DEFAULT_VERL_LIB_PATH=""
fi
VERL_LIB_PATH="${VERL_LIB_PATH:-${DEFAULT_VERL_LIB_PATH}}"
VERL_GR_ENV="${VERL_GR_ENV:-/home/dyvm6xra/dyvm6xrauser45/miniconda3/envs/verl_080}"
PYTHON_BIN="${PYTHON_BIN:-${VERL_GR_ENV}/bin/python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
unset RAY_ADDRESS

N_NODES="${N_NODES:-1}"

SFT_CHECKPOINT="${SFT_CHECKPOINT:-1500}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/rankgrpo_data_ckpts}"
BASE_MODEL="${BASE_MODEL:-${DATA_DIR}/Qwen2.5-0.5B-Instruct/checkpoint-${SFT_CHECKPOINT}}"
BASE_MODEL_DIRNAME="$(basename "${BASE_MODEL%/}")"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-${DATA_DIR}/processed_datasets/grpo/grpo_dataset/train}"
VAL_DATASET_DIR="${VAL_DATASET_DIR:-${DATA_DIR}/processed_datasets/sft_dataset/validation}"
GT_CATALOG_PATH="${GT_CATALOG_PATH:-${DATA_DIR}/processed_datasets/gt_catalog.pkl}"
TRAIN_FILES="${TRAIN_FILES:-[${TRAIN_DATASET_DIR}]}"
VAL_FILES="${VAL_FILES:-[${VAL_DATASET_DIR}]}"
ROLLOUT_N="${ROLLOUT_N:-8}"
REC_NUM="${REC_NUM:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-6}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$((16 * N_GPUS))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
ACTOR_MAX_TOKENS_PER_GPU="${ACTOR_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
LOG_PROB_MAX_TOKENS_PER_GPU="${LOG_PROB_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_TOKENS_PER_GPU}}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}"
DEFAULT_ROLLOUT_TENSOR_PARALLEL_SIZE="${DEFAULT_ROLLOUT_TENSOR_PARALLEL_SIZE:-2}"
if (( DEFAULT_ROLLOUT_TENSOR_PARALLEL_SIZE > N_GPUS )); then
  DEFAULT_ROLLOUT_TENSOR_PARALLEL_SIZE="${N_GPUS}"
fi
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-${DEFAULT_ROLLOUT_TENSOR_PARALLEL_SIZE}}"
if (( N_GPUS % ROLLOUT_TENSOR_PARALLEL_SIZE != 0 )); then
  echo "Error: N_GPUS (${N_GPUS}) must be divisible by ROLLOUT_TENSOR_PARALLEL_SIZE (${ROLLOUT_TENSOR_PARALLEL_SIZE})." >&2
  exit 2
fi
ROLLOUT_DATA_PARALLEL_SIZE="$((N_GPUS / ROLLOUT_TENSOR_PARALLEL_SIZE))"
# vLLM sleep-mode memory release can crash in CUDA/cumem after long runs on this
# stack. Keep rollout memory resident by default; override both to True if needed.
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-${ROLLOUT_FREE_CACHE_ENGINE}}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.99}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.06}"
PPO_CLIP_RATIO_HIGH="${PPO_CLIP_RATIO_HIGH:-0.08}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
SEED="${SEED:-3407}"
PROJECT_NAME="${PROJECT_NAME:-RankGRPO}"
LAUNCH_TIMESTAMP="${LAUNCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${BASE_MODEL_DIRNAME}_${LAUNCH_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
if [[ "${RESUME_MODE}" == "resume_path" ]]; then
  if [[ -z "${RESUME_FROM_PATH}" || ! -d "${RESUME_FROM_PATH}" ]]; then
    echo "Error: RESUME_FROM_PATH does not exist: ${RESUME_FROM_PATH}" >&2
    exit 2
  fi
fi
WANDB_MODE="${WANDB_MODE:-offline}"
DEFAULT_RAY_JOB_TAG="$(printf '%s_%s' "${EXPERIMENT_NAME}" "${CUDA_VISIBLE_DEVICES}" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-16)"
RAY_JOB_TAG="${RAY_JOB_TAG:-${DEFAULT_RAY_JOB_TAG}}"
RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR:-/tmp}/vr_${USER:-u}_${RAY_JOB_TAG}}"
RAY_TMPDIR_FALLBACK_ROOT="${RAY_TMPDIR_FALLBACK_ROOT:-${TMPDIR:-/tmp}}"
RAY_TMPDIR_MAX_LEN="${RAY_TMPDIR_MAX_LEN:-60}"
if (( ${#RAY_TMPDIR} > RAY_TMPDIR_MAX_LEN )); then
  # Ray creates deep session/socket paths under _temp_dir. Long roots can exceed
  # Linux AF_UNIX path limits, so use a short temp root for Ray only.
  SHORT_USER="${USER:-user}"
  SHORT_TAG="$(printf '%s' "${RAY_JOB_TAG}" | cut -c1-24)"
  RAY_TMPDIR="${RAY_TMPDIR_FALLBACK_ROOT}/vr_${SHORT_USER}_${SHORT_TAG}"
  echo "Warning: RAY_TMPDIR path too long, fallback to ${RAY_TMPDIR}" >&2
fi
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${RAY_TMPDIR}/spill}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$((N_GPUS * 24))}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-$((N_GPUS * 32 * 1024 * 1024 * 1024))}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-200}"
TEST_FREQ="${TEST_FREQ:-${SAVE_FREQ}}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
VAL_LOG_GENERATIONS="${VAL_LOG_GENERATIONS:-4}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
BEST_CKPT_PRUNE_ENABLE="${BEST_CKPT_PRUNE_ENABLE:-True}"
BEST_CKPTS_TO_KEEP="${BEST_CKPTS_TO_KEEP:-${TOPK_CKPT_KEEP:-3}}"
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-${TOPK_CKPT_METRIC:-eval/reward_total}}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1600}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[tensorboard]}"
REMOVE_PREVIOUS_CKPT_IN_SAVE="${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

mkdir -p "${OUTPUT_DIR}" "${RAY_TMPDIR}" "${RAY_SPILL_DIR}"

TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
export TENSORBOARD_DIR
if [[ -n "${VERL_LIB_PATH}" ]]; then
  export PYTHONPATH="${VERL_GR_ROOT}:${VERL_LIB_PATH}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${VERL_GR_ROOT}:${PYTHONPATH:-}"
fi
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
echo "GT catalog: ${GT_CATALOG_PATH}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Rec num: ${REC_NUM}"
echo "Train batch size: ${TRAIN_BATCH_SIZE}"
echo "Validation batch size: ${VAL_BATCH_SIZE}"
echo "Rollout free cache engine: ${ROLLOUT_FREE_CACHE_ENGINE}"
echo "Rollout sleep mode: ${ROLLOUT_ENABLE_SLEEP_MODE}"
echo "Rollout tensor parallel size: ${ROLLOUT_TENSOR_PARALLEL_SIZE}"
echo "Rollout data parallel size: ${ROLLOUT_DATA_PARALLEL_SIZE}"
echo "Actor max tokens/GPU: ${ACTOR_MAX_TOKENS_PER_GPU}"
echo "Log-prob max tokens/GPU: ${LOG_PROB_MAX_TOKENS_PER_GPU}"
echo "Rollout max batched tokens: ${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
echo "Rollout max sequences: ${ROLLOUT_MAX_NUM_SEQS}"
echo "Rollout GPU memory utilization: ${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "Training data parallel size: ${N_GPUS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Save/test freq: ${SAVE_FREQ}/${TEST_FREQ}"
echo "Logging steps: ${LOGGING_STEPS}"
echo "Validation generations to log: ${VAL_LOG_GENERATIONS}"
echo "Best checkpoint pruning: enable=${BEST_CKPT_PRUNE_ENABLE}, keep=${BEST_CKPTS_TO_KEEP}, metric=${BEST_CKPT_METRIC}"
echo "Output: ${OUTPUT_DIR}"
echo "Ray temp dir: ${RAY_TMPDIR}"
echo "Ray CPUs/object store/dashboard: ${RAY_NUM_CPUS}/${RAY_OBJECT_STORE_MEMORY}/${RAY_INCLUDE_DASHBOARD}"
echo "Resume mode: ${RESUME_MODE}"
if [[ -n "${RESUME_FROM_PATH}" ]]; then
  echo "Resume checkpoint: ${RESUME_FROM_PATH}"
fi
if [[ -n "${VERL_LIB_PATH}" ]]; then
  echo "verl library path: ${VERL_LIB_PATH}"
fi
echo "==================================="

for arg in "$@"; do
  if [[ "$arg" == *"Rank-GRPO"* || "$arg" == *"trl"* ]]; then
    echo "Error: TRL/reference Rank-GRPO dependency detected in argument: $arg" >&2
    echo "Use only the verl_gr Rank-GRPO recipe path." >&2
    exit 2
  fi
done

"${PYTHON_BIN}" -u -m verl_gr.trainers.main_ppo \
  --config-path "${VERL_GR_ROOT}/configs/verl_gr/rankgrpo" \
  --config-name rankgrpo_trainer \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.shuffle="${DATA_SHUFFLE}" \
  data.seed="${SEED}" \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.custom_cls.path="${RANKGRPO_RECIPE_PATH}" \
  custom_reward_function.path="${RANKGRPO_RECIPE_PATH}" \
  custom_reward_function.reward_kwargs.gt_catalog_path="${GT_CATALOG_PATH}" \
  data.rankgrpo.rec_num="${REC_NUM}" \
  algorithm.rank_grpo.rec_num="${REC_NUM}" \
  algorithm.rank_grpo.gt_catalog_path="${GT_CATALOG_PATH}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.clip_ratio="${PPO_CLIP_RATIO}" \
  actor_rollout_ref.actor.clip_ratio_low="${PPO_CLIP_RATIO}" \
  actor_rollout_ref.actor.clip_ratio_high="${PPO_CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.betas="[${ADAM_BETA1},${ADAM_BETA2}]" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
  +actor_rollout_ref.rollout.enable_sleep_mode="${ROLLOUT_ENABLE_SLEEP_MODE}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  algorithm.rank_grpo.importance_sampling_level=item \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.logging_steps="${LOGGING_STEPS}" \
  trainer.log_val_generations="${VAL_LOG_GENERATIONS}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.best_ckpt_prune_enable="${BEST_CKPT_PRUNE_ENABLE}" \
  trainer.best_ckpts_to_keep="${BEST_CKPTS_TO_KEEP}" \
  trainer.best_ckpt_metric="${BEST_CKPT_METRIC}" \
  trainer.logger="${LOGGER_BACKENDS}" \
  trainer.remove_previous_ckpt_in_save="${REMOVE_PREVIOUS_CKPT_IN_SAVE}" \
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
  +ray_kwargs.ray_init.object_store_memory="${RAY_OBJECT_STORE_MEMORY}" \
  +ray_kwargs.ray_init.include_dashboard="${RAY_INCLUDE_DASHBOARD}" \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  +ray_kwargs.ray_init.object_spilling_directory="${RAY_SPILL_DIR}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD="'${VLLM_WORKER_MULTIPROC_METHOD}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="'${NCCL_IB_DISABLE:-1}'" \
  global_profiler.save_path="${GLOBAL_PROFILER_SAVE_PATH:-${OUTPUT_DIR}/profiles}" \
  actor_rollout_ref.ref.strategy="${FSDP_STRATEGY}" \
  actor_rollout_ref.actor.strategy="${FSDP_STRATEGY}" \
  critic.enable=False \
  "$@"

