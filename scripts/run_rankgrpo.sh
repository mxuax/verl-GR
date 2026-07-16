#!/usr/bin/env bash
# Rank-GRPO runtime launcher for verl-GR.
# Fixed config lives in configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml.
# This script only handles what is dynamic (paths, batch calculus, Ray init).
#
# Short alignment gate (30-step logprob / KL / step-time vs TRL reference):
#   RUN_DEBUG_STEP=30 TRL_REF=/path/to/trl/tb/run bash scripts/run_rankgrpo.sh

set -euo pipefail

# ---- Environment ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
N_GPUS="${N_GPUS:-2}"

# ---- Paths ----
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
export VERL_GR_ROOT
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

if [[ -n "${VERL_LIB_PATH}" ]]; then
  export PYTHONPATH="${VERL_GR_ROOT}:${VERL_LIB_PATH}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${VERL_GR_ROOT}:${PYTHONPATH:-}"
fi

# If the caller set RAY_ADDRESS (for an isolated per-run Ray cluster), keep it.
# Otherwise clear it so Ray auto-detects or starts a default local cluster.
if [[ -n "${RAY_ADDRESS:-}" ]]; then
  echo "Using caller-provided RAY_ADDRESS=${RAY_ADDRESS}"
else
  unset RAY_ADDRESS
fi

N_NODES="${N_NODES:-1}"

# ---- Data & model paths ----
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

# verl-GR's train_batch_size and gen_batch_size are measured in unique prompts.
# TRL's RankGRPO generation_batch_size is measured in repeated generation slots:
# per_device_train_batch_size × num_processes × gradient_accumulation_steps.
# With the 2-GPU TRL reference, 4 × 2 × 6 = 48 slots, and num_generations=8
# means 48 / 8 = 6 unique prompts per optimizer update.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
GEN_BATCH_SIZE="$((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
# When gradient accumulation is active, force fixed micro-batching so the engine
# creates exactly GRADIENT_ACCUMULATION_STEPS micro-batches, accumulating gradients.
# micro_batch_size_per_gpu = TRAIN_BATCH_SIZE × ROLLOUT_N / N_GPUS
#   = prompts_per_gpu_per_microbatch × rollouts
if (( GRADIENT_ACCUMULATION_STEPS > 1 )); then
  USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
  _DEFAULT_MBS_PER_GPU="$((TRAIN_BATCH_SIZE * ROLLOUT_N / N_GPUS))"
  ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-${_DEFAULT_MBS_PER_GPU}}"
else
  USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
  ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-32}"
fi
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$((16 * N_GPUS))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-49152}"
ACTOR_MAX_TOKENS_PER_GPU="${ACTOR_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
LOG_PROB_MAX_TOKENS_PER_GPU="${LOG_PROB_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_TOKENS_PER_GPU}}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-True}"
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
ROLLOUT_CALCULATE_LOG_PROBS="${ROLLOUT_CALCULATE_LOG_PROBS:-True}"
RANKGRPO_BYPASS_OLD_LOG_PROB="${RANKGRPO_BYPASS_OLD_LOG_PROB:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-True}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-False}"
# vLLM sleep-mode memory release can crash in CUDA/cumem after long runs on this
# stack. Keep rollout memory resident by default; override both to True if needed.
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-${ROLLOUT_FREE_CACHE_ENGINE}}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.003}"
# verl's low_var_kl = k3 estimator = exp(ref-log)- (ref-log) - 1.
# This matches the reference Rank-GRPO trainer's KL computation exactly.
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
# seq-mean-token-mean = equal weight per sequence (reference behavior).
# token-mean = longer sequences dominate. Match the reference TRL trainer.
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
APPLY_EXTRA_LENGTH_SHAPING="${APPLY_EXTRA_LENGTH_SHAPING:-True}"
END_OF_LIST_REWARD="${END_OF_LIST_REWARD:-0.1}"
EXTRA_TOKEN_PENALTY="${EXTRA_TOKEN_PENALTY:--0.3}"
EARLY_STOP_PENALTY="${EARLY_STOP_PENALTY:--0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.99}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}"
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.06}"
PPO_CLIP_RATIO_HIGH="${PPO_CLIP_RATIO_HIGH:-0.08}"
# Match the Rank-GRPO reference script's epsilon=0.06 / epsilon_high=0.08
# (clip range [0.94, 1.08]). The reference trainer does not use dual-clip PPO.
# Set clip_ratio_c to a large value so the min() always picks the standard PPO
# clip branch.
PPO_CLIP_RATIO_C="${PPO_CLIP_RATIO_C:-1e6}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-False}"
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
RAY_NAMESPACE="${RAY_NAMESPACE:-${RAY_JOB_TAG}}"
VERL_ZMQ_SOCKET_PREFIX="${VERL_ZMQ_SOCKET_PREFIX:-verl-gr-rankgrpo-${LAUNCH_TIMESTAMP}-$$}"
VERL_ROLLOUT_ZMQ_NAMESPACE="${VERL_ROLLOUT_ZMQ_NAMESPACE:-rankgrpo}"
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
TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-/tmp/${USER:-u}/tvm-ffi}"
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
VERL_GR_DEBUG="${VERL_GR_DEBUG:-0}"
VERL_GR_CONVERGENCE_GATE="${VERL_GR_CONVERGENCE_GATE:-1}"
VERL_GR_KL_GROWTH_GATE="${VERL_GR_KL_GROWTH_GATE:-0}"
VERL_GR_TRL_TB_REF="${VERL_GR_TRL_TB_REF:-${TRL_REF:-}}"
VERL_GR_CONVERGENCE_STEPS="${VERL_GR_CONVERGENCE_STEPS:-}"
VERL_GR_KL_GROWTH_FLOORS="${VERL_GR_KL_GROWTH_FLOORS:-}"
VERL_GR_KL_ABS_FLOORS="${VERL_GR_KL_ABS_FLOORS:-}"
VERL_GR_EVAL_MAX_LAG="${VERL_GR_EVAL_MAX_LAG:-}"
# Length-blowout watchdog (eos_rate / clip_ratio / overflow). Disable with =0.
VERL_GR_LENGTH_GATE="${VERL_GR_LENGTH_GATE:-1}"
VERL_GR_LENGTH_GATE_MIN_STEP="${VERL_GR_LENGTH_GATE_MIN_STEP:-200}"
VERL_GR_MIN_EOS_RATE="${VERL_GR_MIN_EOS_RATE:-0.5}"
VERL_GR_MAX_CLIP_RATIO="${VERL_GR_MAX_CLIP_RATIO:-0.1}"
VERL_GR_MAX_OVERFLOW_RATIO="${VERL_GR_MAX_OVERFLOW_RATIO:-0.2}"
VERL_GR_TRUNCATE_AFTER_REC_NUM="${VERL_GR_TRUNCATE_AFTER_REC_NUM:-1}"
export VERL_GR_CONVERGENCE_GATE VERL_GR_KL_GROWTH_GATE VERL_GR_TRL_TB_REF
export VERL_GR_CONVERGENCE_STEPS VERL_GR_KL_GROWTH_FLOORS VERL_GR_KL_ABS_FLOORS VERL_GR_EVAL_MAX_LAG
export VERL_GR_LENGTH_GATE VERL_GR_LENGTH_GATE_MIN_STEP VERL_GR_MIN_EOS_RATE
export VERL_GR_MAX_CLIP_RATIO VERL_GR_MAX_OVERFLOW_RATIO VERL_GR_TRUNCATE_AFTER_REC_NUM
export REC_NUM

# ---- Debug alignment gate (RUN_DEBUG_STEP) ----
_DEFAULT_TRL_TB_REF="/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/logs/debug_precision_verlgr/runs/Jul07_03-56-22_hk01dgx028"
_DEFAULT_TRL_TRAIN_LOG="/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/logs/debug_precision_verlgr/train_20260707_035454_gpus6,7.log"
_DEBUG_ALIGN=0
_DEBUG_EXTRA_ARGS=()
if [[ -n "${RUN_DEBUG_STEP:-}" && "${RUN_DEBUG_STEP}" != "None" && "${RUN_DEBUG_STEP}" =~ ^[0-9]+$ && "${RUN_DEBUG_STEP}" -gt 0 ]]; then
  _DEBUG_ALIGN=1
  export RUN_DEBUG_STEP
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_s${RUN_DEBUG_STEP}"
  OUTPUT_DIR="${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}"

  export TRL_REF="${TRL_REF:-${VERL_GR_TRL_TB_REF:-${_DEFAULT_TRL_TB_REF}}}"
  export VERL_GR_TRL_TB_REF="${VERL_GR_TRL_TB_REF:-${TRL_REF}}"
  export VERL_GR_ALIGN_REPORT_DIR="${VERL_GR_ALIGN_REPORT_DIR:-${OUTPUT_DIR}}"
  export VERL_GR_ALIGN_GATE_EXIT="${VERL_GR_ALIGN_GATE_EXIT:-1}"
  export VERL_GR_ALIGN_LOGGING_STEPS="${VERL_GR_ALIGN_LOGGING_STEPS:-1}"
  export VERL_GR_TRL_RESUME_OFFSET="${VERL_GR_TRL_RESUME_OFFSET:-0}"
  export VERL_GR_TRL_GATE_SIDECAR="${VERL_GR_TRL_GATE_SIDECAR:-${VERL_GR_ALIGN_REPORT_DIR}/rankgrpo_gate_sidecar.json}"

  # TRL tqdm s/it benchmark (parsed from train log, not TensorBoard).
  if [[ -z "${VERL_GR_TRL_TRAIN_LOG:-}" ]]; then
    if [[ -f "${_DEFAULT_TRL_TRAIN_LOG}" ]]; then
      export VERL_GR_TRL_TRAIN_LOG="${_DEFAULT_TRL_TRAIN_LOG}"
    else
      _trl_debug_root="$(dirname "$(dirname "${VERL_GR_TRL_TB_REF}")")"
      shopt -s nullglob
      _trl_logs=( "${_trl_debug_root}"/train_*.log )
      shopt -u nullglob
      if (( ${#_trl_logs[@]} > 0 )); then
        export VERL_GR_TRL_TRAIN_LOG="${_trl_logs[0]}"
      fi
    fi
  else
    export VERL_GR_TRL_TRAIN_LOG
  fi

  mkdir -p "${VERL_GR_ALIGN_REPORT_DIR}"
  if [[ ! -f "${VERL_GR_TRL_GATE_SIDECAR}" ]]; then
    echo "Exporting TRL logprob gate sidecar from TensorBoard: ${VERL_GR_TRL_TB_REF}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/misc/rankgrpo_alignment/export_trl_gate_sidecar_from_tb.py" \
      --trl-ref "${VERL_GR_TRL_TB_REF}" \
      --output "${VERL_GR_TRL_GATE_SIDECAR}"
  else
    echo "Using existing TRL gate sidecar: ${VERL_GR_TRL_GATE_SIDECAR}"
  fi

  SAVE_FREQ=-1
  TEST_FREQ=-1
  LOGGING_STEPS=1
  DATA_SHUFFLE=False
  _DEBUG_EXTRA_ARGS+=(
    "trainer.total_training_steps=${RUN_DEBUG_STEP}"
    "trainer.logging_steps=1"
    "trainer.save_freq=-1"
    "trainer.test_freq=-1"
    "data.shuffle=false"
  )
else
  unset RUN_DEBUG_STEP 2>/dev/null || true
fi

export EXPERIMENT_NAME OUTPUT_DIR
export VERL_GR_TRL_TB_REF="${VERL_GR_TRL_TB_REF:-${TRL_REF:-}}"
export VERL_ZMQ_SOCKET_PREFIX VERL_ROLLOUT_ZMQ_NAMESPACE RAY_NAMESPACE

mkdir -p "${OUTPUT_DIR}" "${RAY_TMPDIR}" "${RAY_SPILL_DIR}"
mkdir -p "${TVM_FFI_CACHE_DIR}"

TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
export TENSORBOARD_DIR
export WANDB_MODE
export RAY_TMPDIR
export TVM_FFI_CACHE_DIR
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
echo "Train batch size: ${TRAIN_BATCH_SIZE}  (gen_batch_size: ${GEN_BATCH_SIZE}, gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS})"
echo "Train/validation shuffle: ${DATA_SHUFFLE}/${VALIDATION_SHUFFLE}"
echo "Actor model dtype: ${ACTOR_MODEL_DTYPE}"
if (( GRADIENT_ACCUMULATION_STEPS > 1 )); then
  echo "Micro-batches: ${GRADIENT_ACCUMULATION_STEPS} × ${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU} seq/GPU  (total: $((GEN_BATCH_SIZE * ROLLOUT_N)) seq, $((GEN_BATCH_SIZE * ROLLOUT_N / N_GPUS)) seq/GPU → $((GEN_BATCH_SIZE * ROLLOUT_N / N_GPUS / ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU)) micro-batches)"
fi
echo "Validation batch size: ${VAL_BATCH_SIZE}"
echo "Rollout free cache engine: ${ROLLOUT_FREE_CACHE_ENGINE}"
echo "Rollout sleep mode: ${ROLLOUT_ENABLE_SLEEP_MODE}"
echo "Rollout tensor parallel size: ${ROLLOUT_TENSOR_PARALLEL_SIZE}"
echo "Rollout data parallel size: ${ROLLOUT_DATA_PARALLEL_SIZE}"
echo "Actor max tokens/GPU: ${ACTOR_MAX_TOKENS_PER_GPU}"
echo "Log-prob max tokens/GPU: ${LOG_PROB_MAX_TOKENS_PER_GPU}"
echo "Rollout max batched tokens: ${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
echo "Rollout max sequences: ${ROLLOUT_MAX_NUM_SEQS}"
echo "Rollout disable custom all-reduce: ${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE}"
echo "Rollout GPU memory utilization: ${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "Rollout calculate log probs: ${ROLLOUT_CALCULATE_LOG_PROBS}"
echo "Rank-GRPO bypass old log prob: ${RANKGRPO_BYPASS_OLD_LOG_PROB}"
echo "Dynamic bsz/micro_batch_per_gpu/remove padding/fused kernels/activation offload: ${USE_DYNAMIC_BSZ}/${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}/${USE_REMOVE_PADDING}/${USE_FUSED_KERNELS}/${ENABLE_ACTIVATION_OFFLOAD}"
echo "Training data parallel size: ${N_GPUS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "KL loss: coef=${KL_LOSS_COEF} type=${KL_LOSS_TYPE} agg=${LOSS_AGG_MODE}"
echo "Max response length: ${MAX_RESPONSE_LENGTH}"
echo "Length shaping (apply/end/overflow/early): ${APPLY_EXTRA_LENGTH_SHAPING}/${END_OF_LIST_REWARD}/${EXTRA_TOKEN_PENALTY}/${EARLY_STOP_PENALTY}"
echo "Rollout truncate after rec_num: ${VERL_GR_TRUNCATE_AFTER_REC_NUM}"
echo "Length blowout watchdog: ${VERL_GR_LENGTH_GATE} (min_step=${VERL_GR_LENGTH_GATE_MIN_STEP})"
echo "Convergence gate (exit report): ${VERL_GR_CONVERGENCE_GATE}"
echo "Online KL watchdog: ${VERL_GR_KL_GROWTH_GATE}"
echo "Save/test freq: ${SAVE_FREQ}/${TEST_FREQ}"
echo "Logging steps: ${LOGGING_STEPS}"
echo "Validation generations to log: ${VAL_LOG_GENERATIONS}"
echo "Best checkpoint pruning: enable=${BEST_CKPT_PRUNE_ENABLE}, keep=${BEST_CKPTS_TO_KEEP}, metric=${BEST_CKPT_METRIC}"
if (( _DEBUG_ALIGN )); then
  echo "Alignment gate: RUN_DEBUG_STEP=${RUN_DEBUG_STEP} (logprob + KL per-step, step-time avg vs TRL tqdm, modular timing_s/* report)"
  echo "  TRL TB reference: ${VERL_GR_TRL_TB_REF}"
  echo "  TRL gate sidecar: ${VERL_GR_TRL_GATE_SIDECAR}"
  if [[ -n "${VERL_GR_TRL_TRAIN_LOG:-}" ]]; then
    echo "  TRL train log (tqdm s/it): ${VERL_GR_TRL_TRAIN_LOG}"
  else
    echo "  TRL train log (tqdm s/it): (unset — set VERL_GR_TRL_TRAIN_LOG for step-time gate)"
  fi
  echo "  Alignment report dir: ${VERL_GR_ALIGN_REPORT_DIR}/logs"
  echo "  Gate exit on fail: ${VERL_GR_ALIGN_GATE_EXIT}"
fi
if [[ -n "${VERL_GR_TRL_TB_REF:-}" ]] && (( ! _DEBUG_ALIGN )); then
  echo "TRL TB reference: ${VERL_GR_TRL_TB_REF}"
fi
echo "Debug mode: ${VERL_GR_DEBUG}"
echo "Output: ${OUTPUT_DIR}"
echo "Ray temp dir: ${RAY_TMPDIR}"
echo "Ray namespace: ${RAY_NAMESPACE}"
echo "ZMQ socket prefix: ${VERL_ZMQ_SOCKET_PREFIX}"
echo "ZMQ rollout namespace: ${VERL_ROLLOUT_ZMQ_NAMESPACE}"
echo "TVM FFI cache dir: ${TVM_FFI_CACHE_DIR}"
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
  ++data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.shuffle="${DATA_SHUFFLE}" \
  ++data.validation_shuffle="${VALIDATION_SHUFFLE}" \
  data.seed="${SEED}" \
  data.max_prompt_length=2048 \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.custom_cls.path="${RANKGRPO_RECIPE_PATH}" \
  custom_reward_function.path="${RANKGRPO_RECIPE_PATH}" \
  custom_reward_function.reward_kwargs.gt_catalog_path="${GT_CATALOG_PATH}" \
  data.rankgrpo.rec_num="${REC_NUM}" \
  algorithm.rank_grpo.rec_num="${REC_NUM}" \
  algorithm.rank_grpo.gt_catalog_path="${GT_CATALOG_PATH}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  ++actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  ++actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
  actor_rollout_ref.actor.clip_ratio="${PPO_CLIP_RATIO}" \
  actor_rollout_ref.actor.clip_ratio_low="${PPO_CLIP_RATIO}" \
  actor_rollout_ref.actor.clip_ratio_high="${PPO_CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c="${PPO_CLIP_RATIO_C}" \
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
  actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.calculate_log_probs="${ROLLOUT_CALCULATE_LOG_PROBS}" \
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
  actor_rollout_ref.rollout.enable_sleep_mode="${ROLLOUT_ENABLE_SLEEP_MODE}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_activation_offload="${ENABLE_ACTIVATION_OFFLOAD}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
  actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_gr.recipes.rankgrpo.rankgrpo_agent_loop.RankGRPOAgentLoopManager \
  actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
  ++actor_rollout_ref.rollout.name=rankgrpo \
  ++actor_rollout_ref.rollout.mode=async \
  algorithm.rollout_correction.bypass_mode="${RANKGRPO_BYPASS_OLD_LOG_PROB}" \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.loss_type=ppo_clip \
  algorithm.rank_grpo.importance_sampling_level=item \
  algorithm.rank_grpo.apply_extra_length_shaping="${APPLY_EXTRA_LENGTH_SHAPING}" \
  algorithm.rank_grpo.end_of_list_reward="${END_OF_LIST_REWARD}" \
  algorithm.rank_grpo.extra_token_penalty="${EXTRA_TOKEN_PENALTY}" \
  algorithm.rank_grpo.early_stop_penalty="${EARLY_STOP_PENALTY}" \
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
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD="'${VLLM_WORKER_MULTIPROC_METHOD}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF="'${PYTORCH_CUDA_ALLOC_CONF}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="'${NCCL_IB_DISABLE:-1}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_DEBUG="'${VERL_GR_DEBUG}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_CONVERGENCE_GATE="'${VERL_GR_CONVERGENCE_GATE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_KL_GROWTH_GATE="'${VERL_GR_KL_GROWTH_GATE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_TRL_TB_REF="'${VERL_GR_TRL_TB_REF}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_CONVERGENCE_STEPS="'${VERL_GR_CONVERGENCE_STEPS}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_KL_GROWTH_FLOORS="'${VERL_GR_KL_GROWTH_FLOORS}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_KL_ABS_FLOORS="'${VERL_GR_KL_ABS_FLOORS}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_EVAL_MAX_LAG="'${VERL_GR_EVAL_MAX_LAG}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_LENGTH_GATE="'${VERL_GR_LENGTH_GATE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_LENGTH_GATE_MIN_STEP="'${VERL_GR_LENGTH_GATE_MIN_STEP}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_MIN_EOS_RATE="'${VERL_GR_MIN_EOS_RATE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_MAX_CLIP_RATIO="'${VERL_GR_MAX_CLIP_RATIO}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_MAX_OVERFLOW_RATIO="'${VERL_GR_MAX_OVERFLOW_RATIO}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_TRUNCATE_AFTER_REC_NUM="'${VERL_GR_TRUNCATE_AFTER_REC_NUM}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.REC_NUM="'${REC_NUM}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.EXPERIMENT_NAME="'${EXPERIMENT_NAME}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.OUTPUT_DIR="'${OUTPUT_DIR}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_ROOT="'${VERL_GR_ROOT}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TVM_FFI_CACHE_DIR="'${TVM_FFI_CACHE_DIR}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="'${PYTHONPATH:-}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ZMQ_SOCKET_PREFIX="'${VERL_ZMQ_SOCKET_PREFIX}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_ZMQ_NAMESPACE="'${VERL_ROLLOUT_ZMQ_NAMESPACE}'" \
  $(  # Propagate alignment-gate env to Ray workers when RUN_DEBUG_STEP is set
    if (( _DEBUG_ALIGN )); then
      echo "+ray_kwargs.ray_init.runtime_env.env_vars.RUN_DEBUG_STEP='${RUN_DEBUG_STEP}'"
      echo "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_TRL_TB_REF='${VERL_GR_TRL_TB_REF}'"
      echo "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_TRL_GATE_SIDECAR='${VERL_GR_TRL_GATE_SIDECAR}'"
      echo "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_ALIGN_REPORT_DIR='${VERL_GR_ALIGN_REPORT_DIR}'"
      echo "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_ALIGN_GATE_EXIT='${VERL_GR_ALIGN_GATE_EXIT}'"
      echo "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_TRL_RESUME_OFFSET='${VERL_GR_TRL_RESUME_OFFSET}'"
      if [[ -n "${VERL_GR_TRL_TRAIN_LOG:-}" ]]; then
        echo "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_TRL_TRAIN_LOG='${VERL_GR_TRL_TRAIN_LOG}'"
      fi
    elif [[ -n "${VERL_GR_TRL_TB_REF:-}" ]]; then
      echo "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_GR_TRL_TB_REF='${VERL_GR_TRL_TB_REF}'"
    fi
  ) \
  $(  # Ray cluster-creation args — only when we aren't connecting to a pre-existing cluster
    if [[ -z "${RAY_ADDRESS:-}" ]]; then
      echo "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}"
      echo "+ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY}"
      echo "+ray_kwargs.ray_init.include_dashboard=${RAY_INCLUDE_DASHBOARD}"
      echo "+ray_kwargs.ray_init._temp_dir=${RAY_TMPDIR}"
      echo "+ray_kwargs.ray_init.object_spilling_directory=${RAY_SPILL_DIR}"
      echo "+ray_kwargs.ray_init.namespace=${RAY_NAMESPACE}"
    fi
  ) \
  global_profiler.save_path="${GLOBAL_PROFILER_SAVE_PATH:-${OUTPUT_DIR}/profiles}" \
  actor_rollout_ref.ref.strategy="${FSDP_STRATEGY}" \
  actor_rollout_ref.actor.strategy="${FSDP_STRATEGY}" \
  critic.enable=False \
  "${_DEBUG_EXTRA_ARGS[@]}" \
  "$@"
