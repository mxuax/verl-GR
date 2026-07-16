#!/usr/bin/env bash
# OpenOneRec GRPO runtime launcher for verl-GR.
# Mirrors compute/override flow from OpenOneRec recipe/onerec/run_grpo.sh.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
# shellcheck source=lora_env.sh
source "${SCRIPT_DIR}/lora_env.sh"
OPENONEREC_RECIPE_PATH="${OPENONEREC_RECIPE_PATH:-${VERL_GR_ROOT}/verl_gr/recipes/openonerec/onerec_recipe.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

# Cluster auto-discovery via Ray (fallback to single node defaults).
RAY_INFO="$("${PYTHON_BIN}" -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes=[n for n in ray.nodes() if n.get('Alive')]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0),0); print(f'{len(nodes)} {gpus}')" 2>/dev/null || true)"
N_NODES="${N_NODES:-$(echo "${RAY_INFO}" | awk '{print $1}')}"
N_GPUS="${N_GPUS:-$(echo "${RAY_INFO}" | awk '{print $2}')}"
if [[ -z "${N_NODES}" || -z "${N_GPUS}" || "${N_NODES}" == "0" ]]; then
  N_NODES=1
  # N_GPUS=2
fi

BASE_MODEL="${BASE_MODEL:-/path/to/your/model}"
BASE_MODEL_DIRNAME="$(basename "${BASE_MODEL%/}")"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
NORM_ADV_BY_STD_IN_GRPO="${NORM_ADV_BY_STD_IN_GRPO:-True}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2048}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_GPUS * N_NODES))}"

ROLLOUT_N="${ROLLOUT_N:-1}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
# Validation logging controls:
# - test_freq controls when validation runs.
# - log_val_generations controls how many samples are printed per validation.
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-${TEST_FREQ}}"
VAL_LOG_GENERATIONS="${VAL_LOG_GENERATIONS:-4}"
VAL_DUMP_GENERATIONS="${VAL_DUMP_GENERATIONS:-True}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-100}"
VALIDATION_ADAPTIVE_CONCURRENCY="${VALIDATION_ADAPTIVE_CONCURRENCY:-True}"
VALIDATION_MIN_CONCURRENT_REQUESTS="${VALIDATION_MIN_CONCURRENT_REQUESTS:-64}"
VALIDATION_MAX_CONCURRENT_REQUESTS="${VALIDATION_MAX_CONCURRENT_REQUESTS:-512}"
VALIDATION_TARGET_GPU_UTILIZATION="${VALIDATION_TARGET_GPU_UTILIZATION:-85.0}"
VALIDATION_GPU_UTIL_TOLERANCE="${VALIDATION_GPU_UTIL_TOLERANCE:-7.5}"
VALIDATION_CONCURRENCY_STEP="${VALIDATION_CONCURRENCY_STEP:-64}"
VAL_THINKING_TEMPERATURE="${VAL_THINKING_TEMPERATURE:-0.6}"
VAL_THINKING_TOP_P="${VAL_THINKING_TOP_P:-0.95}"
VAL_THINKING_TOP_K="${VAL_THINKING_TOP_K:-50}"
BEST_CKPTS_TO_KEEP="${BEST_CKPTS_TO_KEEP:-3}"
BEST_CKPT_PRUNE_ENABLE="${BEST_CKPT_PRUNE_ENABLE:-True}"
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-val-aux/*/pass_at_32/mean}"
# Allow explicit control at launch time, e.g.:
#   AGENT_LOOP_NUM_WORKERS=2 ./scripts/run_openonerec_grpo.sh
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-${N_GPUS:-1}}"

ENABLE_THINK="${ENABLE_THINK:-False}"
ENABLE_NONTHINK="${ENABLE_NONTHINK:-False}"
USE_FORCE_PREFIX="${USE_FORCE_PREFIX:-False}"
DATA_DIR="${DATA_DIR:-${VERL_GR_ROOT}/verl_gr/recipes/openonerec/output/rl_data}"
TRAIN_FILES="${TRAIN_FILES:-[${DATA_DIR}/train.parquet]}"
VAL_FILES="${VAL_FILES:-[${DATA_DIR}/test.parquet]}"

PROJECT_NAME="${PROJECT_NAME:-OneRec_RL}"
LAUNCH_TIMESTAMP="${LAUNCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${BASE_MODEL_DIRNAME}_${LAUNCH_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}}"
WANDB_MODE="${WANDB_MODE:-offline}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[tensorboard]}"
RAY_TMPDIR="${RAY_TMPDIR:-${OUTPUT_DIR}/ray_tmp}"
RAY_TMPDIR_FALLBACK_ROOT="${RAY_TMPDIR_FALLBACK_ROOT:-${TMPDIR:-/tmp}}"
RAY_TMPDIR_MAX_LEN="${RAY_TMPDIR_MAX_LEN:-30}"
if (( ${#RAY_TMPDIR} > RAY_TMPDIR_MAX_LEN )); then
  # Ray appends session_*/sockets/plasma_store under _temp_dir; keep root short.
  SHORT_TAG="$(printf '%s' "${RAY_JOB_TAG:-$$}" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-16)"
  RAY_TMPDIR="${RAY_TMPDIR_FALLBACK_ROOT}/vr_${SHORT_TAG}"
  echo "Warning: RAY_TMPDIR path too long, fallback to ${RAY_TMPDIR}" >&2
fi
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${RAY_TMPDIR}/spill}"
VERL_ZMQ_SOCKET_PREFIX="${VERL_ZMQ_SOCKET_PREFIX:-verl-gr-openonerec-${LAUNCH_TIMESTAMP}-$$}"
VERL_ROLLOUT_ZMQ_NAMESPACE="${VERL_ROLLOUT_ZMQ_NAMESPACE:-openonerec}"
RAY_NAMESPACE="${RAY_NAMESPACE:-openonerec_${LAUNCH_TIMESTAMP}_$$}"

mkdir -p "${VERL_GR_ROOT}/logs" "${OUTPUT_DIR}" "${RAY_TMPDIR}" "${RAY_SPILL_DIR}"
if [[ "${VAL_DUMP_GENERATIONS}" == "True" ]]; then
  VAL_DATA_DIR="${VAL_DATA_DIR:-${OUTPUT_DIR}/val_generations}"
  mkdir -p "${VAL_DATA_DIR}"
  VALIDATION_DATA_DIR_ARG="${VAL_DATA_DIR}"
else
  VALIDATION_DATA_DIR_ARG="null"
fi

TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
export TENSORBOARD_DIR
export PYTHONPATH="${VERL_GR_ROOT}:${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND
export WANDB_MODE
export RAY_TMPDIR
export TMPDIR="${RAY_TMPDIR}"
export VERL_ZMQ_SOCKET_PREFIX
export VERL_ROLLOUT_ZMQ_NAMESPACE

echo "==================================="
echo "OpenOneRec GRPO (verl-GR runtime)"
echo "==================================="
echo "Cluster: ${N_NODES} node(s) x ${N_GPUS} GPU(s)"
echo "Model: ${BASE_MODEL}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Max tokens per GPU: ${MAX_TOKENS_PER_GPU}"
echo "Validation test_freq: ${TEST_FREQ}, save_freq: ${SAVE_FREQ}, log_val_generations: ${VAL_LOG_GENERATIONS}"
echo "Validation max samples: ${VAL_MAX_SAMPLES}, train max samples: ${TRAIN_MAX_SAMPLES}, val batch size: ${VAL_BATCH_SIZE}"
echo "Validation adaptive concurrency: ${VALIDATION_ADAPTIVE_CONCURRENCY}"
echo "Validation min/max concurrent requests: ${VALIDATION_MIN_CONCURRENT_REQUESTS}/${VALIDATION_MAX_CONCURRENT_REQUESTS}"
echo "Validation target gpu util +/- tol: ${VALIDATION_TARGET_GPU_UTILIZATION}% +/- ${VALIDATION_GPU_UTIL_TOLERANCE}%"
echo "Validation concurrency step: ${VALIDATION_CONCURRENCY_STEP}"
echo "Agent loop workers: ${AGENT_LOOP_NUM_WORKERS}"
echo "FSDP strategy: ${FSDP_STRATEGY}"
if [[ "${#LORA_OVERRIDES[@]}" -gt 0 ]]; then
  echo "LoRA: rank=${LORA_RANK} alpha=${LORA_ALPHA} target=${LORA_TARGET_MODULES} merge=${LORA_MERGE}"
  if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
    echo "LoRA adapter: ${LORA_ADAPTER_PATH}"
  fi
else
  echo "LoRA: disabled (full-parameter training)"
fi
echo "Output: ${OUTPUT_DIR}"
echo "TensorBoard: ${TENSORBOARD_DIR}"
echo "Logger backends: ${LOGGER_BACKENDS}"
echo "Ray temp dir: ${RAY_TMPDIR}"
echo "Ray spill dir: ${RAY_SPILL_DIR}"
echo "ZMQ socket prefix: ${VERL_ZMQ_SOCKET_PREFIX}"
echo "ZMQ rollout namespace: ${VERL_ROLLOUT_ZMQ_NAMESPACE}"
echo "Ray namespace: ${RAY_NAMESPACE}"
echo "==================================="

# Guardrail: block accidental fallback to legacy OpenOneRec recipe imports.
for arg in "$@"; do
  if [[ "$arg" == *"recipe/onerec"* || "$arg" == *"recipe.onerec"* ]]; then
    echo "Error: legacy OpenOneRec recipe reference detected in argument: $arg" >&2
    echo "Use the verl_gr.trainers.main_ppo OpenOneRec launch flow only." >&2
    exit 2
  fi
  if [[ "$arg" == *"transformer_layer_cls_to_wrap={"* ]]; then
    echo "Error: invalid set-style transformer_layer_cls_to_wrap detected: $arg" >&2
    echo "Use list style [...], e.g. [Qwen3DecoderLayer]." >&2
    exit 2
  fi
done

# avoid conflicts: trainer.val_before_train=False needs to be false to avoid 2nd run of val_in_train, as now we have a built-in val run for keeping the ckpt top-k
"${PYTHON_BIN}" -u -m verl_gr.trainers.main_ppo \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.enable_think="${ENABLE_THINK}" \
  data.enable_nonthink="${ENABLE_NONTHINK}" \
  data.use_force_prefix="${USE_FORCE_PREFIX}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.shuffle=false \
  ++data.validation_shuffle=false \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.custom_cls.path="${OPENONEREC_RECIPE_PATH}" \
  custom_reward_function.path="${OPENONEREC_RECIPE_PATH}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.custom.validation_adaptive_concurrency="${VALIDATION_ADAPTIVE_CONCURRENCY}" \
  actor_rollout_ref.rollout.custom.validation_min_concurrent_requests="${VALIDATION_MIN_CONCURRENT_REQUESTS}" \
  actor_rollout_ref.rollout.custom.validation_max_concurrent_requests="${VALIDATION_MAX_CONCURRENT_REQUESTS}" \
  actor_rollout_ref.rollout.custom.validation_target_gpu_utilization="${VALIDATION_TARGET_GPU_UTILIZATION}" \
  actor_rollout_ref.rollout.custom.validation_gpu_util_tolerance="${VALIDATION_GPU_UTIL_TOLERANCE}" \
  actor_rollout_ref.rollout.custom.validation_concurrency_step="${VALIDATION_CONCURRENCY_STEP}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_THINKING_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_THINKING_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_THINKING_TOP_K}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  ++actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
  ++actor_rollout_ref.rollout.name="two_stage" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
  algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.log_val_generations="${VAL_LOG_GENERATIONS}" \
  trainer.validation_data_dir=${VALIDATION_DATA_DIR_ARG} \
  ++trainer.best_ckpt_prune_enable="${BEST_CKPT_PRUNE_ENABLE}" \
  ++trainer.best_ckpts_to_keep="${BEST_CKPTS_TO_KEEP}" \
  ++trainer.best_ckpt_metric="${BEST_CKPT_METRIC}" \
  trainer.logger="${LOGGER_BACKENDS}" \
  trainer.remove_previous_ckpt_in_save=False \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  +ray_kwargs.ray_init.object_spilling_directory="${RAY_SPILL_DIR}" \
  +ray_kwargs.ray_init.namespace="${RAY_NAMESPACE}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ZMQ_SOCKET_PREFIX="'${VERL_ZMQ_SOCKET_PREFIX}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_ZMQ_NAMESPACE="'${VERL_ROLLOUT_ZMQ_NAMESPACE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="'${PYTHONPATH:-}'" \
  global_profiler.save_path="${GLOBAL_PROFILER_SAVE_PATH:-${OUTPUT_DIR}/profiles}" \
  actor_rollout_ref.ref.strategy="${FSDP_STRATEGY}" \
  actor_rollout_ref.actor.strategy="${FSDP_STRATEGY}" \
  critic.enable=False \
  "${LORA_OVERRIDES[@]}" \
  "$@"

