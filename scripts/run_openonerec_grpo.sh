#!/usr/bin/env bash
# OpenOneRec GRPO runtime launcher for verl-GR.
# Mirrors compute/override flow from OpenOneRec recipe/onerec/run_grpo.sh.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
PROJECT_ROOT="$(dirname "${VERL_GR_ROOT}")"
OPENONEREC_RECIPE_PATH="${PROJECT_ROOT}/verl-GR/verl_gr/recipes/openonerec/onerec_recipe.py"
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
  N_GPUS=2
fi

BASE_MODEL="${BASE_MODEL:-/path/to/your/model}"
BASE_MODEL_DIRNAME="$(basename "${BASE_MODEL%/}")"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_GPUS * N_NODES))}"

ROLLOUT_N="${ROLLOUT_N:-1}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
# Validation logging controls:
# - test_freq controls when validation runs.
# - log_val_generations controls how many samples are printed per validation.
TEST_FREQ="${TEST_FREQ:-20}"
VAL_LOG_GENERATIONS="${VAL_LOG_GENERATIONS:-8}"
VAL_DUMP_GENERATIONS="${VAL_DUMP_GENERATIONS:-True}"
# Allow explicit control at launch time, e.g.:
#   AGENT_LOOP_NUM_WORKERS=2 ./scripts/run_openonerec_grpo.sh
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-${N_GPUS:-1}}"

ENABLE_THINK="${ENABLE_THINK:-True}"
ENABLE_NONTHINK="${ENABLE_NONTHINK:-False}"
USE_FORCE_PREFIX="${USE_FORCE_PREFIX:-False}"
DATA_DIR="${VERL_GR_ROOT}/verl_gr/recipes/openonerec/output/rl_data"
TRAIN_FILES="${TRAIN_FILES:-[${DATA_DIR}/train.parquet]}"
VAL_FILES="${VAL_FILES:-[${DATA_DIR}/test.parquet]}"

PROJECT_NAME="${PROJECT_NAME:-OneRec_RL}"
LAUNCH_TIMESTAMP="${LAUNCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${BASE_MODEL_DIRNAME}_${LAUNCH_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}}"
WANDB_MODE="${WANDB_MODE:-offline}"
RAY_TMPDIR="${RAY_TMPDIR:-${OUTPUT_DIR}/ray_tmp}"
RAY_TMPDIR_FALLBACK_ROOT="${RAY_TMPDIR_FALLBACK_ROOT:-${TMPDIR:-/tmp}}"
RAY_TMPDIR_MAX_LEN="${RAY_TMPDIR_MAX_LEN:-60}"
if (( ${#RAY_TMPDIR} > RAY_TMPDIR_MAX_LEN )); then
  # Ray creates deep session/socket paths under _temp_dir. Long roots can exceed
  # Linux AF_UNIX path limits (107 bytes), so fallback to a short root.
  SHORT_USER="${USER:-user}"
  RAY_TMPDIR="${RAY_TMPDIR_FALLBACK_ROOT}/vgr_ray_${SHORT_USER}"
  echo "Warning: RAY_TMPDIR path too long, fallback to ${RAY_TMPDIR}" >&2
fi
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${RAY_TMPDIR}/spill}"

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

echo "==================================="
echo "OpenOneRec GRPO (verl-GR runtime)"
echo "==================================="
echo "Cluster: ${N_NODES} node(s) x ${N_GPUS} GPU(s)"
echo "Model: ${BASE_MODEL}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Max tokens per GPU: ${MAX_TOKENS_PER_GPU}"
echo "Validation test_freq: ${TEST_FREQ}, log_val_generations: ${VAL_LOG_GENERATIONS}"
echo "Agent loop workers: ${AGENT_LOOP_NUM_WORKERS}"
echo "FSDP strategy: ${FSDP_STRATEGY}"
echo "Output: ${OUTPUT_DIR}"
echo "Ray temp dir: ${RAY_TMPDIR}"
echo "Ray spill dir: ${RAY_SPILL_DIR}"
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

"${PYTHON_BIN}" -u -m verl_gr.trainers.main_ppo \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.enable_think="${ENABLE_THINK}" \
  data.enable_nonthink="${ENABLE_NONTHINK}" \
  data.use_force_prefix="${USE_FORCE_PREFIX}" \
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
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  ++actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
  ++actor_rollout_ref.rollout.name="two_stage" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.log_val_generations="${VAL_LOG_GENERATIONS}" \
  trainer.validation_data_dir=${VALIDATION_DATA_DIR_ARG} \
  trainer.logger='[tensorboard]' \
  trainer.remove_previous_ckpt_in_save=True \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  +ray_kwargs.ray_init.object_spilling_directory="${RAY_SPILL_DIR}" \
  global_profiler.save_path="${GLOBAL_PROFILER_SAVE_PATH:-${OUTPUT_DIR}/profiles}" \
  actor_rollout_ref.ref.strategy="${FSDP_STRATEGY}" \
  actor_rollout_ref.actor.strategy="${FSDP_STRATEGY}" \
  critic.enable=False \
  "$@"

