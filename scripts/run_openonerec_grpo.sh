#!/usr/bin/env bash
# OpenOneRec GRPO runtime launcher for verl-GR.
# Mirrors compute/override flow from OpenOneRec recipe/onerec/run_grpo.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_GR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${VERL_GR_ROOT}/.." && pwd)"

# openonerec data paths only, not for code importing
OPENONEREC_ROOT="${OPENONEREC_ROOT:-${PROJECT_ROOT}/OpenOneRec}"
OPENONEREC_VERL_RL="${OPENONEREC_ROOT}/verl_rl"
LOCAL_OPENONEREC_RECIPE_ROOT="${VERL_GR_ROOT}/verl_gr/recipes/openonerec"

if [[ ! -d "${LOCAL_OPENONEREC_RECIPE_ROOT}" ]]; then
  echo "Local OpenOneRec recipe root not found: ${LOCAL_OPENONEREC_RECIPE_ROOT}" >&2
  exit 1
fi
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
  N_GPUS=8
fi

BASE_MODEL="${BASE_MODEL:-/path/to/your/model}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp}"
if [[ "${FSDP_STRATEGY}" == "fsdp2" ]]; then
  # For verl 0.7.1 + fsdp2, the non-fused actor path can hit inplace-view
  # autograd errors (logits.div_). Prefer fused kernels with torch backend.
  USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-True}"
  USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
  FUSED_KERNEL_IMPL_BACKEND="${FUSED_KERNEL_IMPL_BACKEND:-torch}"
else
  USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-False}"
  USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
  FUSED_KERNEL_IMPL_BACKEND="${FUSED_KERNEL_IMPL_BACKEND:-torch}"
fi

USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_GPUS * N_NODES))}"

ROLLOUT_N="${ROLLOUT_N:-1}"
STAGE2_BEAM_SIZE="${STAGE2_BEAM_SIZE:-32}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-2048}"
STAGE1_MAX_TOKENS="${STAGE1_MAX_TOKENS:-1024}"
STAGE2_NUM_TOKENS="${STAGE2_NUM_TOKENS:-3}"
FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-16}"
AGENT_LOOP_NUM_WORKERS="${N_GPUS:-1}"

ENABLE_THINK="${ENABLE_THINK:-False}"
ENABLE_NONTHINK="${ENABLE_NONTHINK:-False}"
USE_FORCE_PREFIX="${USE_FORCE_PREFIX:-False}"

DATA_DIR="${DATA_DIR:-${OPENONEREC_VERL_RL}/output/rl_data}"
TRAIN_FILES="${TRAIN_FILES:-[${DATA_DIR}/train.parquet]}"
VAL_FILES="${VAL_FILES:-[${DATA_DIR}/test.parquet]}"

PROJECT_NAME="${PROJECT_NAME:-OneRec_RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_two_stage}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/openonerec}"
WANDB_MODE="${WANDB_MODE:-offline}"
RAY_TMPDIR="${RAY_TMPDIR:-${OUTPUT_DIR}/ray_tmp}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${RAY_TMPDIR}/spill}"

mkdir -p "${VERL_GR_ROOT}/logs" "${OUTPUT_DIR}" "${RAY_TMPDIR}" "${RAY_SPILL_DIR}"

export PYTHONPATH="${VERL_GR_ROOT}:${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND
export WANDB_MODE
export RAY_TMPDIR
export TMPDIR="${RAY_TMPDIR}"

echo "==================================="
echo "OpenOneRec GRPO (verl-GR runtime)"
echo "==================================="
echo "OpenOneRec data root: ${OPENONEREC_ROOT}"
echo "Cluster: ${N_NODES} node(s) x ${N_GPUS} GPU(s)"
echo "Model: ${BASE_MODEL}"
echo "Rollout N: ${ROLLOUT_N}, Beam: ${STAGE2_BEAM_SIZE}"
echo "Data filter workers: ${FILTER_OVERLONG_PROMPTS_WORKERS}"
echo "Agent loop workers: ${AGENT_LOOP_NUM_WORKERS}"
echo "Use fused kernels: ${USE_FUSED_KERNELS}"
echo "Use remove padding: ${USE_REMOVE_PADDING}"
echo "FSDP strategy: ${FSDP_STRATEGY}"
echo "Fused kernel backend: ${FUSED_KERNEL_IMPL_BACKEND}"
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
  data.max_response_length="${RESPONSE_LENGTH}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS}" \
  data.custom_cls.path="${LOCAL_OPENONEREC_RECIPE_ROOT}/onerec_recipe.py" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  custom_reward_function.path="${LOCAL_OPENONEREC_RECIPE_ROOT}/onerec_recipe.py" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS}" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend="${FUSED_KERNEL_IMPL_BACKEND}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  trainer.logger='[tensorboard]' \
  trainer.remove_previous_ckpt_in_save=True \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  +ray_kwargs.ray_init.object_spilling_directory="${RAY_SPILL_DIR}" \
  global_profiler.tool="${GLOBAL_PROFILER_TOOL:-null}" \
  global_profiler.steps="${GLOBAL_PROFILER_STEPS:-null}" \
  global_profiler.profile_continuous_steps="${GLOBAL_PROFILE_CONTINUOUS_STEPS:-False}" \
  global_profiler.save_path="${GLOBAL_PROFILER_SAVE_PATH:-${OUTPUT_DIR}/profiles}" \
  actor_rollout_ref.actor.profiler.enable="${ACTOR_PROFILER_ENABLE:-False}" \
  actor_rollout_ref.actor.profiler.all_ranks="${ACTOR_PROFILER_ALL_RANKS:-False}" \
  actor_rollout_ref.actor.profiler.ranks="${ACTOR_PROFILER_RANKS:-[0]}" \
  actor_rollout_ref.ref.strategy="${FSDP_STRATEGY}" \
  actor_rollout_ref.actor.strategy="${FSDP_STRATEGY}" \
  critic.enable=False \
  "$@"

