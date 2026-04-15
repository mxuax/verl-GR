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
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
if [[ "${FSDP_STRATEGY}" == "fsdp2" ]]; then
  USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-True}"
  USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
  FUSED_KERNEL_IMPL_BACKEND="${FUSED_KERNEL_IMPL_BACKEND:-triton}"
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

mkdir -p "${VERL_GR_ROOT}/logs" "${OUTPUT_DIR}" "${RAY_TMPDIR}"

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
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.max_prompt_length=10240 \
  ++data.enable_think="${ENABLE_THINK}" \
  ++data.enable_nonthink="${ENABLE_NONTHINK}" \
  ++data.use_force_prefix="${USE_FORCE_PREFIX}" \
  data.prompt_key=prompt \
  data.shuffle=True \
  data.max_response_length="${RESPONSE_LENGTH}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS}" \
  data.truncation=error \
  data.custom_cls.path="${LOCAL_OPENONEREC_RECIPE_ROOT}/onerec_recipe.py" \
  data.custom_cls.name=OneRecDataset \
  data.reward_fn_key=source \
  ++data.data_source_key=source \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  custom_reward_function.path="${LOCAL_OPENONEREC_RECIPE_ROOT}/onerec_recipe.py" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_seqs=2048 \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS}" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend="${FUSED_KERNEL_IMPL_BACKEND}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=320 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  trainer.total_epochs=20 \
  trainer.val_before_train=True \
  ++trainer.logger=tensorboard \
  actor_rollout_ref.ref.strategy="${FSDP_STRATEGY}" \
  actor_rollout_ref.actor.strategy="${FSDP_STRATEGY}" \
  ++critic.enable=False \
  ++actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer] \
  ++actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer] \
  ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  "$@"

