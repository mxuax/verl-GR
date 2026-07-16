#!/usr/bin/env bash
# MiniOneRec GRPO runtime launcher for verl-GR.

# -----------------------------------------------------------------------------
# 8-GPU example (MiniOneRec DDP config, aligned with MiniOneRec/rl.sh: batch 64, 2 epochs,
# lr 1e-5, beam 16, temperature 1.0, Industrial CSV). Run from verl-GR root:
#
#   cd /path/to/verl-GR
#   BASE_MODEL=/home/dyvm6xra/dyvm6xrauser49/xms-gr/MiniOneRec/output_dir/xxx/checkpoint-390 \
#   N_NODES=1 N_GPUS=8 AGENT_LOOP_NUM_WORKERS=8 \
#   PROJECT_NAME=MiniOneRec_RL EXPERIMENT_NAME=minionerec_grpo_rlalign_$(date +%Y%m%d_%H%M%S) \
#   WANDB_MODE=offline bash scripts/run_minionerec_grpo.sh \
#     trainer.save_freq=50 actor_rollout_ref.actor.use_dynamic_bsz=false
#
# Smoke test: increase trainer.save_freq or cap samples: data.train_max_samples=64 data.val_max_samples=0
# MiniOneRec rl.sh-aligned entry: bash scripts/run_minionerec_grpo_rl_aligned.sh
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
# shellcheck source=lora_env.sh
source "${SCRIPT_DIR}/lora_env.sh"
MINIONEREC_RECIPE_PATH="${VERL_GR_ROOT}/verl_gr/recipes/minionerec/minionerec_recipe.py"
MINIONEREC_REWARD_PATH="${VERL_GR_ROOT}/verl_gr/recipes/minionerec/minionerec_reward.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

CLUSTER_CUDA_HOME="${CLUSTER_CUDA_HOME:-/cm/shared/apps/cuda12.2/toolkit/12.2.2}"
if [[ ! -x "${CUDA_HOME:-}/bin/nvcc" ]] && [[ -x "${CLUSTER_CUDA_HOME}/bin/nvcc" ]]; then
  export CUDA_HOME="${CLUSTER_CUDA_HOME}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

if [[ -z "${N_GPUS:-}" || -z "${N_NODES:-}" ]]; then
  RAY_INFO="$("${PYTHON_BIN}" -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes=[n for n in ray.nodes() if n.get('Alive')]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0),0); print(f'{len(nodes)} {gpus}')" 2>/dev/null || true)"
  N_NODES="${N_NODES:-$(echo "${RAY_INFO}" | awk '{print $1}')}"
  N_GPUS="${N_GPUS:-$(echo "${RAY_INFO}" | awk '{print $2}')}"
fi
if [[ -z "${N_NODES}" || -z "${N_GPUS}" || "${N_NODES}" == "0" ]]; then
  N_NODES=1
  N_GPUS=8
fi

BASE_MODEL="${BASE_MODEL:-/path/to/your/model}"
TRAIN_FILE="${TRAIN_FILE:-${VERL_GR_ROOT}/../MiniOneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
VAL_FILE="${VAL_FILE:-${VERL_GR_ROOT}/../MiniOneRec/data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
INFO_FILE="${INFO_FILE:-${VERL_GR_ROOT}/../MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt}"
SID_INDEX_FILE="${SID_INDEX_FILE:-${VERL_GR_ROOT}/../MiniOneRec/data/Amazon/index/Industrial_and_Scientific.index.json}"
ITEM_META_FILE="${ITEM_META_FILE:-${VERL_GR_ROOT}/../MiniOneRec/data/Amazon/index/Industrial_and_Scientific.item.json}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
BASE_MODEL_DIRNAME="$(basename "${BASE_MODEL%/}")"

# rl.sh: num_generations=16 -> beam width; temperature=1.0; train_batch_size=64
# completion rows per process, gradient_accumulation_steps=2, epochs=2, lr=1e-5.
BEAM_WIDTH="${BEAM_WIDTH:-16}"
VAL_BEAM_WIDTH="${VAL_BEAM_WIDTH:-50}"
ITEM_MAX_TOKENS="${ITEM_MAX_TOKENS:-128}"
LOGPROBS_MULTIPLIER="${LOGPROBS_MULTIPLIER:-2}"
CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS="${CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS:-64}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
ACTOR_OPTIMIZER="${ACTOR_OPTIMIZER:-paged_adamw_32bit}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-minionerec_low_var_kl}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-16}"
ORIG_TRAIN_BATCH_COMPLETIONS_PER_GPU="${ORIG_TRAIN_BATCH_COMPLETIONS_PER_GPU:-64}"
ORIG_GRAD_ACCUM_STEPS="${ORIG_GRAD_ACCUM_STEPS:-2}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-${N_GPUS:-1}}"
if [[ -z "${TRAIN_BATCH_SIZE:-}" ]]; then
  _TOTAL_ORIG_COMPLETIONS=$((ORIG_TRAIN_BATCH_COMPLETIONS_PER_GPU * N_GPUS * N_NODES * ORIG_GRAD_ACCUM_STEPS))
  if ((_TOTAL_ORIG_COMPLETIONS % BEAM_WIDTH != 0)); then
    echo "ERROR: original completion batch ${_TOTAL_ORIG_COMPLETIONS} is not divisible by BEAM_WIDTH=${BEAM_WIDTH}" >&2
    exit 1
  fi
  # verl-GR's data.train_batch_size is prompt groups per optimizer step.
  # Original MiniOneRec's train_batch_size is completion rows per process.
  TRAIN_BATCH_SIZE=$((_TOTAL_ORIG_COMPLETIONS / BEAM_WIDTH))
fi
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
TEST_FREQ="${TEST_FREQ:-20}"
VAL_LOG_GENERATIONS="${VAL_LOG_GENERATIONS:-8}"
DATA_SHUFFLE="${DATA_SHUFFLE:-true}"
DATA_SEED="${DATA_SEED:-42}"
TASK_NAME="${TASK_NAME:-minionerec}"
TASK_CLASS_PATH="${TASK_CLASS_PATH:-verl_gr.recipes.minionerec.minionerec_recipe.MiniOneRecTask}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-1}"
CONFIG_NAME="${CONFIG_NAME:-minionerec/grpo_trainer_ddp}"
DECODE_MODE_TRAIN="${DECODE_MODE_TRAIN:-hf_constrained_beam_sample}"
DECODE_MODE_VAL="${DECODE_MODE_VAL:-hf_constrained_beam_eval}"
DISABLE_CACHE_IN_TRAIN="${DISABLE_CACHE_IN_TRAIN:-true}"
MINIONEREC_FORCE_PADDED_LOGPROB="${MINIONEREC_FORCE_PADDED_LOGPROB:-false}"

FSDP_TRANSFORMER_LAYERS="${FSDP_TRANSFORMER_LAYERS:-Qwen2DecoderLayer}"

PROJECT_NAME="${PROJECT_NAME:-MiniOneRec_RL}"
LAUNCH_TIMESTAMP="${LAUNCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${BASE_MODEL_DIRNAME}_minionerec_${LAUNCH_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}}"
WANDB_MODE="${WANDB_MODE:-offline}"
# Ray uses AF_UNIX sockets under _temp_dir; Linux limits the full path to ~107 bytes.
# Defaulting ray_tmp under OUTPUT_DIR (long $HOME + experiment name) often exceeds that limit.
RAY_TMPDIR="${RAY_TMPDIR:-$(mktemp -d "/tmp/rvXXXXXX")}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${RAY_TMPDIR}/spill}"
VERL_ZMQ_SOCKET_PREFIX="${VERL_ZMQ_SOCKET_PREFIX:-verl-gr-minionerec-${LAUNCH_TIMESTAMP}-$$}"

mkdir -p "${VERL_GR_ROOT}/logs" "${OUTPUT_DIR}" "${RAY_TMPDIR}" "${RAY_SPILL_DIR}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${OUTPUT_DIR}/val_generations}"
mkdir -p "${VAL_DATA_DIR}"

export PYTHONPATH="${VERL_GR_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE
# Async rollout + wandb 0.26 telemetry can crash on finish; see verl.utils.tracking._safe_wandb_finish
export WANDB_DISABLE_TELEMETRY="${WANDB_DISABLE_TELEMETRY:-true}"
export RAY_TMPDIR
export TMPDIR="${RAY_TMPDIR}"
export VERL_ZMQ_SOCKET_PREFIX

CONFIG_NAME_FROM_ARGS=0
PREV_WAS_CONFIG_NAME=0
for arg in "$@"; do
  if [[ "${PREV_WAS_CONFIG_NAME}" -eq 1 ]]; then
    CONFIG_NAME="${arg}"
    CONFIG_NAME_FROM_ARGS=1
    PREV_WAS_CONFIG_NAME=0
    continue
  fi

  case "${arg}" in
    --config-name)
      PREV_WAS_CONFIG_NAME=1
      ;;
    --config-name=*)
      CONFIG_NAME="${arg#--config-name=}"
      CONFIG_NAME_FROM_ARGS=1
      break
      ;;
  esac
done

CONFIG_NAME_ARG=()
if [[ "${CONFIG_NAME_FROM_ARGS}" -eq 0 ]]; then
  CONFIG_NAME_ARG=(--config-name="${CONFIG_NAME}")
fi

ENGINE_OVERRIDES=()
ENGINE_FAMILY="ddp"
if [[ "${CONFIG_NAME}" != *"ddp"* ]]; then
  ENGINE_FAMILY="fsdp"
  ENGINE_OVERRIDES+=(
    "actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[${FSDP_TRANSFORMER_LAYERS}]"
    "actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[${FSDP_TRANSFORMER_LAYERS}]"
  )
else
  ENGINE_OVERRIDES+=(
    "++actor_rollout_ref.actor._target_=verl_gr.workers.config.ddp_engine.DDPActorConfig"
    "++actor_rollout_ref.actor.rollout_n=1"
    "++actor_rollout_ref.actor.strategy=ddp"
    "++actor_rollout_ref.actor.engine_config._target_=verl_gr.workers.config.ddp_engine.DDPEngineConfig"
    "++actor_rollout_ref.actor.engine_config.strategy=ddp"
    "++actor_rollout_ref.actor.engine_config.model_dtype=bf16"
    "++actor_rollout_ref.actor.engine_config.use_torch_compile=false"
    "++actor_rollout_ref.actor.engine_config.disable_flash_sdp=true"
    "++actor_rollout_ref.actor.engine_config.seed=42"
    "++actor_rollout_ref.actor.engine_config.completion_only_force_padded=${MINIONEREC_FORCE_PADDED_LOGPROB}"
    "++actor_rollout_ref.ref._target_=verl_gr.workers.config.ddp_engine.DDPActorConfig"
    "++actor_rollout_ref.ref.rollout_n=1"
    "++actor_rollout_ref.ref.strategy=ddp"
    "++actor_rollout_ref.ref.engine_config._target_=verl_gr.workers.config.ddp_engine.DDPEngineConfig"
    "++actor_rollout_ref.ref.engine_config.strategy=ddp"
    "++actor_rollout_ref.ref.engine_config.forward_only=true"
    "++actor_rollout_ref.ref.engine_config.model_dtype=bf16"
    "++actor_rollout_ref.ref.engine_config.use_torch_compile=false"
    "++actor_rollout_ref.ref.engine_config.disable_flash_sdp=true"
    "++actor_rollout_ref.ref.engine_config.seed=42"
    "++actor_rollout_ref.ref.engine_config.completion_only_force_padded=${MINIONEREC_FORCE_PADDED_LOGPROB}"
  )
fi

echo "==================================="
echo "MiniOneRec GRPO (verl-GR runtime)"
echo "==================================="
echo "Cluster: ${N_NODES} node(s) x ${N_GPUS} GPU(s)"
echo "Model: ${BASE_MODEL}"
echo "Train: ${TRAIN_FILE}"
echo "Val: ${VAL_FILE}"
echo "Info: ${INFO_FILE}"
echo "SID index: ${SID_INDEX_FILE}"
echo "Item meta: ${ITEM_META_FILE}"
echo "Beam width: ${BEAM_WIDTH} (rl.sh num_generations)"
echo "Validation beam width: ${VAL_BEAM_WIDTH}"
_NUM_GEN_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-1}"
_EFFECTIVE_ROLLOUT_N=$((_NUM_GEN_PER_PROMPT * BEAM_WIDTH))
_EFFECTIVE_PPO_MINI_BATCH=$((TRAIN_BATCH_SIZE * _EFFECTIVE_ROLLOUT_N))
_TOTAL_ORIG_COMPLETIONS=$((ORIG_TRAIN_BATCH_COMPLETIONS_PER_GPU * N_GPUS * N_NODES * ORIG_GRAD_ACCUM_STEPS))
echo "Original batch contract: ${ORIG_TRAIN_BATCH_COMPLETIONS_PER_GPU} completions/GPU x $((N_GPUS * N_NODES)) GPU(s) x GAS ${ORIG_GRAD_ACCUM_STEPS} = ${_TOTAL_ORIG_COMPLETIONS} completions/update"
echo "Rollout mode: ${ROLLOUT_MODE} | Hydra ++rollout.n=1 -> effective rollout.n=${_EFFECTIVE_ROLLOUT_N} (expand_rollout_counts)"
echo "Effective ppo_mini_batch_size: ${TRAIN_BATCH_SIZE} x ${_EFFECTIVE_ROLLOUT_N} = ${_EFFECTIVE_PPO_MINI_BATCH} (actor update chunk)"
echo "Completions/step: ${TRAIN_BATCH_SIZE} prompts x ${BEAM_WIDTH} beam = $((TRAIN_BATCH_SIZE * BEAM_WIDTH))"
echo "Decode mode (train/val): ${DECODE_MODE_TRAIN}/${DECODE_MODE_VAL}"
echo "Train batch size: ${TRAIN_BATCH_SIZE} | epochs: ${TOTAL_EPOCHS} | lr: ${LEARNING_RATE}"
echo "Actor optimizer: ${ACTOR_OPTIMIZER}"
echo "Data shuffle: ${DATA_SHUFFLE} | seed: ${DATA_SEED}"
echo "KL loss: type=${KL_LOSS_TYPE} coef=${KL_LOSS_COEF}"
echo "Completion logprob forward: force_padded=${MINIONEREC_FORCE_PADDED_LOGPROB}"
echo "Task: ${TASK_NAME} (${TASK_CLASS_PATH})"
echo "Config: ${CONFIG_NAME} | backend: ${ENGINE_FAMILY}"
echo "Reward workers: ${REWARD_NUM_WORKERS}"
echo "PPO micro_batch/GPU: ${PPO_MICRO_BATCH_PER_GPU} (actor/ref forward micro-batch rows)"
if [[ "${ENGINE_FAMILY}" == "fsdp" ]]; then
  echo "FSDP wrap layer: ${FSDP_TRANSFORMER_LAYERS}"
fi
echo "Item max tokens: ${ITEM_MAX_TOKENS}"
if [[ "${#LORA_OVERRIDES[@]}" -gt 0 ]]; then
  echo "LoRA: rank=${LORA_RANK} alpha=${LORA_ALPHA} target=${LORA_TARGET_MODULES} merge=${LORA_MERGE}"
  if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
    echo "LoRA adapter: ${LORA_ADAPTER_PATH}"
  fi
else
  echo "LoRA: disabled (full-parameter training)"
fi
echo "Output: ${OUTPUT_DIR}"
echo "Ray tmp (short path for Unix socket limit): ${RAY_TMPDIR}"
echo "ZMQ socket prefix: ${VERL_ZMQ_SOCKET_PREFIX}"
echo "==================================="

"${PYTHON_BIN}" -u -m verl_gr.trainers.main_ppo \
  "${CONFIG_NAME_ARG[@]}" \
  ++task.name="${TASK_NAME}" \
  ++task.class_path="${TASK_CLASS_PATH}" \
  ++task.trainer_adapter_class="verl_gr.recipes.minionerec.minionerec_trainer.MiniOneRecTrainerAdapter" \
  ++reward.num_workers="${REWARD_NUM_WORKERS}" \
  ++data.train_files="[${TRAIN_FILE}]" \
  ++data.val_files="[${VAL_FILE}]" \
  ++data.custom_cls.name="MiniOneRecDataset" \
  ++data.custom_cls.path="${MINIONEREC_RECIPE_PATH}" \
  ++data.category="${CATEGORY}" \
  ++data.sid_index_path="${SID_INDEX_FILE}" \
  ++data.item_meta_path="${ITEM_META_FILE}" \
  ++data.include_alignment_tasks=true \
  ++data.include_alignment_tasks_for_val=false \
  ++data.shuffle="${DATA_SHUFFLE}" \
  ++data.seed="${DATA_SEED}" \
  ++data.seq_title_sample="${SEQ_TITLE_SAMPLE:-10000}" \
  ++data.seq_title_sample_seed="${SEQ_TITLE_SAMPLE_SEED:-0}" \
  ++reward.custom_reward_function.name="compute_score" \
  ++reward.custom_reward_function.path="${MINIONEREC_REWARD_PATH}" \
  ++data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  ++data.max_prompt_length="${MAX_PROMPT_LENGTH:-2560}" \
  ++data.max_response_length="${MAX_RESPONSE_LENGTH:-64}" \
  ++actor_rollout_ref.model._target_="verl.workers.config.HFModelConfig" \
  ++actor_rollout_ref.model.external_lib="verl_gr.workers.engine.ddp" \
  ++actor_rollout_ref.model.path="${BASE_MODEL}" \
  ++actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  ++actor_rollout_ref.rollout.name="constrained_beam" \
  ++actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
  ++actor_rollout_ref.rollout.n=1 \
  ++actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  ++actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
  ++actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}" \
  ++actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  ++actor_rollout_ref.rollout.calculate_log_probs=true \
  ++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_PER_GPU}" \
  ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  ++actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  ++actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  ++actor_rollout_ref.actor.optim.optimizer="${ACTOR_OPTIMIZER}" \
  ++actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  ++actor_rollout_ref.actor.policy_loss.loss_mode=minionerec_reinforce \
  ++actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  ++actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  ++actor_rollout_ref.actor.optim.clip_grad=0.3 \
  ++actor_rollout_ref.actor.optim.weight_decay=0.0 \
  ++actor_rollout_ref.actor.use_kl_loss=True \
  ++actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  ++actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_PER_GPU}" \
  ++actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_PER_GPU}" \
  ++trainer.total_epochs="${TOTAL_EPOCHS}" \
  ++actor_rollout_ref.rollout.custom.beam_width="${BEAM_WIDTH}" \
  ++actor_rollout_ref.rollout.custom.val_beam_width="${VAL_BEAM_WIDTH}" \
  ++actor_rollout_ref.rollout.custom.decode_mode_train="${DECODE_MODE_TRAIN}" \
  ++actor_rollout_ref.rollout.custom.decode_mode_val="${DECODE_MODE_VAL}" \
  ++actor_rollout_ref.rollout.custom.disable_cache_in_train="${DISABLE_CACHE_IN_TRAIN}" \
  ++actor_rollout_ref.rollout.custom.constrained_beam_max_inflight_requests="${CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS}" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.max_tokens="${ITEM_MAX_TOKENS}" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.logprobs_multiplier="${LOGPROBS_MULTIPLIER}" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.type="minionerec_prefix_trie" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.info_file="${INFO_FILE}" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.base_model="${BASE_MODEL}" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.fallback_to_eos=true \
  ++trainer.n_gpus_per_node="${N_GPUS}" \
  ++trainer.nnodes="${N_NODES}" \
  ++trainer.project_name="${PROJECT_NAME}" \
  ++trainer.experiment_name="${EXPERIMENT_NAME}" \
  ++trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  ++trainer.validation_data_dir="${VAL_DATA_DIR}" \
  ++trainer.best_ckpt_metric="${BEST_CKPT_METRIC:-val-aux/*/hr@20/mean}" \
  ++trainer.test_freq="${TEST_FREQ}" \
  ++trainer.log_val_generations="${VAL_LOG_GENERATIONS}" \
  ++trainer.logger='[wandb]' \
  ++ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  ++ray_kwargs.ray_init.object_spilling_directory="${RAY_SPILL_DIR}" \
  ++global_profiler.save_path="${OUTPUT_DIR}/profiles" \
  ++critic.enable=False \
  "${ENGINE_OVERRIDES[@]}" \
  "${LORA_OVERRIDES[@]}" \
  "$@"

# -----------------------------------------------------------------------------
# Dynamic batching (verl default; better for variable-length sequences): override use_dynamic_bsz at the end of Hydra.
# Do not pass use_dynamic_bsz=false at the same time.
#
#   ... bash scripts/run_minionerec_grpo.sh \
#     actor_rollout_ref.actor.use_dynamic_bsz=true \
#     actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
#
# (Second arg can be omitted when tied to actor; explicit is clearer.)
# -----------------------------------------------------------------------------

