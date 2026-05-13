#!/usr/bin/env bash
# MiniOneRec GRPO runtime launcher for verl-GR.

# -----------------------------------------------------------------------------
# 八卡示例（默认已与 MiniOneRec/rl.sh 对齐：batch 64、epochs 2、lr 1e-5、beam 16、
# temperature 1.0、Industrial CSV；在 verl-GR 根目录执行）：
#
#   cd /path/to/verl-GR
#   BASE_MODEL=/home/dyvm6xra/dyvm6xrauser49/xms-gr/MiniOneRec/output_dir/xxx/checkpoint-390 \
#   N_NODES=1 N_GPUS=8 AGENT_LOOP_NUM_WORKERS=8 \
#   PROJECT_NAME=MiniOneRec_RL EXPERIMENT_NAME=minionerec_grpo_rlalign_$(date +%Y%m%d_%H%M%S) \
#   WANDB_MODE=offline bash scripts/run_minionerec_grpo.sh \
#     trainer.save_freq=50 actor_rollout_ref.actor.use_dynamic_bsz=false
#
# 冒烟仍可提高 trainer.save_freq / 限制样本：data.train_max_samples=16 data.val_max_samples=8
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
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

RAY_INFO="$("${PYTHON_BIN}" -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes=[n for n in ray.nodes() if n.get('Alive')]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0),0); print(f'{len(nodes)} {gpus}')" 2>/dev/null || true)"
N_NODES="${N_NODES:-$(echo "${RAY_INFO}" | awk '{print $1}')}"
N_GPUS="${N_GPUS:-$(echo "${RAY_INFO}" | awk '{print $2}')}"
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

# rl.sh: num_generations=16 → beam 宽度对齐；temperature=1.0；train_batch_size=64；epochs=2；lr=1e-5
BEAM_WIDTH="${BEAM_WIDTH:-16}"
ITEM_MAX_TOKENS="${ITEM_MAX_TOKENS:-128}"
LOGPROBS_MULTIPLIER="${LOGPROBS_MULTIPLIER:-2}"
CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS="${CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS:-64}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-2}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-${N_GPUS:-1}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-40960}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
TEST_FREQ="${TEST_FREQ:-20}"
VAL_LOG_GENERATIONS="${VAL_LOG_GENERATIONS:-8}"
TASK_NAME="${TASK_NAME:-minionerec}"
TASK_CLASS_PATH="${TASK_CLASS_PATH:-verl_gr.recipes.minionerec.minionerec_recipe.MiniOneRecTask}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-1}"
DECODE_MODE_TRAIN="${DECODE_MODE_TRAIN:-hf_constrained_beam_sample}"
DECODE_MODE_VAL="${DECODE_MODE_VAL:-hf_constrained_beam_eval}"
DISABLE_CACHE_IN_TRAIN="${DISABLE_CACHE_IN_TRAIN:-true}"

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
export RAY_TMPDIR
export TMPDIR="${RAY_TMPDIR}"
export VERL_ZMQ_SOCKET_PREFIX

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
echo "Decode mode (train/val): ${DECODE_MODE_TRAIN}/${DECODE_MODE_VAL}"
echo "Train batch size: ${TRAIN_BATCH_SIZE} | epochs: ${TOTAL_EPOCHS} | lr: ${LEARNING_RATE}"
echo "Task: ${TASK_NAME} (${TASK_CLASS_PATH})"
echo "Reward workers: ${REWARD_NUM_WORKERS}"
echo "PPO micro_batch/GPU: ${PPO_MICRO_BATCH_PER_GPU} (≈rl gradient_accum_steps 2)"
echo "FSDP wrap layer: ${FSDP_TRANSFORMER_LAYERS}"
echo "Item max tokens: ${ITEM_MAX_TOKENS}"
echo "Output: ${OUTPUT_DIR}"
echo "Ray tmp (short path for Unix socket limit): ${RAY_TMPDIR}"
echo "ZMQ socket prefix: ${VERL_ZMQ_SOCKET_PREFIX}"
echo "==================================="

"${PYTHON_BIN}" -u -m verl_gr.trainers.main_ppo \
  ++task.name="${TASK_NAME}" \
  ++task.class_path="${TASK_CLASS_PATH}" \
  ++task.trainer_adapter_class="verl_gr.recipes.minionerec.minionerec_trainer.MiniOneRecTrainerAdapter" \
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  data.train_files="[${TRAIN_FILE}]" \
  data.val_files="[${VAL_FILE}]" \
  data.custom_cls.name="MiniOneRecDataset" \
  data.custom_cls.path="${MINIONEREC_RECIPE_PATH}" \
  +data.category="${CATEGORY}" \
  +data.sid_index_path="${SID_INDEX_FILE}" \
  +data.item_meta_path="${ITEM_META_FILE}" \
  +data.include_alignment_tasks=true \
  +data.include_alignment_tasks_for_val=false \
  +data.seq_title_sample="${SEQ_TITLE_SAMPLE:-10000}" \
  custom_reward_function.name="compute_score" \
  custom_reward_function.path="${MINIONEREC_REWARD_PATH}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2560}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-64}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.rollout.name="constrained_beam" \
  ++actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_PER_GPU}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="[${FSDP_TRANSFORMER_LAYERS}]" \
  actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="[${FSDP_TRANSFORMER_LAYERS}]" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  actor_rollout_ref.rollout.custom.beam_width="${BEAM_WIDTH}" \
  ++actor_rollout_ref.rollout.custom.decode_mode_train="${DECODE_MODE_TRAIN}" \
  ++actor_rollout_ref.rollout.custom.decode_mode_val="${DECODE_MODE_VAL}" \
  ++actor_rollout_ref.rollout.custom.disable_cache_in_train="${DISABLE_CACHE_IN_TRAIN}" \
  ++actor_rollout_ref.rollout.custom.constrained_beam_max_inflight_requests="${CONSTRAINED_BEAM_MAX_INFLIGHT_REQUESTS}" \
  actor_rollout_ref.rollout.custom.beam_search_params.max_tokens="${ITEM_MAX_TOKENS}" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.logprobs_multiplier="${LOGPROBS_MULTIPLIER}" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.type="minionerec_prefix_trie" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.info_file="${INFO_FILE}" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.base_model="${BASE_MODEL}" \
  ++actor_rollout_ref.rollout.custom.beam_search_params.constraint.fallback_to_eos=true \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  trainer.validation_data_dir="${VAL_DATA_DIR}" \
  ++trainer.best_ckpt_metric="${BEST_CKPT_METRIC:-val-aux/*/hr@20/mean}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.log_val_generations="${VAL_LOG_GENERATIONS}" \
  trainer.logger='[wandb]' \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  +ray_kwargs.ray_init.object_spilling_directory="${RAY_SPILL_DIR}" \
  global_profiler.save_path="${OUTPUT_DIR}/profiles" \
  critic.enable=False \
  "$@"

# -----------------------------------------------------------------------------
# 动态 batch（verl 默认语义，变长序列更省显存）：在 Hydra 末尾覆盖 use_dynamic_bsz，
# 切勿同时传入 use_dynamic_bsz=false。
#
#   ... bash scripts/run_minionerec_grpo.sh \
#     actor_rollout_ref.actor.use_dynamic_bsz=true \
#     actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
#
# （第二个参数与 actor 联动时可省略；显式写上更清晰。）
# -----------------------------------------------------------------------------

