#!/usr/bin/env bash
# H22: uid fix + pandas seq_title2sid sampling (seed=0). Same hparams as H21.
set -eo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,6,7}"
export N_GPUS="${N_GPUS:-4}"
export STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-minionerec_full_h22_seqsample_${STAMP}}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_GR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG="${VERL_GR_ROOT}/logs/convergence/verl_full_h22_seqsample_${STAMP}.log"
MON="${VERL_GR_ROOT}/logs/convergence/monitor_h22_seqsample_${STAMP}.log"
METRICS_JSONL="${VERL_GR_ROOT}/logs/convergence/h22_metrics_${STAMP}.jsonl"
EXP_MON="${VERL_GR_ROOT}/logs/convergence/experiment_monitor.log"

mkdir -p "${VERL_GR_ROOT}/logs/convergence"
: > "${METRICS_JSONL}"

echo "H22 seq_title pandas-sample fix | lr=${LEARNING_RATE} | GPUs=${CUDA_VISIBLE_DEVICES}"
echo "  exp=${EXPERIMENT_NAME}"
echo "  log=${LOG}"
echo "  metrics=${METRICS_JSONL}"

cd "${VERL_GR_ROOT}"
export VERL_GR_METRICS_JSONL="${METRICS_JSONL}"

nohup bash "${SCRIPT_DIR}/run_verl_full_2epoch.sh" \
  ++actor_rollout_ref.actor.engine_config.model_dtype=fp32 \
  ++actor_rollout_ref.ref.engine_config.model_dtype=fp32 \
  ++actor_rollout_ref.ref.sync_freq=512 \
  ++actor_rollout_ref.ref.ref_model_mixup_alpha=0.6 \
  ++actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  ++data.seq_title_sample_seed=0 \
  ++trainer.default_local_dir="${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}/ckpt" \
  ++trainer.validation_data_dir="${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}/val_generations" \
  ++trainer.experiment_name="${EXPERIMENT_NAME}" \
  > "${LOG}" 2>&1 &

TRAIN_PID=$!
echo "train PID=${TRAIN_PID}"
echo "$(date -Iseconds) H22 seqsample full STARTED exp=${EXPERIMENT_NAME} pid=${TRAIN_PID} log=${LOG} metrics=${METRICS_JSONL}" >> "${EXP_MON}"

nohup bash -c "
LOG='${LOG}'
MON='${MON}'
METRICS='${METRICS_JSONL}'
EXP='${EXPERIMENT_NAME}'
VERL_GR='${VERL_GR_ROOT}'
STAMP='${STAMP}'
while true; do
  s=\$(grep -oE '[0-9]+/3298' \"\$LOG\" 2>/dev/null | tail -1)
  echo \"\$(date -Iseconds) h22_seqsample step=\$s\" >> \"\$MON\"
  echo \"\$(date -Iseconds) h22_seqsample step=\$s\" >> \"${EXP_MON}\"
  if [ -f \"\$METRICS\" ]; then
    python \"\$VERL_GR/scripts/convergence/analyze_h21_metrics.py\" \
      --metrics \"\$METRICS\" --tail 1 >> \"\$MON\" 2>/dev/null || true
  fi
  if [ \"\$s\" = '3298/3298' ]; then
    echo \"\$(date -Iseconds) TRAINING_DONE\" >> \"\$MON\"
    cd \"\$VERL_GR\"
    export CUDA_VISIBLE_DEVICES=0,1,6,7 CUDA_LIST=0,1,6,7
    export VERL_RUN=\"\$VERL_GR/outputs/\$EXP\" VERL_STEP=3298 STAMP=\${STAMP}_eval
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate vllm-gr
    nohup bash scripts/convergence/run_compare_full_2epoch.sh \
      > logs/convergence/compare_full_h22_seqsample_\${STAMP}.log 2>&1 &
    echo \"\$(date -Iseconds) h22 eval PID=\$!\" >> \"\$MON\"
    echo \"\$(date -Iseconds) h22_seqsample eval started exp=\$EXP\" >> \"${EXP_MON}\"
    break
  fi
  if ! kill -0 ${TRAIN_PID} 2>/dev/null; then
    echo \"\$(date -Iseconds) PROCESS_EXITED step=\$s\" >> \"\$MON\"
    tail -30 \"\$LOG\" >> \"\$MON\"
    break
  fi
  sleep 300
done
" > /dev/null 2>&1 &

echo "monitor started; tail -f ${LOG}"
