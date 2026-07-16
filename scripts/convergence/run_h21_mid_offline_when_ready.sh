#!/usr/bin/env bash
# Wait for H21 checkpoints and run offline HR@20 at mid steps.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_GR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN="${RUN:-minionerec_full_h21_uidfix_*}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CUDA_LIST="${CUDA_LIST:-6}"
STEPS="${STEPS:-330,660,990}"
OUT="${VERL_GR_ROOT}/logs/convergence/h21_mid_offline_${STAMP}"
MON="${VERL_GR_ROOT}/logs/convergence/experiment_monitor.log"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-gr
mkdir -p "${OUT}"

# Resolve latest H21 run if wildcard given.
if [[ "${RUN}" == *"*"* ]]; then
  RUN="$(ls -dt "${VERL_GR_ROOT}"/outputs/minionerec_full_h21_uidfix_* 2>/dev/null | head -1 | xargs basename)"
fi

echo "$(date -Iseconds) H21 mid offline waiter START run=${RUN} steps=${STEPS} out=${OUT}" >> "${MON}"

wait_ckpt() {
  local step="$1"
  local ckpt="${VERL_GR_ROOT}/outputs/${RUN}/ckpt/global_step_${step}/actor/huggingface"
  while [[ ! -f "${ckpt}/config.json" ]]; do
    sleep 180
  done
  echo "${ckpt}"
}

IFS=',' read -ra STEP_ARR <<< "${STEPS}"
for step in "${STEP_ARR[@]}"; do
  ckpt="$(wait_ckpt "${step}")"
  echo "$(date -Iseconds) H21 offline step=${step} START ckpt=${ckpt}" >> "${MON}"
  CUDA_LIST="${CUDA_LIST}" \
    BASE_MODEL="${ckpt}" \
    RESULT_NAME="h21_${RUN}_step${step}_${STAMP}" \
    bash "${VERL_GR_ROOT}/scripts/eval_minionerec_offline_test.sh" \
    > "${OUT}/step${step}.log" 2>&1
  grep "^HR" "${OUT}/step${step}.log" | tee "${OUT}/step${step}_hr.txt"
  grep "^NDCG" "${OUT}/step${step}.log" | tee "${OUT}/step${step}_ndcg.txt" || true
  echo "$(date -Iseconds) H21 offline step=${step} DONE" >> "${MON}"
done

python - <<PY | tee "${OUT}/summary.md"
import re
from pathlib import Path
out = Path("${OUT}")
steps = [int(s) for s in "${STEPS}".split(",")]
orig_hr20 = 0.17758659
print("# H21 mid offline summary\\n")
print("| step | HR@20 | Δ vs orig@full |")
print("|------|-------|----------------|")
for s in steps:
    p = out / f"step{s}_hr.txt"
    hr20 = None
    if p.exists():
        m = re.search(r"\\[([^\\]]+)\\]", p.read_text())
        if m:
            vals = [float(x) for x in m.group(1).split()]
            if len(vals) >= 5:
                hr20 = vals[4]
    if hr20 is None:
        print(f"| {s} | — | — |")
    else:
        print(f"| {s} | {hr20:.6f} | {orig_hr20 - hr20:+.6f} |")
PY

echo "$(date -Iseconds) H21 mid offline waiter DONE out=${OUT}" >> "${MON}"
echo "DONE ${OUT}"
