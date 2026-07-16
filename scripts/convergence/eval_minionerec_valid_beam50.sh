#!/usr/bin/env bash
# Same-evaluator valid split (MiniOneRec evaluate.py + calc.py), beam=50.
# Usage:
#   BASE_MODEL=/path/to/hf RESULT_NAME=h113_valid_step660 \
#     bash scripts/convergence/eval_minionerec_valid_beam50.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_GR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MINIONEREC_ROOT="${MINIONEREC_ROOT:-${VERL_GR_ROOT}/../MiniOneRec}"
PYTHON_BIN="${PYTHON_BIN:-${MINIONEREC_PYTHON:-python}}"

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to a HuggingFace checkpoint directory}"
RESULT_NAME="${RESULT_NAME:-$(basename "${BASE_MODEL}")}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
NUM_BEAMS="${NUM_BEAMS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CUDA_LIST="${CUDA_LIST:-0,1,2,3}"

VALID_FILE="$(ls "${MINIONEREC_ROOT}/data/Amazon/valid/${CATEGORY}"*11.csv 2>/dev/null | head -1)"
INFO_FILE="$(ls "${MINIONEREC_ROOT}/data/Amazon/info/${CATEGORY}"*.txt 2>/dev/null | head -1)"
[[ -f "${VALID_FILE}" ]] || { echo "ERROR: valid file missing for ${CATEGORY}" >&2; exit 1; }
[[ -f "${INFO_FILE}" ]] || { echo "ERROR: info file missing for ${CATEGORY}" >&2; exit 1; }

TEMP_DIR="${MINIONEREC_ROOT}/temp/${CATEGORY}-${RESULT_NAME}"
OUT_DIR="${MINIONEREC_ROOT}/results/${RESULT_NAME}"
mkdir -p "${TEMP_DIR}" "${OUT_DIR}"

echo "[SAME_EVAL] model=${BASE_MODEL}"
echo "[SAME_EVAL] valid=${VALID_FILE}"
echo "[SAME_EVAL] result=${OUT_DIR}"

cd "${MINIONEREC_ROOT}"
"${PYTHON_BIN}" ./split.py --input_path "${VALID_FILE}" --output_path "${TEMP_DIR}" --cuda_list "${CUDA_LIST}"

IFS=',' read -r -a GPUS <<< "${CUDA_LIST}"
pids=()
for i in "${!GPUS[@]}"; do
  split_csv="${TEMP_DIR}/${i}.csv"
  [[ -f "${split_csv}" ]] || continue
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${PYTHON_BIN}" -u ./evaluate.py \
    --base_model "${BASE_MODEL}" \
    --info_file "${INFO_FILE}" \
    --category "${CATEGORY}" \
    --test_data_path "${split_csv}" \
    --result_json_data "${TEMP_DIR}/${i}.json" \
    --batch_size "${BATCH_SIZE}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens 256 \
    --length_penalty 0.0 \
    > "${TEMP_DIR}/gpu${i}.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "${pid}"; done

# Merge shards first (calc.py / fire expects a single --path file, as in evaluate.sh).
FINAL_JSON="${OUT_DIR}/final_result_${CATEGORY}.json"
"${PYTHON_BIN}" - <<PY
import json, glob
paths = sorted(glob.glob("${TEMP_DIR}/[0-9].json"))
assert paths, "ERROR: no result json"
merged = []
for p in paths:
    with open(p) as f:
        merged.extend(json.load(f))
with open("${FINAL_JSON}", "w") as f:
    json.dump(merged, f)
print(f"[SAME_EVAL] wrote ${FINAL_JSON} n={len(merged)}")
PY

"${PYTHON_BIN}" ./calc.py \
  --path "${FINAL_JSON}" \
  --item_path "${INFO_FILE}" \
  | tee "${OUT_DIR}/calc.log"
