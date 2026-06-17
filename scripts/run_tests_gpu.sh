#!/usr/bin/env bash
# Run verl-GR tests on hk01dgx036 GPUs 4-7 (vllm-gr conda env).
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-gr

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="${VERL_GR_ROOT}:${VERL_GR_ROOT}/../verl:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

cd "${VERL_GR_ROOT}"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -c "import torch; print('cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

echo "=== CPU-safe tests ==="
pytest tests/ -m "not gpu" -q --tb=line

echo "=== Hydra compose smoke ==="
pytest tests/smoke/test_startup.py::test_hydra_compose -q --tb=line

echo "=== GPU tests (LoRA DDP) ==="
pytest tests/minionerec/test_lora.py -m gpu -q --tb=line

echo "Done."
