#!/usr/bin/env bash
set -euo pipefail
cd /home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 | tr -d " ")
  if [ "${used}" -lt 5000 ]; then break; fi
  echo "$(date +%H:%M:%S) GPU4 used=${used}MiB, waiting..."
  sleep 120
done
export STAMP=20260703_manifest_v3 GPU_FREE_MAX_MIB=20000
exec bash scripts/convergence/run_verl_4gpu_prefix_330step.sh
