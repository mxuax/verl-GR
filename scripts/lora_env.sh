#!/usr/bin/env bash
# Optional LoRA launch overrides for verl-GR GRPO scripts.
# No effect unless LORA_RANK > 0 or LORA_ADAPTER_PATH is set.

LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
LORA_MERGE="${LORA_MERGE:-false}"

LORA_OVERRIDES=()
if [[ "${LORA_RANK}" -gt 0 || -n "${LORA_ADAPTER_PATH}" ]]; then
  # Use ++ so overrides apply on @package _ configs (MiniOneRec DDP) and _global_ configs.
  LORA_OVERRIDES+=(
    "++actor_rollout_ref.model.lora_rank=${LORA_RANK}"
    "++actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
    "++actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}"
    "++actor_rollout_ref.model.lora.merge=${LORA_MERGE}"
  )
  if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
    LORA_OVERRIDES+=("++actor_rollout_ref.model.lora_adapter_path=${LORA_ADAPTER_PATH}")
  fi
fi
