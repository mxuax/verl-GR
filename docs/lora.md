# LoRA support for MiniOneRec / OpenOneRec GRPO

This document summarizes the `lora` branch changes intended for merge into `main`. It adds **optional LoRA fine-tuning** for MiniOneRec and OpenOneRec GRPO training in verl-GR.

## Overview

LoRA is **opt-in**: `lora_rank` defaults to `0`, so behavior matches existing full-parameter training. LoRA is enabled only when `lora_rank > 0` or `lora_adapter_path` is set.

The current implementation targets the **DDP engine path** (commonly used by MiniOneRec). It reuses verl's existing PEFT / LoRA infrastructure and adds configuration, runtime wiring, checkpoint export, and launch-script support.

**Branch diff vs `main`:** 1 commit, 14 files, +505 / −24 lines.

## Main changes

### 1. LoRA configuration helpers (`verl_gr/utils/lora_config.py`)

| Function | Purpose |
| --- | --- |
| `resolve_lora_rank()` | Resolve effective rank from `model.lora.rank` or `model.lora_rank` |
| `is_lora_enabled()` | Return true when rank > 0 or an adapter path is configured |
| `should_merge_lora()` | Whether to merge LoRA into base weights before export / vLLM sync |
| `normalize_lora_config()` | Infer rank from adapter metadata when only `lora_adapter_path` is set |
| `trainable_parameters()` | Return parameters with `requires_grad=True` (adapter-only under PEFT) |

When `lora_rank == 0` and `lora_adapter_path` is unset, all helpers are no-ops and full-parameter training is unchanged.

### 2. Recipe runtime integration (`verl_gr/recipes/task_runtime.py`)

- `RecipeTaskRuntime.prepare()` calls new `configure_lora()` after rollout configuration.
- For DDP + LoRA, sets `ddp_find_unused_parameters=True` (required when base weights are frozen).
- No effect when LoRA is disabled.

### 3. DDP engine enhancements (`verl_gr/workers/engine/ddp/transformer_impl.py`)

- Uses `is_lora_enabled()` instead of a simple `lora_rank > 0` check.
- **Optimizer:** under LoRA, only trainable (adapter) parameters are optimized.
- **Checkpoint export:**
  - PEFT models: export to `lora_adapter/` and `huggingface/`, and write `lora_base_model.txt` with the base model path.
  - Full-parameter models: unchanged `huggingface/` export behavior.
- Added `disable_adapter()` to delegate to the inner PEFT module (DDP wraps `PeftModel`).
- Refactored tokenizer file copying into `_copy_tokenizer_files()`.

### 4. Hydra configuration

- **New:** `configs/verl_gr/model/lora_defaults.yaml` — reusable LoRA defaults.
- **Updated:** `configs/verl_gr/minionerec/grpo_trainer.yaml`, `configs/verl_gr/openonerec/grpo_trainer.yaml` — inline LoRA fields (`lora_rank: 0` by default).
- **Updated:** `configs/verl_gr/model/minionerec_hf_model.yaml` — composes `lora_defaults` via Hydra defaults.

Default LoRA fields:

```yaml
lora_rank: 0
lora_alpha: 16
target_modules: all-linear
exclude_modules: null
lora_adapter_path: null
lora:
  merge: false
  rank: 0
```

### 5. Launch scripts

- **New:** `scripts/lora_env.sh` — inject LoRA Hydra overrides from environment variables.

| Environment variable | Default | Description |
| --- | --- | --- |
| `LORA_RANK` | `0` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA alpha |
| `LORA_TARGET_MODULES` | `all-linear` | Target modules |
| `LORA_ADAPTER_PATH` | (empty) | Path to a pre-trained adapter |
| `LORA_MERGE` | `false` | Merge LoRA into base weights |

- **Updated:** `scripts/run_minionerec_grpo.sh`, `scripts/run_openonerec_grpo.sh` — source `lora_env.sh` and pass `LORA_OVERRIDES`.
- **Updated:** `scripts/run_minionerec_grpo_rl_aligned.sh` — logs effective `rollout.n` and `ppo_mini_batch_size`.
- **Updated:** `scripts/run_minionerec_grpo.sh` — defaults `WANDB_DISABLE_TELEMETRY=true` to avoid wandb 0.26 telemetry crashes on finish.

### 6. Tests

- `tests/minionerec/test_lora.py` — LoRA config parsing, checkpoint export, DDP optimizer (GPU).

### 7. Incidental fix (`verl_gr/recipes/openonerec/onerec_recipe.py`)

- `OneRecDataset` returns an empty dataset when no data files are configured, avoiding load failures. Unrelated to LoRA but included in the same commit.

## Usage

### Option 1: Environment variables (recommended for shell launchers)

```bash
LORA_RANK=16 LORA_ALPHA=32 bash scripts/run_minionerec_grpo.sh
```

### Option 2: Hydra CLI overrides

```bash
python -m verl_gr.trainers.main_ppo \
  --config-name minionerec/grpo_trainer_ddp \
  ++actor_rollout_ref.model.lora_rank=16 \
  ++actor_rollout_ref.model.lora_alpha=32
```

### Option 3: Load an existing adapter

```bash
LORA_ADAPTER_PATH=/path/to/adapter bash scripts/run_minionerec_grpo.sh
# rank is inferred automatically from adapter_config.json
```

### Hydra defaults composition

```yaml
defaults:
  - /model@actor_rollout_ref.model: lora_defaults
```

Or override at launch:

```bash
actor_rollout_ref.model.lora_rank=16
actor_rollout_ref.model.lora_alpha=32
actor_rollout_ref.model.target_modules=all-linear
actor_rollout_ref.model.lora_adapter_path=/path/to/adapter
```

## Design principles

1. **Backward compatible** — default full-parameter training; existing configs and scripts run unchanged.
2. **Minimal surface area** — LoRA logic is centralized in `lora_config.py` and the DDP engine; the FSDP path is not modified in this branch.
3. **Aligned with verl** — reuses `get_lora_rank_from_adapter`, `collect_lora_params`, etc. `main_ppo.py` already merges the ref policy into the actor worker when LoRA is enabled (`ref_in_actor`).

## Test plan

- [ ] `pytest tests/minionerec/test_lora.py`
- [ ] MiniOneRec GRPO with default config (`lora_rank=0`) matches `main` behavior
- [ ] `LORA_RANK=16` MiniOneRec GRPO: logs show LoRA enabled and lower VRAM usage
- [ ] After training, checkpoint contains `lora_adapter/adapter_config.json` and `lora_base_model.txt`
- [ ] `LORA_ADAPTER_PATH` correctly infers rank from adapter metadata
- [ ] OpenOneRec GRPO launcher works with LoRA enabled and disabled

## Changed files

```
configs/verl_gr/minionerec/grpo_trainer.yaml      (+7)
configs/verl_gr/model/lora_defaults.yaml          (new, +21)
configs/verl_gr/model/minionerec_hf_model.yaml    (+1)
configs/verl_gr/openonerec/grpo_trainer.yaml      (+7)
scripts/lora_env.sh                               (new, +23)
scripts/run_minionerec_grpo.sh                    (+19)
scripts/run_minionerec_grpo_rl_aligned.sh         (+9)
scripts/run_openonerec_grpo.sh                    (+11)
tests/test_lora_config.py                         (new, +104)
tests/test_lora_integration.py                    (new, +146)
verl_gr/recipes/openonerec/onerec_recipe.py       (+4)
verl_gr/recipes/task_runtime.py                   (+25)
verl_gr/utils/lora_config.py                      (new, +68)
verl_gr/workers/engine/ddp/transformer_impl.py    (+60/-24)
```
