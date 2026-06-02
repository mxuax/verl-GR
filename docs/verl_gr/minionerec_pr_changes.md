# MiniOneRec: workingbranch changes vs `main`

This document summarizes what the `workingbranch` adds on top of `main` for MiniOneRec GRPO in verl-GR. It is intended for PR reviewers comparing against `origin/main`.

## Scope

`main` in this repository already contains OpenOneRec / RankGRPO scaffolding and a large trainer refactor. The working branch **introduces the full MiniOneRec recipe** and **performance / training-alignment work** that is not on `main`.

## 1. MiniOneRec recipe (new on `main`)

| Area | Files | Behavior |
| --- | --- | --- |
| Dataset | `minionerec_dataset.py`, `minionerec_format.py` | `SidDataset`-style prompts, alignment tasks (`RLTitle2Sid`, `RLSeqTitle2Sid` with `seq_title_sample=10000`). |
| Rollout | `constrained_beam_agent_loop.py`, `hf_constrained_generation.py`, `minionerec_fsdp_workers.py` | HF constrained beam (primary path for DDP); prefix-trie constraint from `info_file`. |
| Reward | `minionerec_reward.py` | Rule + NDCG-style group ranking; **training penalties** for empty / invalid SID (`task.reward_penalties` in config). |
| Loss | `minionerec_loss.py` | `minionerec_reinforce` policy loss (skips heavy `old_log_prob` forward when configured). |
| Trainer adapter | `minionerec_trainer.py` | Validation HR/NDCG, reward postprocess, gen-batch prep. |
| Task wiring | `minionerec_recipe.py` | Registers constrained beam rollout, DDP worker, optimizations (below). |

See also: [minionerec_mapping.md](./minionerec_mapping.md) for the behavioral contract vs upstream MiniOneRec.

## 2. DDP training path (new)

- Config: `configs/verl_gr/minionerec/grpo_trainer_ddp.yaml`, `minionerec_ddp_actor.yaml`, `minionerec_ddp_ref.yaml`.
- Engine: `verl_gr/workers/engine/ddp/transformer_impl.py` — PyTorch DDP LM-head engine for smaller models (faster than FSDP for this workload).
- Launcher: `scripts/run_minionerec_grpo_rl_aligned.sh` maps `MiniOneRec/rl.sh` hyperparameters:

| `rl.sh` | verl-GR (`run_minionerec_grpo_rl_aligned.sh`) |
| --- | --- |
| 4 processes | `N_GPUS=4` |
| `train_batch_size=64` (completions) | `TRAIN_BATCH_SIZE=32` prompts × `BEAM_WIDTH=16` = 512 completions/step |
| `gradient_accumulation_steps=2` | `PPO_MICRO_BATCH_PER_GPU=2` |
| `num_train_epochs=2` | `TOTAL_EPOCHS=2` |
| `num_generations=16` | `BEAM_WIDTH=16` |
| `learning_rate=1e-5` | `LEARNING_RATE=1e-5` |
| `beta=1e-3` | `kl_loss_coef=0.001` |
| `test_during_training=False` | `trainer.val_before_train=false` |

## 3. Performance optimizations (workingbranch, not on `main`)

### 3.1 Completion-only logprob (`logits_to_keep`)

- **Files**: `verl_gr/workers/engine/completion_only_logprob.py`, `minionerec_engine_patch.py`, recipe `configure_training_optimizations()`.
- **Goal**: Match MiniOneRec TRL behavior — LM head / logprob only on completion tokens, not full prompt+response.
- **Paths**:
  - **Ref** (`forward_only=True`): padded forward + `logits_to_keep` (closer to original `_get_per_token_logps`).
  - **Actor** (training): rmpad forward + `index_select` / scatter on completion positions.

### 3.2 Optimizer alignment

- **File**: `verl_gr/workers/optimizer.py`
- **Change**: `paged_adamw_32bit` via `bitsandbytes` for actor (32-bit optimizer state, bf16 compute) — aligned with MiniOneRec / DeepSpeed setup for low LR stability.

### 3.3 Trainer shortcuts

- `RLTrainer._compute_old_log_prob`: bypassed when `policy_loss.loss_mode=minionerec_reinforce` (zeros placeholder; saves one actor forward per step).
- Rmpad + chunked entropy in actor config to avoid full-vocab softmax OOM on long sequences.

### 3.4 NVTX profiling (optional)

- Same range names in MiniOneRec trainer and verl-GR (`gen.generate`, `ref.forward`, `reward.compute`, `actor.forward_backward`, `logprob.completion_only`).
- Tooling: `scripts/compare_nsys_nvtx.py`.

Observed overhead: Nsight typically adds ~10–40% per step; ~1s on a ~4s step is within expectations.

## 4. Tests (workingbranch)

| Test | Covers |
| --- | --- |
| `tests/test_minionerec_parity.py` | Prompt templates, rewards, beam constraint behavior vs MiniOneRec semantics. |
| `tests/test_minionerec_contracts.py` | Trainer/recipe wiring (AST-level contracts). |
| `tests/test_minionerec_completion_only_logprob.py` | `logits_to_keep` / nested logprob layout. |
| `tests/test_minionerec_reward_penalties.py` | Empty / invalid SID shaping penalties. |

Run locally:

```bash
cd verl-GR
PYTHONPATH="${PWD}:${PWD}/../verl" python tests/test_minionerec_parity.py
PYTHONPATH="${PWD}:${PWD}/../verl" python tests/test_minionerec_completion_only_logprob.py
PYTHONPATH="${PWD}:${PWD}/../verl" python tests/test_minionerec_reward_penalties.py
```

## 5. Scripts cleanup (this PR)

**Kept**

- `run_minionerec_grpo_rl_aligned.sh` — primary MiniOneRec training entry (renamed from `run_minionerec_grpo_align_rl_sh_4gpu.sh`).
- `run_minionerec_grpo.sh` — generic launcher.
- `compare_nsys_nvtx.py`, `convert_ddp_to_hf.py`, `eval_compare_ckpts.py`, other recipe launchers.

**Removed** (redundant / machine-specific ablation helpers)

- `run_minionerec_grpo_stabilized.sh` (thin wrapper; same as aligned script).
- `run_ablation_train_all.sh`, `run_ablation_post_eval.sh`, `analyze_ablation_training.py`.

## 6. Intentional non-goals / known gaps

- vLLM async constrained beam remains a separate path; DDP training uses **HF beam** for parity with original MiniOneRec `generate`.
- Semantic / SASRec rewards not ported (rule + ranking only).
- Full multi-epoch parity vs MiniOneRec still requires cluster runs; local tests are unit/contract level only.

## 7. `.gitignore` updates

- `outputs/`, `logs/`, `wandb/`, `*.ckpt`, `checkpoints/`, `ckpt/` (existing).
- Added: `nsys/`, `*.nsys-rep`, `*.qdrep`, `profiles/` for profiling artifacts.
