# Full Training Benchmark Plan (Loss + Wall Clock)

**Goal:** On a **small, fixed data subset**, compare verl-GR vs original repo on **loss/metric trends** and **end-to-end wall clock**. Skip OpenOneRec distill / RL-pretrain pipeline — start from the official **post-trained** checkpoint and run GRPO only.

## Scope

| Item | In scope | Out of scope |
|---|---|---|
| OpenOneRec GRPO from **OneRec-1.7B-pro** | Yes | Full data pipeline rebuild, distill, RL-pretrain |
| MiniOneRec GRPO from **checkpoint-390** | Yes | Full Amazon catalog sweep |
| Beam backend merge | No | Open/Mini keep separate backends by design |
| Rank-GRPO | No | Separate track |

## Environment & hardware

| Recipe | Conda env | GPUs (hk01dgx036) |
|---|---|---|
| OpenOneRec (verl-GR + original) | `vllm-gr` | `CUDA_VISIBLE_DEVICES=4,5,6,7` |
| MiniOneRec (verl-GR + original) | `MiniOneRec` | `CUDA_VISIBLE_DEVICES=4,5,6,7` |

Record `hostname`, `nvidia-smi`, and git commit for each run.

---

## Fixed assets

### OpenOneRec

| Key | Path |
|---|---|
| **BASE_MODEL** (pro, post-trained) | `/scratch/fq9hpsac/huggingface/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/5dc1b097ab8194f48f14730e5400a276a22f4ca1` |
| **Train parquet** | `verl-GR/verl_gr/recipes/openonerec/output/rl_data/train.parquet` |
| **Val parquet** | `verl-GR/verl_gr/recipes/openonerec/output/rl_data/test.parquet` |
| **Subset (time control)** | `data.train_max_samples=2000`, `data.val_max_samples=200` |

Do **not** use the old pretrain blob path (`models--OpenOneRec--OneRec-1.7B-pretrain/blobs/...`).

### MiniOneRec

| Key | Path |
|---|---|
| **BASE_MODEL** | `/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390` |
| **TRAIN_FILE** | `.../MiniOneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv` |
| **VAL_FILE** | `.../MiniOneRec/data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv` |
| **INFO_FILE** | `.../MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt` |
| **SID_INDEX_FILE** | `.../MiniOneRec/data/Amazon/index/Industrial_and_Scientific.index.json` |
| **ITEM_META_FILE** | `.../MiniOneRec/data/Amazon/index/Industrial_and_Scientific.item.json` |
| **Category** | `Industrial_and_Scientific` |

Optional subset for faster MiniOneRec runs: `data.train_max_samples=4096` (add via Hydra override).

---

## Post-refactor command notes

Changes since your last runs:

1. **OpenOneRec** default config already sets `rollout.name: two_stage` — `++actor_rollout_ref.rollout.name=two_stage` is optional.
2. **`custom_cls.path`** still points at `onerec_recipe.py` (re-export shim); behavior unchanged.
3. **`task.trainer_adapter_class`** is set in yaml; adapter loads `OpenOneRecTrainerAdapter` (includes checkpoint prune).
4. Remove duplicate / broken shell continuations (e.g. two conflicting `two_stage_max_inflight_requests` lines).
5. **MiniOneRec** `run_minionerec_grpo.sh` passes `"$@"` last — Hydra overrides at the end win.

---

## Phase A — verl-GR runs (primary)

Use the wrapper scripts (log wall clock + tensorboard path):

```bash
# OpenOneRec
conda activate vllm-gr
cd /home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/benchmark/run_openonerec_benchmark.sh

# MiniOneRec
conda activate MiniOneRec
cd /home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/benchmark/run_minionerec_benchmark.sh
```

Artifacts per run:

- `outputs/<experiment>/benchmark_wallclock.txt` — start/end/duration
- `outputs/<experiment>/tensorboard/` — scalars
- `outputs/<experiment>/ckpt/` — checkpoints (if save_freq hit)

---

## Phase B — original repo baselines (same env, same subset)

Run **after** Phase A configs are frozen. Align hparams using [openonerec_hparams_and_metrics.md](./openonerec_hparams_and_metrics.md) and [minionerec_mapping.md](./minionerec_mapping.md).

| Side | Entry | Env |
|---|---|---|
| OpenOneRec | `OpenOneRec/recipe/onerec/run_grpo.sh` | `vllm-gr` |
| MiniOneRec | `MiniOneRec/rl.sh` | `MiniOneRec` |

Pass criteria (initial):

- **Loss:** `actor/pg_loss`, `actor/kl_loss` same trend / order of magnitude over first epoch
- **OpenOneRec val:** `val-aux/*/pass_at_32/mean` or `pass_at_1` comparable direction
- **MiniOneRec val:** `hr@20`, `ndcg@20` from trainer validation
- **Wall clock:** report sec/step and total time for the **same** `train_max_samples` / epochs / batch / beam

For wall-clock-only comparison, set `trainer.val_before_train=false` on both sides.

---

## Metrics checklist

### OpenOneRec (every `test_freq` steps)

- `actor/pg_loss`, `actor/kl_loss`
- `val-aux/*/pass_at_32/mean`, `val-core/*/reward/mean` (see hparams doc)
- Step time: log `training/global_step` vs wall clock from `benchmark_wallclock.txt`

### MiniOneRec

- `actor/pg_loss`, `actor/kl_loss`
- `val-aux/*/hr@20/mean`, `val-aux/*/ndcg@20/mean`
- Effective batch: 32 prompts × 16 beam = 512 completions/step (when `TRAIN_BATCH_SIZE=32`)

---

## Recommended benchmark hyperparameters (time-boxed)

### OpenOneRec (4 GPU, ~sub-day target)

| Param | Value |
|---|---|
| `N_GPUS` | 4 |
| `TRAIN_BATCH_SIZE` | 4 |
| `train_max_samples` | 2000 |
| `val_max_samples` | 200 |
| `val_batch_size` | 32 |
| `total_epochs` | 1 |
| `beam_width` | 32 |
| `test_freq` / `save_freq` | 50 / 100 |
| `ppo_max_token_len_per_gpu` | 12288 |
| `two_stage_max_inflight_requests` | 16 |
| `val_before_train` | true (metric baseline); false for pure wall-clock |

### MiniOneRec (4 GPU)

| Param | Value |
|---|---|
| `N_GPUS` | 4 |
| `TRAIN_BATCH_SIZE` | 32 |
| `BEAM_WIDTH` | 16 |
| `TOTAL_EPOCHS` | 2 |
| `PPO_MICRO_BATCH_PER_GPU` | 2 |
| `LEARNING_RATE` | 1e-5 |
| `kl_loss_coef` | 0.001 |
| `ITEM_MAX_TOKENS` | 128 |
| `MAX_RESPONSE_LENGTH` | 128 |
| `test_freq` / `save_freq` | 50 |
| `val_before_train` | false |

---

## Analysis template

| Run ID | Recipe | Side | Env | Samples | Epochs | Total sec | Sec/step | pg_loss@step50 | kl_loss@step50 | Val metric@last |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| O1 | OpenOneRec | verl-GR | vllm-gr | 2000 | 1 | | | | | |
| O2 | OpenOneRec | original | vllm-gr | 2000 | 1 | | | | | |
| M1 | MiniOneRec | verl-GR | MiniOneRec | full/subset | 2 | | | | | |
| M2 | MiniOneRec | original | MiniOneRec | full/subset | 2 | | | | | |

---

## Execution order

1. [ ] Run `scripts/benchmark/run_openonerec_benchmark.sh` on hk01dgx036
2. [ ] Run `scripts/benchmark/run_minionerec_benchmark.sh` on hk01dgx036
3. [ ] Export tensorboard scalars (or wandb offline logs) for loss curves
4. [ ] Run original baselines with matched hparams
5. [ ] Fill analysis table; file issues if loss diverges or wall clock regresses >X%

See [architecture.md](./architecture.md) for decode-backend separation policy.
