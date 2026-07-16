# Scripts

| Script | Env | Purpose |
| --- | --- | --- |
| `run_rankgrpo.sh` | `verl_080_fromscratch` | Qwen2.5-0.5B-Instruct checkpoint-1500, GRPO train/val |
| `run_minionerec_grpo_rl_aligned.sh` | — | **Primary entry**: 4-GPU DDP GRPO aligned with `MiniOneRec/rl.sh` |
| `run_minionerec_grpo.sh` | — | Generic Hydra launcher; set `CONFIG_NAME`, paths, and GPU count |

See `misc/profiling/compare_nsys_nvtx.py` for NVTX A/B comparison.

### Quick start

#### Rank-GRPO

Hydra config: `configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml`. Use `.match_rankgrpo.sh` for a per-GPU-set Ray cluster (ports 6380, 6382, …); call `run_rankgrpo.sh` directly when Ray is already up.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/.match_rankgrpo.sh

# Ray already running (e.g. ray start --head --port=6380):
export RAY_ADDRESS=127.0.0.1:6380
bash scripts/run_rankgrpo.sh
```

Key env overrides (all optional; defaults live in `run_rankgrpo.sh`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` / `N_GPUS` | `0,1` / `2` | GPUs (wrapper derives `N_GPUS` from visible devices) |
| `VERL_GR_ENV` | `…/envs/verl_080` | Conda env for `python` + `ray` |
| `DATA_DIR` | `<repo>/../rankgrpo_data_ckpts` | Dataset & checkpoint root |
| `BASE_MODEL` | `…/Qwen2.5-0.5B-Instruct/checkpoint-1500` | HF model path |
| `TRAIN_BATCH_SIZE` | `6` | Unique prompts per optimizer step |
| `ROLLOUT_N` / `REC_NUM` | `8` / `20` | Rollouts per prompt / recommendations |
| `SAVE_FREQ` / `TEST_FREQ` | `200` | Checkpoint & validation interval (steps) |
| `RESUME_MODE` | `auto` | `auto`, `disable`, or `resume_path` |

Smoke (few steps, no ckpt/val):

```bash
bash scripts/run_rankgrpo.sh \
  ++data.train_max_samples=64 ++data.val_max_samples=0 \
  ++trainer.total_epochs=1 \
  ++trainer.save_freq=1000000 ++trainer.test_freq=1000000
```

Resume: wrapper auto-detects latest ckpt under `OUTPUT_DIR/ckpt/`, or set explicitly:

```bash
RESUME_MODE=resume_path \
  RESUME_FROM_PATH=<output_dir>/ckpt/global_step_<N> \
  bash scripts/.match_rankgrpo.sh
```

#### MiniOneRec GRPO

```bash
cd verl-GR
export BASE_MODEL=/path/to/checkpoint
export PYTHON_BIN=/path/to/vllm-gr/bin/python
bash scripts/run_minionerec_grpo_rl_aligned.sh
```

Profiling smoke (limit prompts; `train_max_samples` must be ≥ `TRAIN_BATCH_SIZE`):

```bash
bash scripts/run_minionerec_grpo_rl_aligned.sh \
  ++trainer.total_epochs=1 \
  ++data.train_max_samples=64 \
  ++data.val_max_samples=0 \
  ++trainer.test_freq=1000000 \
  ++trainer.save_freq=1000000
```

### Checkpoint / eval utilities

See `misc/checkpoint/` and `misc/sft_eval/` — ad-hoc tooling, not production launchers.

| Script | Purpose |
| --- | --- |
| `misc/checkpoint/convert_ddp_to_hf.py` | Export DDP actor checkpoint to HuggingFace layout. |
| `misc/checkpoint/merge_fsdp_ckpt.py` | Merge FSDP shards. |
| `misc/checkpoint/eval_compare_ckpts.py` | Compare checkpoints on MiniOneRec-style metrics. |
| `misc/sft_eval/eval_sft_minionerec.sh` | MiniOneRec SFT eval. |
| `misc/sft_eval/eval_sft_onerec.sh` | OpenOneRec SFT eval. |

## Other recipes

| Script | Purpose |
| --- | --- |
| `run_openonerec_grpo.sh` | OpenOneRec two-stage GRPO. |
| `run_rankgrpo.sh` | Rank-GRPO. |
| `run_minionerec_sft.sh` / `run_onerec_sft.sh` | SFT launchers. |

## misc/

Ad-hoc and alignment-debug scripts — not long-run production entry points. See `misc/README.md`.
