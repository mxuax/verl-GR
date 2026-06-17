# Scripts

## Full-training benchmark (loss + wall clock)

| Script | Env | Purpose |
| --- | --- | --- |
| `benchmark/run_openonerec_benchmark.sh` | `vllm-gr` | OneRec-1.7B-pro, 2k train / 200 val subset |
| `benchmark/run_minionerec_benchmark.sh` | `MiniOneRec` | checkpoint-390, Industrial_and_Scientific |

See [docs/verl_gr/full_training_benchmark.md](../docs/verl_gr/full_training_benchmark.md).

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/benchmark/run_openonerec_benchmark.sh   # after: conda activate vllm-gr
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/benchmark/run_minionerec_benchmark.sh  # after: conda activate MiniOneRec
```

## MiniOneRec GRPO (recommended)

| Script | Purpose |
| --- | --- |
| `run_minionerec_grpo_rl_aligned.sh` | **Primary entry**: 4-GPU DDP GRPO aligned with `MiniOneRec/rl.sh` (lr, KL, beam, batch semantics). |
| `run_minionerec_grpo.sh` | Generic Hydra launcher; set `CONFIG_NAME`, paths, and GPU count. |
| `compare_nsys_nvtx.py` | Compare two `nsys stats --report nvtxsum` CSVs for NVTX range diffs. |

### Quick start (aligned training)

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

| Script | Purpose |
| --- | --- |
| `convert_ddp_to_hf.py` | Export DDP actor checkpoint to HuggingFace layout. |
| `merge_fsdp_ckpt.py` | Merge FSDP shards. |
| `eval_compare_ckpts.py` | Compare checkpoints on MiniOneRec-style metrics. |

## Other recipes

| Script | Purpose |
| --- | --- |
| `run_openonerec_grpo.sh` | OpenOneRec two-stage GRPO. |
| `run_rankgrpo.sh` | Rank-GRPO. |
| `run_minionerec_sft.sh` / `run_onerec_sft.sh` | SFT launchers. |
| `eval_sft_minionerec.sh` / `eval_sft_onerec.sh` | SFT eval. |
