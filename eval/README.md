# OpenOneRec GRPO Evaluation

`openonerec_eval.py` evaluates actor checkpoints produced by
`scripts/run_openonerec_grpo.sh`.

The flow is:

1. Resolve a verl actor checkpoint from `outputs/<experiment>/ckpt/global_step_*/actor`.
2. Merge FSDP actor shards into a HuggingFace model with `python -m verl.model_merger`.
3. Run the self-contained OpenOneRec two-stage evaluator on `test.parquet`.
4. Report SID `pass@k`, `position1_pass@k`, and `recall@k` metrics through the same
   formulas used by the OpenOneRec benchmark.

## Quick Start

From `verl-GR-fork`:

```bash
python eval/openonerec_eval.py \
  --checkpoint-root outputs \
  --test-max-sample 32 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --enforce-eager
```

By default the script searches for the newest `global_step_*/actor` checkpoint under
`outputs`, merges it under `eval/outputs/merged_models`, and evaluates the default
OpenOneRec RL test split:

```text
verl_gr/recipes/openonerec/output/rl_data/test.parquet
```

If that file does not exist in this fork, the script falls back to the sibling
`../verl-GR/verl_gr/recipes/openonerec/output/rl_data/test.parquet` path used in the
original launcher notes.

## Common Commands

Evaluate a specific checkpoint:

```bash
python eval/openonerec_eval.py \
  --actor-checkpoint outputs/my_run/ckpt/global_step_100/actor \
  --test-parquet verl_gr/recipes/openonerec/output/rl_data/test.parquet \
  --test-max-sample 100
```

Merge once, then reuse the merged model:

```bash
python eval/openonerec_eval.py \
  --actor-checkpoint outputs/my_run/ckpt/global_step_100/actor \
  --merged-model-dir eval/outputs/merged_models/my_run_step100 \
  --test-max-sample 16

python eval/openonerec_eval.py \
  --model-path eval/outputs/merged_models/my_run_step100 \
  --skip-merge \
  --test-max-sample 1000
```

Evaluate all rows in `test.parquet`:

```bash
python eval/openonerec_eval.py --test-max-sample -1
```

Run against an already running vLLM server:

```bash
python eval/openonerec_eval.py \
  --model-path /path/to/merged_hf_model \
  --skip-merge \
  --backend serving \
  --host 127.0.0.1 \
  --port 8000 \
  --test-max-sample 100
```

## Useful Knobs

- `--test-max-sample`: fast evaluation subset size; `-1` means all rows.
- `--global-step`: choose `global_step_N` under a checkpoint root.
- `--num-beams`: stage-2 beam width, default `32`.
- `--max-thinking-tokens`: stage-1 thinking budget, default `1024`.
- `--max-new-tokens`: stage-2 SID token budget, default `3`.
- `--k-values`: comma-separated SID metric cutoffs, default `1,32`.
- `--disable-thinking`: run the benchmark single-stage path instead of two-stage thinking.

Results are saved under `eval/outputs/results` unless `--result-dir` is provided.
