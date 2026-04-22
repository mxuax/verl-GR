# verl-GR

## Source Code Overview

- `verl_gr/recipes/`: task-specific implementations and data/reward logic (for example, OpenOneRec runtime preparation and workers).
- `verl_gr/trainers/`: trainer-side wrappers around upstream `verl` trainer code.
- `verl_gr/workers/`: rollout-side extensions that are still useful outside a single recipe.
- `verl_gr/third_party/`: small compatibility helpers for non-`verl` dependencies such as `vllm`.

## Docs

- `docs/verl_gr/openonerec_mapping.md`: maps legacy OpenOneRec runtime modules to the current `verl_gr` layout.
- `docs/verl_gr/openonerec_parity_plan.md`: tracks the current Phase B parity/smoke checklist after the cleanup refactor.

## Data preparation

You will need to download `OpenOneRec/OpenOneRec-RecIF` first and then curate the RL data one-stop as follows. The flow is `OpenOneRec-RecIF -> recommendation data preprocessing -> RL data split`. Patch `verl-GR/verl_gr/recipes/openonerec/data/recif_preprocessing.sh` before getting started.

```bash
RECIF_DIR=/YOUR/RECIF/DIR
```

Then run:

```bash
cd verl-GR/verl_gr/recipes/openonerec/data
bash recif_preprocessing.sh
bash prepare_rl.sh
```

You will get the RL training data:
- `verl-GR/verl_gr/recipes/openonerec/output/rl_data/train.parquet` - Training set (remaining data after merging all tasks)
- `verl-GR/verl_gr/recipes/openonerec/output/rl_data/test.parquet` - Test set (1000 samples randomly sampled from merged data)

## Launching Guide

1. Install base dependencies from the official script in `requirements.txt` comments, then install pinned packages in this repo.

```bash
cd verl-GR
pip install -r requirements.txt
```

2. Run the OpenOneRec GRPO launcher (set your model path first).

```bash
cd verl-GR
export BASE_MODEL=/path/to/your/model
bash scripts/run_openonerec_grpo.sh
```

## Two-Stage Notes

- OpenOneRec `two_stage` is implemented entirely inside `verl-GR`.
- The async path uses `verl_gr/recipes/openonerec/two_stage_agent_loop.py` together with `verl_gr/workers/rollout/two_stage_vllm_async.py`.
- No local source patch to the upstream `verl` repo is required or expected.
