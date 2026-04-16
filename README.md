# verl-GR


## `verl_gr/` Folder Overview

- `contracts/`: shared schemas and interfaces for samples, objectives, artifacts, tokenizers, and RL/SFT/eval stages.
- `components/`: reusable runtime pieces such as tokenization helpers, evaluation metrics, and two-stage rollout logic.
- `integrations/`: adapters/bridges to external runtimes (`verl`, `vllm`) and training orchestration glue.
- `recipes/`: task-specific recipe implementations and registry (for example, OpenOneRec reward/data logic).
- `trainers/`: trainer entrypoints and wrappers (`main_ppo.py`, RL/SFT/distill trainer skeletons).

## Data preparation

Use the open-source OpenOneRec repo script `data/prepare_rl.sh` first to generate RL train/test parquet files in-situ.

## Launching Guide

1. Install base dependencies from the official script in `requirements.txt` comments, then install pinned packages in this repo.

```bash
cd verl-GR
pip install -r requirements.txt
```

2. Run the OpenOneRec GRPO launcher (set your model path first).

```bash
cd verl-GR
export OPENONEREC_ROOT=/path/to/OpenOneRec
export BASE_MODEL=/path/to/your/model
bash scripts/run_openonerec_grpo.sh
```
