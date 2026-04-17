# verl-GR


## `verl_gr/` Folder Overview

- `contracts/`: shared schemas and interfaces for samples, objectives, artifacts, tokenizers, and RL/SFT/eval stages.
- `components/`: reusable runtime pieces such as tokenization helpers, evaluation metrics, and two-stage rollout logic.
- `integrations/`: adapters/bridges to external runtimes (`verl`, `vllm`) and training orchestration glue.
- `recipes/`: task-specific recipe implementations and registry (for example, OpenOneRec reward/data logic).
- `trainers/`: trainer entrypoints and wrappers (`main_ppo.py`, RL/SFT/distill trainer skeletons).

## Data preparation

You will need to download `OpenOneRec/OpenOneRec-RecIF` first and then curate the RL data one-stop as follows.

```
OpenOneRec-RecIF -> Recommendataion data preprocessing -> RL data split
```

Patch `verl_gr/recipes/openonerec/data/recif_preprocessing.sh` as below:

```bash
RECIF_DIR=YOUR/RECIF/DIR
```

Then run:

```bash
cd verl_gr/recipes/openonerec
bash recif_preprocessing.sh
bash prepare_rl.sh
```

Output:
- `verl_gr/recipes/openonerec/output/rl_data/train.parquet` - Training set (remaining data after merging all tasks)
- `verl_gr/recipes/openonerec/output/rl_data/test.parquet` - Test set (1000 samples randomly sampled from merged data)

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
