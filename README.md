# verl-GR


## `verl_gr/` Folder Overview

- `contracts/`: shared schemas and interfaces for samples, objectives, artifacts, tokenizers, and RL/SFT/eval stages.
- `components/`: reusable runtime pieces such as tokenization helpers, evaluation metrics, and two-stage rollout logic.
- `integrations/`: adapters/bridges to external runtimes (`verl`, `vllm`) and training orchestration glue.
- `recipes/`: task-specific recipe implementations and registry (for example, OpenOneRec reward/data logic).
- `trainers/`: trainer entrypoints and wrappers (`main_ppo.py`, RL/SFT/distill trainer skeletons).

## Data preparation

Data processing for reinforcement learning (RL) training. Merges multiple RL task datasets and splits them into training and test sets. Edit `prepare_rl.sh` and modify the following configuration:

```bash
REC_DATA_PATH="data/onerec_data"                  # OneRec dataset path
OUTPUT_DIR="./output/rl_data"                     # Output directory path
TEST_SIZE=1000                                     # Number of test samples per subtask
SEED=42                                            # Random seed
```

The script processes the following 5 RL task datasets:
- `sft_video_rec.parquet` - Video recommendation task
- `sft_ad_rec.parquet` - Ad recommendation task
- `sft_product_rec.parquet` - Product recommendation task
- `sft_interactive_rec.parquet` - Interactive recommendation task
- `sft_label_cond_rec.parquet` - Label-conditioned recommendation task

Then run:

```bash
cd verl_gr/recipes/openonerec
bash prepare_rl.sh
```

Output:
- `./output/rl_data/train.parquet` - Training set (remaining data after merging all tasks)
- `./output/rl_data/test.parquet` - Test set (1000 samples randomly sampled from merged data)

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
