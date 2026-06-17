"""Backward-compatible re-exports for OpenOneRec recipe modules."""

from verl_gr.recipes.common.collate import collate_fn
from verl_gr.recipes.openonerec.onerec_dataset import OneRecDataset, extract_prompt_fields
from verl_gr.recipes.openonerec.onerec_reward import compute_score
from verl_gr.recipes.openonerec.onerec_task import OneRecTask

__all__ = ["collate_fn", "OneRecDataset", "compute_score", "OneRecTask", "extract_prompt_fields"]
