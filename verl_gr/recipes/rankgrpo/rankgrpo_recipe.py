"""Compatibility exports for the Rank-GRPO recipe."""

from __future__ import annotations

from verl_gr.recipes.rankgrpo.rankgrpo_dataset import RankGRPODataset, collate_fn
from verl_gr.recipes.rankgrpo.rankgrpo_reward import compute_score
from verl_gr.recipes.rankgrpo.rankgrpo_task import RankGRPOTask

__all__ = ["RankGRPODataset", "RankGRPOTask", "collate_fn", "compute_score"]

