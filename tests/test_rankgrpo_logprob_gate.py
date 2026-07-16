"""Tests for Rank-GRPO logprob gate metrics."""

from __future__ import annotations

import torch
from verl import DataProto

from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import calculate_rankgrpo_logprob_gate_metrics


def test_logprob_gate_actor_ref_diff_zero_when_equal():
    b, t = 2, 5
    lp = torch.full((b, t), -1.5)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    batch = DataProto.from_single_dict(
        {
            "old_log_probs": lp.clone(),
            "ref_log_prob": lp.clone(),
            "item_token_mask": mask,
        }
    )
    metrics = calculate_rankgrpo_logprob_gate_metrics(batch)
    assert metrics["logprob_gate/actor_minus_ref/valid"] == 1.0
    assert metrics["logprob_gate/actor_minus_ref/abs_mean"] == 0.0


def test_logprob_gate_rollout_vs_actor():
    b, t = 1, 4
    actor = torch.tensor([[-0.1, -0.2, -0.3, 0.0]])
    rollout = torch.tensor([[-0.15, -0.2, -0.25, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    batch = DataProto.from_single_dict(
        {
            "old_log_probs": actor,
            "rollout_log_probs": rollout,
            "item_token_mask": mask,
        }
    )
    metrics = calculate_rankgrpo_logprob_gate_metrics(batch)
    assert metrics["logprob_gate/actor_minus_rollout/valid"] == 1.0
    assert metrics["logprob_gate/actor_minus_rollout/abs_mean"] > 0.0
