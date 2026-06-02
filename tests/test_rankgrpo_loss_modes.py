"""Numerical checks for RankGRPO `trl_match` vs dual-clip `verl_default` loss behavior."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

# `python tests/foo.py` puts `tests/` first on sys.path, not the repo root — ensure root
# (the directory that contains `verl_gr/`) is on sys.path.
_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "verl_gr").is_dir():
    _p = _p.parent
if (_p / "verl_gr").is_dir() and str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import torch

from verl.trainer.ppo.core_algos import agg_loss

from verl_gr.trainers.rl_trainer import _prune_unkept_checkpoint_dirs
from verl_gr.recipes.rankgrpo.rankgrpo_agent_loop import (
    _build_rankgrpo_sampling_params,
    _mask_rollout_logprobs,
    build_trl_completion_mask,
)
from verl_gr.recipes.rankgrpo.rankgrpo_loss import (
    _compute_item_mean_log_ratio,
    _resolve_old_log_prob,
    _trl_clipped_pg_loss,
)
from verl_gr.recipes.rankgrpo.rankgrpo_algorithm import (
    _compute_rank_grpo_completion_stats,
    _rankgrpo_should_dump_debug_step,
    compute_rank_grpo_training_reward_metrics,
)


def _dual_clip_pg_losses(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_c: float,
) -> torch.Tensor:
    """Mirror of `compute_policy_loss_vanilla` surrogate (per-token, before agg)."""

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    return torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)


def test_trl_clipped_pg_matches_manual_min_formulation():
    torch.manual_seed(0)
    b, t = 3, 7
    w = torch.randn(b, t, dtype=torch.float64) * 0.15
    adv = torch.randn(b, t, dtype=torch.float64) * 0.4
    eps_l, eps_h = 0.2, 0.2

    out = _trl_clipped_pg_loss(
        log_importance_weights=w,
        advantages=adv,
        loss_mask=torch.ones(b, t, dtype=torch.bool),
        clip_ratio_low=eps_l,
        clip_ratio_high=eps_h,
        kl_per_token=None,
        kl_coef=0.0,
    )
    c1 = torch.exp(w)
    c2 = torch.clamp(c1, 1 - eps_l, 1 + eps_h)
    expected = -torch.min(c1 * adv, c2 * adv)
    assert torch.allclose(out, expected, rtol=0, atol=1e-12)


def test_item_mean_log_ratio_broadcasts_within_segments():
    """Two sequences, two items each: mean log-ratio per item then gather to tokens."""

    b, t, rec_num = 2, 4, 2
    # seg 0 and 1 per row
    rank_seg_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=torch.long)
    log_prob = torch.tensor(
        [[0.0, 0.0, 1.0, -1.0], [0.5, 0.5, 0.5, 0.5]], dtype=torch.float64
    )
    old_log_prob = torch.zeros(b, t, dtype=torch.float64)
    mask = torch.ones(b, t, dtype=torch.bool)

    liw = _compute_item_mean_log_ratio(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        rank_seg_ids=rank_seg_ids,
        response_mask=mask,
        rec_num=rec_num,
    )
    # row0 item0 mean = 0, item1 mean = 0
    assert torch.allclose(liw[0, :2], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(liw[0, 2:], torch.zeros(2, dtype=torch.float64))
    # row1 item0 tokens at 0,3: log ratios 0.5,0.5 -> 0.5; item1 at 1,2: 0.5
    assert torch.allclose(liw[1], torch.full((t,), 0.5, dtype=torch.float64))


def test_current_old_log_prob_mode_matches_trl_aligned_generation():
    log_prob = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    recomputed_old = torch.tensor([[10.0, 20.0]], dtype=torch.float64)

    resolved = _resolve_old_log_prob(
        log_prob=log_prob,
        old_log_prob=recomputed_old,
        rank_grpo_config={"old_log_prob_mode": "current"},
    )

    assert torch.allclose(resolved, log_prob)
    assert resolved.requires_grad is False
    assert torch.allclose(
        _resolve_old_log_prob(
            log_prob=log_prob,
            old_log_prob=recomputed_old,
            rank_grpo_config={"old_log_prob_mode": "recomputed"},
        ),
        recomputed_old,
    )


def test_trl_match_agg_differs_from_dual_clip_when_negative_adv_and_large_ratio():
    """When adv < 0 and ratio is above the upper clip, dual-clip caps loss at -adv * clip_ratio_c."""

    b, t = 1, 4
    old_lp = torch.zeros(b, t, dtype=torch.float64)
    # Large positive log_ratio -> ratio > 1+eps so PPO clip engages; dual-clip then min(..., -adv*3).
    log_prob = torch.full((b, t), 2.0, dtype=torch.float64)
    advantages = torch.full((b, t), -1.0, dtype=torch.float64)
    loss_mask = torch.ones(b, t, dtype=torch.bool)
    eps_l, eps_h = 0.2, 0.2
    clip_c = 3.0

    trl_tok = _trl_clipped_pg_loss(
        log_importance_weights=log_prob - old_lp,
        advantages=advantages,
        loss_mask=loss_mask,
        clip_ratio_low=eps_l,
        clip_ratio_high=eps_h,
        kl_per_token=None,
        kl_coef=0.0,
    )
    dual_tok = _dual_clip_pg_losses(log_prob, old_lp, advantages, eps_l, eps_h, clip_c)

    global_info = dict(dp_size=1, batch_num_tokens=None, global_batch_size=None, loss_scale_factor=None)
    trl_scalar = agg_loss(trl_tok, loss_mask, "seq-mean-token-mean", **global_info)
    dual_scalar = agg_loss(dual_tok, loss_mask, "seq-mean-token-mean", **global_info)
    assert not torch.allclose(trl_scalar, dual_scalar, rtol=1e-2, atol=1e-6), (
        "expected dual-clip and trl_match to differ on this synthetic batch"
    )


def test_rankgrpo_training_reward_metrics_match_trl_reward_total_semantics():
    class _Batch:
        non_tensor_batch = {
            "rank_reward_sum": [0.0, 2.0, 1.0],
            "rank_reward_mean": [0.0, 0.1, 0.05],
        }

    metrics = compute_rank_grpo_training_reward_metrics(_Batch())

    assert metrics["train/rankgrpo/reward_total"] == 1.0
    assert math.isclose(metrics["train/rankgrpo/reward"], 0.05, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(metrics["train/rankgrpo/hit_any"], 2 / 3, rel_tol=1e-6, abs_tol=1e-6)


def test_rankgrpo_completion_stats_match_trl_length_semantics():
    responses = torch.tensor(
        [
            [11, 12, 99, 0, 0],
            [21, 22, 23, 24, 25],
            [31, 99, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    rank_seg_ids = torch.tensor(
        [
            [0, 0, 0, -1, -1],
            [0, 1, 2, 3, 3],
            [0, 0, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    eos_mask = response_mask & responses.eq(99)
    overflow_token_mask = response_mask & rank_seg_ids.ge(3)

    stats = _compute_rank_grpo_completion_stats(
        response_mask=response_mask,
        rank_seg_ids=rank_seg_ids,
        overflow_token_mask=overflow_token_mask,
        eos_mask=eos_mask,
    )

    assert stats["completion_lengths"].tolist() == [3.0, 5.0, 2.0]
    assert stats["terminated_with_eos"].tolist() == [1.0, 0.0, 1.0]
    assert stats["items_detected"].tolist() == [1.0, 4.0, 1.0]
    assert stats["overflow_token_counts"].tolist() == [0.0, 2.0, 0.0]
    assert stats["terminated_lengths"].tolist() == [3.0, 0.0, 2.0]

    class _Batch:
        non_tensor_batch = {
            "rankgrpo_completion_length": stats["completion_lengths"],
            "rankgrpo_terminated_with_eos": stats["terminated_with_eos"],
            "rankgrpo_terminated_length": stats["terminated_lengths"],
            "rankgrpo_items_detected": stats["items_detected"],
            "rankgrpo_overflow_token_count": stats["overflow_token_counts"],
        }

    metrics = compute_rank_grpo_training_reward_metrics(_Batch())

    assert math.isclose(metrics["train/rankgrpo/completions/mean_length"], 10 / 3, rel_tol=1e-6, abs_tol=1e-6)
    assert metrics["train/rankgrpo/completions/min_length"] == 2.0
    assert metrics["train/rankgrpo/completions/max_length"] == 5.0
    assert math.isclose(metrics["train/rankgrpo/completions/clipped_ratio"], 1 / 3, rel_tol=1e-6, abs_tol=1e-6)
    assert metrics["train/rankgrpo/completions/mean_terminated_length"] == 2.5
    assert metrics["train/rankgrpo/completions/min_terminated_length"] == 2.0
    assert metrics["train/rankgrpo/completions/max_terminated_length"] == 3.0
    assert metrics["train/rankgrpo/items/detected_mean"] == 2.0
    assert metrics["train/rankgrpo/items/detected_max"] == 4.0
    assert math.isclose(metrics["train/rankgrpo/items/overflow_token_ratio"], 2 / 10, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(metrics["train/rankgrpo/items/eos_rate"], 2 / 3, rel_tol=1e-6, abs_tol=1e-6)


def test_rankgrpo_debug_dump_step_filter():
    old_debug = os.environ.get("VERL_GR_DEBUG")
    old_steps = os.environ.get("VERL_GR_RANKGRPO_DEBUG_STEPS")
    try:
        os.environ["VERL_GR_DEBUG"] = "0"
        os.environ["VERL_GR_RANKGRPO_DEBUG_STEPS"] = "2800,5000"
        assert _rankgrpo_should_dump_debug_step(2800)
        assert _rankgrpo_should_dump_debug_step(5000)
        assert not _rankgrpo_should_dump_debug_step(3000)

        os.environ["VERL_GR_DEBUG"] = "1"
        os.environ.pop("VERL_GR_RANKGRPO_DEBUG_STEPS", None)
        assert _rankgrpo_should_dump_debug_step(None)
        assert _rankgrpo_should_dump_debug_step(123)
    finally:
        if old_debug is None:
            os.environ.pop("VERL_GR_DEBUG", None)
        else:
            os.environ["VERL_GR_DEBUG"] = old_debug
        if old_steps is None:
            os.environ.pop("VERL_GR_RANKGRPO_DEBUG_STEPS", None)
        else:
            os.environ["VERL_GR_RANKGRPO_DEBUG_STEPS"] = old_steps


def test_rankgrpo_sampling_params_match_trl_vllm_defaults():
    class _ValKwargs:
        temperature = 1.0
        top_p = 1.0
        top_k = -1

    class _Config:
        temperature = 1.0
        top_p = 1.0
        top_k = -1
        min_p = 0.0
        response_length = 1024
        calculate_log_probs = True
        val_kwargs = _ValKwargs()

    params = _build_rankgrpo_sampling_params(_Config(), validate=False)

    assert params["n"] == 1
    assert params["repetition_penalty"] == 1.0
    assert params["temperature"] == 1.0
    assert params["top_p"] == 1.0
    assert params["top_k"] == -1
    assert params["min_p"] == 0.0
    assert params["max_tokens"] == 1024
    assert params["logprobs"] is True


def _trl_completion_mask_manual(completion_ids: list[int], eos_token_id: int) -> list[int]:
    """Mirror TRL rank_grpo_trainer completion_mask (mask_truncated_completions=False)."""
    if not completion_ids:
        return []
    is_eos = [token_id == eos_token_id for token_id in completion_ids]
    if not any(is_eos):
        return [1] * len(completion_ids)
    eos_idx = next(idx for idx, flag in enumerate(is_eos) if flag)
    return [1 if idx <= eos_idx else 0 for idx in range(len(completion_ids))]


def test_build_trl_completion_mask_matches_trl_formula():
    eos = 151643
    cases = [
        [10, 20, eos, 99, 100],
        [10, 20, 30],
        [eos],
        [],
    ]
    for completion_ids in cases:
        expected = _trl_completion_mask_manual(completion_ids, eos)
        actual = build_trl_completion_mask(completion_ids, eos)
        assert actual == expected, (completion_ids, expected, actual)


def test_build_trl_completion_mask_no_eos_token_id_falls_back_to_all_ones():
    completion_ids = [1, 2, 3, 4]
    assert build_trl_completion_mask(completion_ids, None) == [1, 1, 1, 1]


def test_mask_rollout_logprobs_zeros_tokens_after_eos():
    mask = [1, 1, 1, 0, 0]
    logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5]
    masked = _mask_rollout_logprobs(logprobs, mask)
    assert masked == [-0.1, -0.2, -0.3, 0.0, 0.0]


def test_topk_pruning_removes_unkept_checkpoint_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_root = Path(tmpdir)
        for step in [100, 200, 300, 400]:
            (ckpt_root / f"global_step_{step}").mkdir()
        (ckpt_root / "not_a_checkpoint").mkdir()

        removed = _prune_unkept_checkpoint_dirs(
            str(ckpt_root),
            keep_paths={str(ckpt_root / "global_step_200"), str(ckpt_root / "global_step_400")},
        )

        assert sorted(Path(path).name for path in removed) == ["global_step_100", "global_step_300"]
        assert not (ckpt_root / "global_step_100").exists()
        assert (ckpt_root / "global_step_200").is_dir()
        assert (ckpt_root / "global_step_400").is_dir()
        assert (ckpt_root / "not_a_checkpoint").is_dir()


if __name__ == "__main__":
    test_trl_clipped_pg_matches_manual_min_formulation()
    test_item_mean_log_ratio_broadcasts_within_segments()
    test_current_old_log_prob_mode_matches_trl_aligned_generation()
    test_trl_match_agg_differs_from_dual_clip_when_negative_adv_and_large_ratio()
    test_rankgrpo_training_reward_metrics_match_trl_reward_total_semantics()
    test_rankgrpo_completion_stats_match_trl_length_semantics()
    test_rankgrpo_debug_dump_step_filter()
    test_rankgrpo_sampling_params_match_trl_vllm_defaults()
    test_build_trl_completion_mask_matches_trl_formula()
    test_build_trl_completion_mask_no_eos_token_id_falls_back_to_all_ones()
    test_mask_rollout_logprobs_zeros_tokens_after_eos()
    test_topk_pruning_removes_unkept_checkpoint_dirs()
    print("test_rankgrpo_loss_modes: all checks passed")
