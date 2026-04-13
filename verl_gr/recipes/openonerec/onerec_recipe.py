"""Local OpenOneRec dataset/reward helpers for GRPO runtime."""

from __future__ import annotations

import re
from importlib import import_module
from typing import Any

try:
    RLHFDataset = getattr(import_module("verl.utils.dataset.rl_dataset"), "RLHFDataset")
except Exception:  # pragma: no cover - fallback for environments without verl deps
    RLHFDataset = object


class OneRecDataset(RLHFDataset):
    """Local dataset hook for OpenOneRec GRPO.

    Phase B keeps the default RLHFDataset behavior and only binds the class name
    expected by runtime overrides.
    """


_TUPLE_PATTERN = re.compile(r"<\|sid_begin\|>\s*(.*?)\s*<\|sid_end\|>", re.DOTALL)


def _extract_all_tuples(text: str) -> list[str]:
    return [match.strip() for match in _TUPLE_PATTERN.findall(text or "") if match.strip()]


def think_format_reward(prediction: str) -> float:
    return float("</think>" in (prediction or ""))


def partial_hit_reward(prediction: str, ground_truth: str) -> float:
    pred_items = set(_extract_all_tuples(prediction))
    gt_items = set(_extract_all_tuples(ground_truth))
    if not pred_items or not gt_items:
        return 0.0
    return float(len(pred_items & gt_items) / max(len(gt_items), 1))


def hit_reward(prediction: str, ground_truth: str) -> float:
    pred_items = set(_extract_all_tuples(prediction))
    gt_items = set(_extract_all_tuples(ground_truth))
    return float(bool(pred_items and gt_items and pred_items == gt_items))


def first_sid_hit_reward(prediction: str, ground_truth: str) -> float:
    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    return float(pred_tuples[0] in set(gt_tuples))


def pass_rate(prediction: str, ground_truth: str) -> float:
    pred_set = set(_extract_all_tuples(prediction))
    gt_set = set(_extract_all_tuples(ground_truth))
    if not pred_set or not gt_set:
        return 0.0
    return float(len(pred_set & gt_set) > 0)


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],  # noqa: ARG001
) -> dict[str, float]:
    """Compute reward bundle aligned to OpenOneRec GRPO usage."""

    prediction = solution_str or ""
    format_reward_value = think_format_reward(prediction)
    partial_hit_reward_value = partial_hit_reward(prediction, ground_truth)
    hit_reward_value = hit_reward(prediction, ground_truth)
    pass_rate_value = pass_rate(prediction, ground_truth)
    pass_at_1_value = first_sid_hit_reward(prediction, ground_truth)
    return {
        "score": pass_at_1_value,
        "format_reward": format_reward_value,
        "partial_hit_reward": partial_hit_reward_value,
        "hit_reward": hit_reward_value,
        "pass_rate": pass_rate_value,
        "pass_at_1": pass_at_1_value,
    }

