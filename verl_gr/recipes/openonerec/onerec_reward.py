"""OpenOneRec reward functions."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "compute_score",
    "think_format_reward",
    "partial_hit_reward",
    "hit_reward",
    "first_sid_hit_reward",
    "pass_rate",
]

SLOT_PATTERN = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def _extract_sid_region(prediction: str) -> str:
    """Extract the region that should contain SID tuples.

    In two-stage rollout, decoded responses may not always include an opening
    `<think>` token, but they usually include `</think>` before SID tokens.
    """
    if not isinstance(prediction, str):
        return ""
    think_end = prediction.find("</think>")
    if think_end != -1:
        return prediction[think_end + len("</think>") :]
    return prediction


def _extract_all_tuples(text: Any) -> list[tuple[str, str, str]]:
    if not isinstance(text, str):
        logger.warning("_extract_all_tuples received non-string input: %s", type(text))
        return []
    matches = SLOT_PATTERN.findall(text)
    return [tuple(match) for match in matches] if matches else []


def think_format_reward(prediction: str) -> float:
    if "<think>" not in prediction or "</think>" not in prediction:
        return 0.0
    start_idx = prediction.find("<think>") + len("<think>")
    end_idx = prediction.find("</think>")
    if end_idx < start_idx:
        return 0.0
    content = prediction[start_idx:end_idx]
    content_stripped = content.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    return 1.0 if len(content_stripped) > 10 else 0.0


def partial_hit_reward(prediction: str, ground_truth: str) -> float:
    pred_tuples = _extract_all_tuples(_extract_sid_region(prediction))
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    total_reward = 0.0
    for pred_tuple in pred_tuples:
        max_score = 0.0
        for gt_tuple in gt_tuples:
            if pred_tuple == gt_tuple:
                max_score = max(max_score, 100.0)
            elif pred_tuple[:2] == gt_tuple[:2]:
                max_score = max(max_score, 10.0)
            elif pred_tuple[0] == gt_tuple[0]:
                max_score = max(max_score, 1.0)
        total_reward += max_score
    return total_reward / len(pred_tuples)


def hit_reward(prediction: str, ground_truth: str) -> float:
    pred_tuples = _extract_all_tuples(_extract_sid_region(prediction))
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    pred_set = set(pred_tuples)
    gt_set = set(gt_tuples)
    return len(pred_set & gt_set) / len(pred_tuples)


def first_sid_hit_reward(prediction: str, ground_truth: str) -> float:
    pred_tuples = _extract_all_tuples(_extract_sid_region(prediction))
    if not pred_tuples:
        return 0.0
    first_pred_tuple = pred_tuples[0]
    gt_tuples = _extract_all_tuples(ground_truth)
    if not gt_tuples:
        return 0.0
    gt_set = set(gt_tuples)
    return float(first_pred_tuple in gt_set)


def pass_rate(prediction: str, ground_truth: str) -> float:
    pred_tuples = _extract_all_tuples(_extract_sid_region(prediction))
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    pred_set = set(pred_tuples)
    gt_set = set(gt_tuples)
    return float(len(pred_set & gt_set) > 0)


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],  # noqa: ARG001
) -> dict[str, float]:
    prediction = solution_str
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

