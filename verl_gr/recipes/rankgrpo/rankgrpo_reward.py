"""Local Rank-GRPO parsing and reward helpers.

This intentionally reimplements the small behavior needed from the Rank-GRPO
reference without importing TRL or the backup reference package.
"""

from __future__ import annotations

import ast
import re
from typing import Any


_TITLE_YEAR_RE = re.compile(r"(.+?)\s+\((\d{4})\)")


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"^\d+\s*[\.\)、\-\u2014\u2013]\s*", "", text)
    text = text.strip("*_#- \t")
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return text.strip()


def _coerce_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            return [stripped]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        return [parsed]
    return [value]


def parse_title_year(value: Any) -> tuple[str, int | None]:
    if isinstance(value, (list, tuple)) and value:
        title = normalize_text(value[0])
        year = None
        if len(value) > 1:
            try:
                year = int(value[1])
            except Exception:
                year = None
        return title.lower(), year

    text = normalize_text(value)
    match = _TITLE_YEAR_RE.match(text)
    if not match:
        return text.lower(), None
    title = normalize_text(re.sub(r"\([^()]*\)", "", match.group(1)))
    return title.lower(), int(match.group(2))


def parse_recommendation_lines(text: str, rec_num: int) -> list[tuple[str, int | None]]:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    return [parse_title_year(line) for line in lines[:rec_num]]


def rank_rewards_from_text(
    completion: str,
    reward_model: dict[str, Any] | None,
    *,
    rec_num: int,
    year_tolerance: int = 2,
    exclude_seen: bool = True,
) -> list[float]:
    """Return one reward per rank position.

    The minimal reward is direct title/year matching against
    `groundtruth_with_release_year` or `ground_truth`. If the output has fewer
    than `rec_num` items, missing ranks receive zero.
    """

    reward_model = reward_model or {}
    gt_raw = reward_model.get("groundtruth_with_release_year", reward_model.get("ground_truth"))
    seen_raw = reward_model.get("seen_titles", [])
    gt_items = {parse_title_year(item) for item in _coerce_sequence(gt_raw)}
    seen_items = {parse_title_year(item) for item in _coerce_sequence(seen_raw)}

    rewards: list[float] = []
    for pred in parse_recommendation_lines(completion, rec_num):
        if exclude_seen and pred in seen_items:
            rewards.append(0.0)
            continue
        pred_title, pred_year = pred
        hit = False
        for gt_title, gt_year in gt_items:
            if pred_title != gt_title:
                continue
            if pred_year is None or gt_year is None or abs(pred_year - gt_year) <= year_tolerance:
                hit = True
                break
        rewards.append(1.0 if hit else 0.0)

    if len(rewards) < rec_num:
        rewards.extend([0.0] * (rec_num - len(rewards)))
    return rewards[:rec_num]


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra_info = extra_info or {}
    reward_model = dict(extra_info.get("reward_model") or {})
    if ground_truth is not None and "ground_truth" not in reward_model:
        reward_model["ground_truth"] = ground_truth

    rec_num = int(reward_model.get("rec_num", extra_info.get("rec_num", 20)))
    rank_rewards = rank_rewards_from_text(solution_str, reward_model, rec_num=rec_num)
    return {
        "score": float(any(rank_rewards)),
        "rank_rewards": rank_rewards,
        "rank_reward_sum": float(sum(rank_rewards)),
    }

