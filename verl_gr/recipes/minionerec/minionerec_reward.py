"""MiniOneRec reward functions ported to verl reward API."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def normalize_sid(text: Any) -> str:
    """Match MiniOneRec's loose completion/target stripping."""

    if text is None:
        return ""
    return str(text).split("Response:\n")[-1].strip("\n\" ")


def exact_match_reward(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_sid(prediction) == normalize_sid(ground_truth) else 0.0


def rank_discounted_hit(prediction: str, ground_truth: str, extra_info: dict[str, Any]) -> float:
    """A per-sample approximation of MiniOneRec's rank-aware evaluation signal."""

    if exact_match_reward(prediction, ground_truth) == 0.0:
        return 0.0
    beam_index = int(extra_info.get("_beam_index", extra_info.get("beam_index", 0)) or 0)
    return 1.0 / math.log2(beam_index + 2)


def ndcg_penalties(group_size: int) -> list[float]:
    """Mirror MiniOneRec's rank-aware negative rewards."""

    raw = [-1.0 / math.log2(i + 2) for i in range(group_size)]
    denom = sum(raw)
    return [(-value / denom) for value in raw]


def looks_like_sid(text: str) -> bool:
    sid = normalize_sid(text)
    return bool(sid) and sid.startswith("<a_") and "<b_" in sid and "<c_" in sid


def is_valid_sid(prediction: str, valid_sid_set: set[str] | None = None) -> float:
    sid = normalize_sid(prediction)
    if not sid:
        return 0.0
    if valid_sid_set is None:
        return float(looks_like_sid(prediction))
    return float(sid in valid_sid_set)


class RewardPenaltyConfig:
    """Shaping penalties for empty / invalid rollouts (verl GRPO training).

    Plain class (not ``@dataclass``): Ray ``load_extern_object`` can import this
    module without registering it in ``sys.modules``, which breaks dataclass setup.
    """

    __slots__ = ("empty_completion", "invalid_sid")

    def __init__(
        self,
        empty_completion: float = 0.0,
        invalid_sid: float = 0.0,
    ):
        self.empty_completion = empty_completion
        self.invalid_sid = invalid_sid


def completion_shape_penalty(prediction: str, cfg: RewardPenaltyConfig | None = None) -> tuple[float, str]:
    """Return (penalty, tag) where tag is empty | invalid | valid."""

    cfg = cfg or RewardPenaltyConfig()
    sid = normalize_sid(prediction)
    if not sid:
        return cfg.empty_completion, "empty"
    if not looks_like_sid(prediction):
        return cfg.invalid_sid, "invalid"
    return 0.0, "valid"


def compute_group_training_rewards(
    completions: list[str],
    targets: list[str],
    group_keys: list[Any],
    *,
    penalty_cfg: RewardPenaltyConfig | None = None,
) -> dict[str, Any]:
    """Group-aware MiniOneRec ranking reward + empty/invalid shaping.

    Mirrors original rule + ndcg ranking, then adds:
    - empty completion penalty
    - invalid SID penalty
    """

    penalty_cfg = penalty_cfg or RewardPenaltyConfig()
    n = len(completions)
    rule_rewards = [float(normalize_sid(p) == normalize_sid(t) and normalize_sid(t) != "") for p, t in zip(completions, targets, strict=True)]
    ranking_rewards = [0.0] * n
    shape_penalties = [0.0] * n
    shape_tags = [""] * n
    group_has_hit = [0.0] * n

    for i, pred in enumerate(completions):
        pen, tag = completion_shape_penalty(pred, penalty_cfg)
        shape_penalties[i] = pen
        shape_tags[i] = tag

    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, key in enumerate(group_keys):
        groups[key].append(idx)

    for indices in groups.values():
        hit = any(rule_rewards[idx] > 0 for idx in indices)
        if not hit:
            continue
        for idx in indices:
            group_has_hit[idx] = 1.0
        discounts = ndcg_penalties(len(indices))
        for local_rank, idx in enumerate(indices):
            if rule_rewards[idx] == 0:
                ranking_rewards[idx] = discounts[local_rank]

    total_rewards = [
        rule_rewards[i] + ranking_rewards[i] + shape_penalties[i] for i in range(n)
    ]
    return {
        "total_rewards": total_rewards,
        "rule_rewards": rule_rewards,
        "ranking_rewards": ranking_rewards,
        "shape_penalties": shape_penalties,
        "shape_tags": shape_tags,
        "group_has_hit": group_has_hit,
        "invalid_sid": [float(tag != "valid") for tag in shape_tags],
        "empty_completion": [float(tag == "empty") for tag in shape_tags],
    }


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute MiniOneRec-compatible rule reward.

    MiniOneRec's original `ndcg_rule_reward` is group-aware. verl calls the
    reward function per completion, so the scalar training reward uses exact
    match while exposing a rank-discounted hit metric for validation analysis.
    """

    extra_info = extra_info or {}
    hit = exact_match_reward(solution_str, ground_truth)
    rank_hit = rank_discounted_hit(solution_str, ground_truth, extra_info)
    valid = is_valid_sid(solution_str)
    return {
        "score": hit,
        "rule_reward": hit,
        "rank_discounted_hit": rank_hit,
        "valid_sid": valid,
        "invalid_sid": 1.0 - valid,
    }
