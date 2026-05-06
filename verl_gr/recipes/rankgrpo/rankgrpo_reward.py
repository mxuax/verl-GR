"""Local Rank-GRPO parsing and reward helpers.

This intentionally reimplements the small behavior needed from the Rank-GRPO
reference without importing TRL or the backup reference package.
"""

from __future__ import annotations

import ast
import pickle
import re
from functools import lru_cache
from typing import Any


_TITLE_YEAR_RE = re.compile(r"(.+?)\s+\((\d{4})\)")


def _remove_parentheses(text: str) -> str:
    return re.sub(r"\([^()]*\)", "", text.strip())


def _remove_quotes(text: str) -> str:
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


def _format_like_reference(text: Any) -> str:
    text = re.sub(r"\s+", " ", str(text).strip()).strip()
    while True:
        new_text = re.sub(r"^([\*_#-]+)", "", text).strip()
        new_text = re.sub(r"([\*_#-]+)$", "", new_text).strip()
        new_text = re.sub(r"^(?:[\*_#-]+\s+|(?:\d+\s*[\.\)、\-\u2014\u2013]\s+))", "", new_text).strip()
        new_text = re.sub(r"^#+\s*", "", new_text)
        if new_text == text:
            return _remove_quotes(new_text)
        text = new_text


def _process_rec_raw(text: str, rec_num: int) -> list[tuple[str, int]]:
    rec_list: list[tuple[str, int]] = []
    lines = [line.strip() for line in re.sub(r"\n+", "\n", str(text or "")).strip().split("\n") if line.strip()]
    for line in lines[:rec_num]:
        line = _remove_quotes(_format_like_reference(line))
        match = _TITLE_YEAR_RE.match(line)
        if not match:
            continue
        movie_name = match.group(1)
        new_movie_name = _remove_quotes(_format_like_reference(_remove_parentheses(movie_name.strip())))
        while new_movie_name != movie_name:
            movie_name = new_movie_name
            new_movie_name = _remove_quotes(_format_like_reference(_remove_parentheses(movie_name.strip())))
        rec_list.append((movie_name, int(match.group(2))))
    return rec_list


def _normalize_catalog_item(item: Any) -> Any:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return item
    title, year = item[0], item[1]
    try:
        year = int(year)
    except Exception:
        pass
    return (title, year)


@lru_cache(maxsize=8)
def _load_catalog(catalog_path: str) -> frozenset[Any]:
    with open(catalog_path, "rb") as f:
        return frozenset(_normalize_catalog_item(item) for item in pickle.load(f))


def _safe_int_year(year: Any) -> int | None:
    try:
        return int(str(year))
    except Exception:
        return None


def _catalog_contains(gt_catalog: set[Any] | frozenset[Any], title: str, year: int) -> bool:
    return (title, year) in gt_catalog or (title, str(year)) in gt_catalog


def _default_title_normalizer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


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


def _reference_aligned_rank_rewards(
    completion: str,
    reward_model: dict[str, Any],
    *,
    rec_num: int,
    gt_catalog: set[Any] | frozenset[Any],
    year_tolerance: int,
    exclude_seen: bool,
    title_normalizer=None,
) -> list[float]:
    recs = _process_rec_raw(completion, rec_num)
    seen_titles = set(_coerce_sequence(reward_model.get("seen_titles", [])))
    groundtruths = _coerce_sequence(reward_model.get("groundtruth_with_release_year", reward_model.get("ground_truth")))

    norm = title_normalizer or _default_title_normalizer
    hits = [0.0] * rec_num
    gt_years_by_title: dict[str, dict[int, int]] = {}
    for gt_item in groundtruths:
        if not isinstance(gt_item, (list, tuple)) or len(gt_item) < 2:
            continue
        gt_title, gt_year = gt_item[0], _safe_int_year(gt_item[1])
        if gt_year is None:
            continue
        bucket = gt_years_by_title.setdefault(norm(gt_title), {})
        bucket[gt_year] = bucket.get(gt_year, 0) + 1

    for pos, (rec_title, rec_year) in enumerate(recs[:rec_num]):
        if exclude_seen and rec_title in seen_titles:
            continue
        if not _catalog_contains(gt_catalog, rec_title, rec_year):
            continue

        key = norm(rec_title)
        bucket = gt_years_by_title.get(key)
        if not bucket:
            continue

        matched_year = None
        if year_tolerance <= 0:
            if bucket.get(rec_year, 0) > 0:
                matched_year = rec_year
        elif bucket.get(rec_year, 0) > 0:
            matched_year = rec_year
        else:
            for delta in range(1, year_tolerance + 1):
                for candidate_year in (rec_year + delta, rec_year - delta):
                    if bucket.get(candidate_year, 0) > 0:
                        matched_year = candidate_year
                        break
                if matched_year is not None:
                    break

        if matched_year is None:
            continue

        hits[pos] = 1.0
        remaining = bucket[matched_year] - 1
        if remaining <= 0:
            del bucket[matched_year]
            if not bucket:
                del gt_years_by_title[key]
        else:
            bucket[matched_year] = remaining

    return hits


def rank_rewards_from_text(
    completion: str,
    reward_model: dict[str, Any] | None,
    *,
    rec_num: int,
    year_tolerance: int = 2,
    exclude_seen: bool = True,
    gt_catalog_path: str | None = None,
    gt_catalog: set[Any] | frozenset[Any] | None = None,
    title_normalizer=None,
) -> list[float]:
    """Return one reward per rank position.

    The minimal reward is direct title/year matching against
    `groundtruth_with_release_year` or `ground_truth`. If the output has fewer
    than `rec_num` items, missing ranks receive zero.
    """

    reward_model = reward_model or {}
    if gt_catalog is None and gt_catalog_path:
        gt_catalog = _load_catalog(gt_catalog_path)
    if gt_catalog is not None:
        return _reference_aligned_rank_rewards(
            completion,
            reward_model,
            rec_num=rec_num,
            gt_catalog=gt_catalog,
            year_tolerance=year_tolerance,
            exclude_seen=exclude_seen,
            title_normalizer=title_normalizer,
        )

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
    gt_catalog_path: str | None = None,
    year_tolerance: int = 2,
    exclude_seen: bool = True,
) -> dict[str, Any]:
    extra_info = extra_info or {}
    reward_model = dict(extra_info.get("reward_model") or {})
    if ground_truth is not None and "ground_truth" not in reward_model:
        reward_model["ground_truth"] = ground_truth

    rec_num = int(reward_model.get("rec_num", extra_info.get("rec_num", 20)))
    rank_rewards = rank_rewards_from_text(
        solution_str,
        reward_model,
        rec_num=rec_num,
        gt_catalog_path=gt_catalog_path,
        year_tolerance=year_tolerance,
        exclude_seen=exclude_seen,
    )
    return {
        "score": float(any(rank_rewards)),
        "rank_rewards": rank_rewards,
        "rank_reward_sum": float(sum(rank_rewards)),
    }

