"""Rank-GRPO reward helpers — byte-identical to the TRL reference implementation.

All parsing and matching logic is copied from the reference
``Rank-GRPO/libs/{utils,metrics_align}.py`` so that reward values
are exactly aligned without any runtime dependency on that directory.
"""

from __future__ import annotations

import pickle
import re
from functools import lru_cache
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helpers — from Rank-GRPO/libs/utils.py
# ---------------------------------------------------------------------------

def _del_parentheses(text: str) -> str:
    return re.sub(r"\([^()]*\)", "", text.strip())


def _del_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).strip()


def _del_format(text: str) -> str:
    text = text.strip()
    while True:
        new_text = re.sub(r"^([\*\_\-\#]+)", "", text).strip()
        new_text = re.sub(r"([\*\_\-\#]+)$", "", new_text).strip()
        new_text = re.sub(r"^(?:[\*\_\-\#]+\s+|(?:\d+\s*[\.\)\、\-\—\–]\s+))", "", new_text).strip()
        new_text = re.sub(r"^#+\s*", "", new_text)
        if new_text == text:
            break
        text = new_text
    return text


def _remove_quotes(s: str) -> str:
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _process_rec_raw(item: dict[str, Any], raw_rec_field: str, rec_field: str):
    """TRL reference parser — byte-identical to libs/utils.process_rec_raw."""
    rec_list_raw = item[raw_rec_field]
    rec_list_raw = re.sub(r"\n+", "\n", rec_list_raw)
    lines = [line.strip() for line in rec_list_raw.strip().split("\n") if line.strip()]
    pattern = r"(.+?)\s+\((\d{4})\)"
    try:
        rec_list = []
        for line in lines:
            line = _remove_quotes(_del_format(_del_space(line)))
            match = re.match(pattern, line)
            if match:
                movie_name = match.group(1)
                new_movie_name = _remove_quotes(_del_format(_del_space(_del_parentheses(movie_name.strip()))))
                while new_movie_name != movie_name:
                    movie_name = new_movie_name
                    new_movie_name = _remove_quotes(_del_format(_del_space(_del_parentheses(movie_name.strip()))))
                year = int(match.group(2))
                rec_list.append((movie_name, year))
        item[rec_field] = rec_list
        error = False
    except Exception:
        item[rec_field] = []
        error = True
    return error, item


# ---------------------------------------------------------------------------
# Matching — from Rank-GRPO/libs/metrics_align.py
# ---------------------------------------------------------------------------

def _safe_int_year(y: Any) -> int | None:
    try:
        return int(str(y))
    except Exception:
        return None


def _catalog_contains(gt_catalog: set | frozenset, title: str, year: int) -> bool:
    return (title, year) in gt_catalog or (title, str(year)) in gt_catalog


def _default_title_normalizer(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def _evaluate_direct_match_aligned(
    item: dict[str, Any],
    rec_num: int,
    seen_field: str,
    rec_field: str,
    gt_field: str,
    gt_catalog: set | frozenset,
    *,
    title_normalizer=None,
    year_tolerance: int = 2,
) -> np.ndarray:
    """TRL reference matcher — byte-identical to libs/metrics_align.evaluate_direct_match_aligned."""
    recs = item[rec_field]
    seen_titles = set(item[seen_field])
    groundtruths = item[gt_field]

    norm = title_normalizer or _default_title_normalizer
    rec_num = int(rec_num)
    L = min(len(recs), rec_num)
    hits = np.zeros(rec_num, dtype=np.int32)

    gt_years_by_title: dict[str, dict[int, int]] = {}
    for gt_title, gt_year in groundtruths:
        key = norm(gt_title)
        y = _safe_int_year(gt_year)
        if y is None:
            continue
        bucket = gt_years_by_title.setdefault(key, {})
        bucket[y] = bucket.get(y, 0) + 1

    for pos in range(L):
        rec_title, rec_year = recs[pos]

        if rec_title in seen_titles:
            continue

        if not _catalog_contains(gt_catalog, rec_title, rec_year):
            continue

        y = _safe_int_year(rec_year)
        if y is None:
            continue

        key = norm(rec_title)
        bucket = gt_years_by_title.get(key)
        if not bucket:
            continue

        matched_year = None
        if year_tolerance <= 0:
            if bucket.get(y, 0) > 0:
                matched_year = y
        else:
            if bucket.get(y, 0) > 0:
                matched_year = y
            else:
                for d in range(1, year_tolerance + 1):
                    yp, ym = y + d, y - d
                    if bucket.get(yp, 0) > 0:
                        matched_year = yp
                        break
                    if bucket.get(ym, 0) > 0:
                        matched_year = ym
                        break

        if matched_year is None:
            continue

        hits[pos] = 1
        cnt = bucket[matched_year] - 1
        if cnt <= 0:
            del bucket[matched_year]
            if not bucket:
                del gt_years_by_title[key]
        else:
            bucket[matched_year] = cnt

    return hits


# ---------------------------------------------------------------------------
# Public API  (compatible with verl's reward dispatch)
# ---------------------------------------------------------------------------

def _load_catalog(catalog_path: str) -> frozenset:
    with open(catalog_path, "rb") as f:
        return frozenset(
            tuple(item[:2]) if isinstance(item, (list, tuple)) else item
            for item in pickle.load(f)
        )


_GT_CATALOG_CACHE: dict[str, frozenset] = {}


def _get_catalog(catalog_path: str | None) -> frozenset | None:
    if catalog_path is None:
        return None
    if catalog_path not in _GT_CATALOG_CACHE:
        _GT_CATALOG_CACHE[catalog_path] = _load_catalog(catalog_path)
    return _GT_CATALOG_CACHE[catalog_path]


def rank_rewards_from_text(
    completion: str,
    reward_model: dict[str, Any] | None,
    *,
    rec_num: int,
    year_tolerance: int = 2,
    exclude_seen: bool = True,
    gt_catalog_path: str | None = None,
    gt_catalog: set | frozenset | None = None,
    title_normalizer=None,
) -> list[float]:
    """Return per-rank reward using TRL's exact matching logic (self-contained)."""

    reward_model = reward_model or {}
    catalog = gt_catalog or _get_catalog(gt_catalog_path)

    item: dict[str, Any] = {
        "raw_recs": completion,
        "groundtruth_with_release_year": reward_model.get(
            "groundtruth_with_release_year",
            reward_model.get("ground_truth", []),
        ),
        "seen_titles": reward_model.get("seen_titles", []),
        "rec_num": rec_num,
    }

    error, item = _process_rec_raw(item, "raw_recs", "recs")
    if error:
        return [0.0] * rec_num

    seen_field = "seen_titles" if exclude_seen else "_no_seen"
    if not exclude_seen:
        item["_no_seen"] = []

    hits = _evaluate_direct_match_aligned(
        item=item,
        rec_num=rec_num,
        seen_field=seen_field,
        rec_field="recs",
        gt_field="groundtruth_with_release_year",
        gt_catalog=catalog,
        title_normalizer=title_normalizer,
        year_tolerance=year_tolerance,
    )

    return hits.astype(np.float64).tolist()


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
