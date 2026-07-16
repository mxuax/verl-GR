"""Rollout-side helpers for Rank-GRPO (no Ray / vLLM imports)."""

from __future__ import annotations

import os


def rankgrpo_truncate_enabled() -> bool:
    return os.environ.get("VERL_GR_TRUNCATE_AFTER_REC_NUM", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def rankgrpo_rec_num() -> int:
    return int(os.environ.get("REC_NUM", "20"))


def rankgrpo_rank_separator() -> str:
    raw = os.environ.get("RANK_SEPARATOR", "\n")
    if raw == "\\n":
        return "\n"
    return raw


def _separator_count_for_token(
    token_id: int,
    tokenizer,
    *,
    rank_separator: str,
    single_separator_id: int | None,
) -> int:
    if single_separator_id is not None and token_id == single_separator_id:
        return 1
    try:
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False)
    except TypeError:
        piece = tokenizer.decode([token_id])
    except Exception:
        piece = ""
    return str(piece).count(rank_separator)


def truncate_response_after_rec_num(
    response_ids: list[int],
    tokenizer,
    *,
    rec_num: int,
    rank_separator: str = "\n",
) -> list[int]:
    """Truncate rollout tokens once ``rec_num`` ranked items are complete.

    Mirrors ``_segment_rank_tokens`` in ``rankgrpo_algorithm``: tokens assigned
    item id ``>= rec_num`` are overflow and are dropped from the completion.
    """
    if rec_num <= 0 or not response_ids:
        return list(response_ids)

    try:
        separator_ids = tokenizer.encode(rank_separator, add_special_tokens=False)
    except Exception:
        separator_ids = []
    single_separator_id = int(separator_ids[0]) if len(separator_ids) == 1 else None

    item_id = 0
    cut_idx = len(response_ids)
    for token_idx, token_id in enumerate(response_ids):
        if item_id >= rec_num:
            cut_idx = token_idx
            break
        separator_count = _separator_count_for_token(
            int(token_id),
            tokenizer,
            rank_separator=rank_separator,
            single_separator_id=single_separator_id,
        )
        if separator_count > 0:
            item_id += separator_count
    return response_ids[:cut_idx]


def maybe_truncate_rankgrpo_response(
    response_ids: list[int],
    tokenizer,
    *,
    eos_token_id: int | list[int] | None,
    rec_num: int | None = None,
    rank_separator: str | None = None,
    truncate_enabled: bool | None = None,
) -> list[int]:
    if truncate_enabled is None:
        truncate_enabled = rankgrpo_truncate_enabled()
    if not truncate_enabled:
        return response_ids

    rec_num = rankgrpo_rec_num() if rec_num is None else rec_num
    rank_separator = rankgrpo_rank_separator() if rank_separator is None else rank_separator
    truncated = truncate_response_after_rec_num(
        response_ids,
        tokenizer,
        rec_num=rec_num,
        rank_separator=rank_separator,
    )
    if len(truncated) >= len(response_ids):
        return response_ids

    if eos_token_id is not None:
        eos_ids = {int(eos_token_id)} if isinstance(eos_token_id, int) else {int(x) for x in eos_token_id}
        if not truncated or int(truncated[-1]) not in eos_ids:
            primary_eos = int(eos_token_id) if isinstance(eos_token_id, int) else int(eos_token_id[0])
            truncated = [*truncated, primary_eos]
    return truncated
