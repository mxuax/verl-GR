"""Tokenizer helpers for natural-language Rank-GRPO outputs.

Unlike OpenOneRec, Rank-GRPO recommendations are newline-separated text
items. This module intentionally avoids SID tokens, force prefixes, and
two-stage item-generation tokenizer assumptions.
"""

from __future__ import annotations

from typing import Any

from verl.utils import hf_processor, hf_tokenizer


def build_rankgrpo_tokenizer_and_processor(
    model_path: str,
    *,
    trust_remote_code: bool,
    use_processor: bool = False,
    rank_separator: str = "\n",
    force_pad_to_eos: bool = True,
) -> dict[str, Any]:
    """Build tokenizer/processor objects expected by the verl training stack."""

    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    if force_pad_to_eos and tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = None
    if use_processor:
        processor = hf_processor(model_path, trust_remote_code=trust_remote_code, use_fast=True)

    try:
        separator_token_ids = tokenizer.encode(rank_separator, add_special_tokens=False)
    except Exception:
        separator_token_ids = []

    return {
        "tokenizer": tokenizer,
        "processor": processor,
        "rank_separator_token_ids": separator_token_ids,
    }

