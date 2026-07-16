"""Tests for Rank-GRPO rollout truncation after rec_num items."""

from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "verl_gr").is_dir():
    _p = _p.parent
if (_p / "verl_gr").is_dir() and str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from verl_gr.recipes.rankgrpo.rankgrpo_rollout_utils import truncate_response_after_rec_num


class _FakeTokenizer:
    def __init__(self):
        self._vocab = {"\n": 10, "a": 1, "b": 2, "eos": 99}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "\n":
            return [10]
        return [self._vocab[ch] for ch in text if ch in self._vocab]

    def decode(self, token_ids, **kwargs) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv.get(int(t), "?") for t in token_ids)


def test_truncate_after_rec_num_keeps_first_n_items():
    tok = _FakeTokenizer()
    # 3 items: a\n b\n c\n overflow
    response_ids = [1, 10, 2, 10, 1, 10, 2, 2, 2]
    truncated = truncate_response_after_rec_num(response_ids, tok, rec_num=3, rank_separator="\n")
    assert truncated == [1, 10, 2, 10, 1, 10]


def test_truncate_noop_when_fewer_items_than_rec_num():
    tok = _FakeTokenizer()
    response_ids = [1, 10, 2, 10]
    truncated = truncate_response_after_rec_num(response_ids, tok, rec_num=20, rank_separator="\n")
    assert truncated == response_ids
