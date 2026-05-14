"""MiniOneRec prompt and SID formatting helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def parse_maybe_list(value: Any) -> list[Any]:
    """Parse MiniOneRec CSV list fields — exact mirror of the original ``eval()``
    pattern used in ``data.py`` (e.g. ``eval(row['history_item_sid'])``)."""

    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        try:
            parsed = eval(value)
        except Exception:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def build_sid_prompt(history_item_sid: list[Any]) -> tuple[str, str]:
    """Mirror MiniOneRec.data.SidDataset prompt and history formatting."""

    history = ", ".join(str(item) for item in history_item_sid)
    history_key = "::".join(str(item) for item in history_item_sid)
    prompt = (
        "### User Input: \n"
        f"The user has interacted with items {history} in chronological order. "
        "Can you predict the next possible item that the user may expect?\n\n"
        "### Response:\n"
    )
    return prompt, history_key


def build_title2sid_prompt(task: str, text: str) -> tuple[str, str]:
    """Mirror MiniOneRec.data.RLTitle2SidDataset prompt formatting."""

    if task == "title2sid":
        prompt = f"Which item has the title: {text}?"
    else:
        prompt = f"An item can be described as follows: \"{text}\". Which item is it describing?"
    formatted = f"### User Input: \n{prompt}\n\n### Response:\n"
    return formatted, text


def build_seq_title2sid_prompt(history_item_title: list[Any]) -> tuple[str, str]:
    """Mirror MiniOneRec.data.RLSeqTitle2SidDataset prompt formatting."""

    inter_titles = ", ".join([f'"{title}"' for title in history_item_title])
    history_key = "::".join(str(title) for title in history_item_title)
    prompt = (
        f"Given the title sequence of user historical interactive items: {inter_titles}, "
        "can you recommend a suitable next item for the user?"
    )
    formatted = f"### User Input: \n{prompt}\n\n### Response:\n"
    return formatted, history_key


def maybe_parse_description(description: Any) -> Any:
    """Exact mirror of RLTitle2SidDataset description parsing (data.py:822-827).

    The original uses ``eval()`` with bare ``except: pass`` on list-format
    description strings.  All other values — including ``None`` — pass
    through unchanged.
    """

    if isinstance(description, str) and description.startswith("['") and description.endswith("']"):
        try:
            desc_list = eval(description)
            description = desc_list[0] if desc_list else description
        except Exception:
            pass
    return description


def sample_records(records: list[dict[str, Any]], sample: int, *, seed: int | None = None) -> list[dict[str, Any]]:
    if sample <= 0 or sample >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, sample)
