"""Shared RL dataset collate for recommendation recipes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch


def recommendation_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    non_tensors: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)
    batch: dict[str, Any] = {}
    for key, value in tensors.items():
        batch[key] = torch.stack(value, dim=0)
    for key, value in non_tensors.items():
        batch[key] = np.array(value, dtype=object)
    return batch


def extract_prompt_fields(
    row: dict[str, Any],
    *,
    prompt_key: str,
    enable_think: bool,
    enable_nonthink: bool,
) -> dict[str, Any]:
    raw_messages = row.get("messages")
    if isinstance(raw_messages, str):
        messages = ast.literal_eval(raw_messages)
    else:
        messages = raw_messages or []

    clean_chats = [
        {
            "role": message.get("role"),
            "content": "".join(
                segment.get("text", "")
                for segment in message.get("content", [])
                if segment.get("type") == "text"
            ),
        }
        for message in messages
    ]
    if not clean_chats:
        raise ValueError("Sample has empty messages; please check data integrity.")

    prompt_messages = clean_chats[:-1]
    if enable_think:
        for message in prompt_messages:
            if message["role"] == "user":
                message["content"] = message["content"] + "/think"
    if enable_nonthink:
        for message in prompt_messages:
            if message["role"] == "user":
                message["content"] = message["content"] + "/no_think"

    ground_truth_message = clean_chats[-1]["content"]
    row[prompt_key] = prompt_messages
    row["reward_model"] = {"ground_truth": ground_truth_message, "style": "rule"}
    return row




# Backward-compatible alias
collate_fn = recommendation_collate_fn
