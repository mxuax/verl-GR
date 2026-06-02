"""Rank-GRPO advantage helpers."""

from __future__ import annotations

from collections import defaultdict
import os
from typing import Any

import torch
from verl import DataProto

from verl_gr.recipes.rankgrpo.rankgrpo_reward import rank_rewards_from_text

__all__ = [
    "compute_rank_grpo_advantage",
    "compute_rank_grpo_training_reward_metrics",
    "rankgrpo_enabled",
    "_compute_rank_grpo_completion_stats",
    "_rankgrpo_should_dump_debug_step",
]


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def rankgrpo_enabled(config: Any) -> bool:
    rank_cfg = _cfg_get(config, "rank_grpo", None)
    return bool(_cfg_get(rank_cfg, "enable", False))


def _decode_response_texts(responses: torch.Tensor, response_mask: torch.Tensor, tokenizer) -> list[str]:
    texts: list[str] = []
    for ids, mask in zip(responses, response_mask, strict=True):
        valid_ids = ids[mask.bool()].detach().cpu().tolist()
        texts.append(tokenizer.decode(valid_ids, skip_special_tokens=True))
    return texts


def _segment_rank_tokens(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    *,
    rank_separator: str,
    rec_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign natural rank ids to response tokens using newline-like separators."""

    device = responses.device
    batch_size, response_length = responses.size()
    seg_ids = torch.full((batch_size, response_length), -1, dtype=torch.long, device=device)
    try:
        separator_ids = tokenizer.encode(rank_separator, add_special_tokens=False)
    except Exception:
        separator_ids = []
    single_separator_id = int(separator_ids[0]) if len(separator_ids) == 1 else None

    for row_idx in range(batch_size):
        valid = int(response_mask[row_idx].sum().item())
        item_id = 0
        for token_idx in range(valid):
            seg_ids[row_idx, token_idx] = item_id
            token_id = int(responses[row_idx, token_idx].item())
            separator_count = 0
            if single_separator_id is not None and token_id == single_separator_id:
                separator_count = 1
            else:
                try:
                    piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False)
                except TypeError:
                    piece = tokenizer.decode([token_id])
                except Exception:
                    piece = ""
                separator_count = str(piece).count(rank_separator)
            if separator_count > 0:
                item_id += separator_count

    rank_token_mask = response_mask.bool() & (seg_ids >= 0) & (seg_ids < rec_num)
    return seg_ids, rank_token_mask


def _compute_rank_grpo_completion_stats(
    *,
    response_mask: torch.Tensor,
    rank_seg_ids: torch.Tensor,
    overflow_token_mask: torch.Tensor,
    eos_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    completion_lengths = response_mask.sum(dim=1).float()
    terminated_with_eos = eos_mask.any(dim=1).float()
    eos_any = eos_mask.any(dim=1)
    eos_first = eos_mask.float().argmax(dim=1) + 1
    terminated_lengths = torch.where(
        eos_any,
        eos_first.float(),
        torch.zeros_like(completion_lengths),
    )
    items_detected = (rank_seg_ids.where(response_mask.bool(), torch.full_like(rank_seg_ids, -1)).amax(dim=1) + 1).clamp(min=0).float()
    overflow_token_counts = overflow_token_mask.sum(dim=1).float()
    return {
        "completion_lengths": completion_lengths,
        "terminated_with_eos": terminated_with_eos,
        "terminated_lengths": terminated_lengths,
        "items_detected": items_detected,
        "overflow_token_counts": overflow_token_counts,
    }


def _rankgrpo_should_dump_debug_step(step: int | None) -> bool:
    explicit = os.environ.get("VERL_GR_RANKGRPO_DEBUG_STEPS", "").strip()
    if explicit:
        wanted = {s.strip() for s in explicit.split(",") if s.strip()}
        if step is None:
            return False
        return str(step) in wanted
    debug = os.environ.get("VERL_GR_DEBUG", "0").strip().lower()
    return debug in {"1", "true", "yes", "on"}


def _mean_tensor(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def compute_rank_grpo_training_reward_metrics(batch_like: Any) -> dict[str, float]:
    non_tensor = getattr(batch_like, "non_tensor_batch", {}) or {}
    out: dict[str, float] = {}

    rank_reward_sum = non_tensor.get("rank_reward_sum")
    if rank_reward_sum is not None and len(rank_reward_sum) > 0:
        tensor = torch.as_tensor(rank_reward_sum, dtype=torch.float32)
        out["train/rankgrpo/reward_total"] = _mean_tensor(tensor)
        out["train/rankgrpo/hit_any"] = float((tensor > 0).float().mean().item())

    rank_reward_mean = non_tensor.get("rank_reward_mean")
    if rank_reward_mean is not None and len(rank_reward_mean) > 0:
        out["train/rankgrpo/reward"] = _mean_tensor(torch.as_tensor(rank_reward_mean, dtype=torch.float32))

    completion_lengths = non_tensor.get("rankgrpo_completion_length")
    if completion_lengths is not None:
        completion_lengths = torch.as_tensor(completion_lengths, dtype=torch.float32)
        if completion_lengths.numel() > 0:
            out["train/rankgrpo/completions/mean_length"] = float(completion_lengths.mean().item())
            out["train/rankgrpo/completions/min_length"] = float(completion_lengths.min().item())
            out["train/rankgrpo/completions/max_length"] = float(completion_lengths.max().item())

    terminated_with_eos = non_tensor.get("rankgrpo_terminated_with_eos")
    if terminated_with_eos is not None:
        terminated_with_eos = torch.as_tensor(terminated_with_eos, dtype=torch.float32)
        if terminated_with_eos.numel() > 0:
            out["train/rankgrpo/items/eos_rate"] = float(terminated_with_eos.mean().item())

    terminated_length = non_tensor.get("rankgrpo_terminated_length")
    if terminated_length is not None:
        terminated_length = torch.as_tensor(terminated_length, dtype=torch.float32)
        valid = terminated_length[terminated_length > 0]
        if valid.numel() > 0:
            out["train/rankgrpo/completions/mean_terminated_length"] = float(valid.mean().item())
            out["train/rankgrpo/completions/min_terminated_length"] = float(valid.min().item())
            out["train/rankgrpo/completions/max_terminated_length"] = float(valid.max().item())

    items_detected = non_tensor.get("rankgrpo_items_detected")
    if items_detected is not None:
        items_detected = torch.as_tensor(items_detected, dtype=torch.float32)
        if items_detected.numel() > 0:
            out["train/rankgrpo/items/detected_mean"] = float(items_detected.mean().item())
            out["train/rankgrpo/items/detected_max"] = float(items_detected.max().item())

    overflow_counts = non_tensor.get("rankgrpo_overflow_token_count")
    if overflow_counts is not None and completion_lengths is not None:
        overflow_counts = torch.as_tensor(overflow_counts, dtype=torch.float32)
        denom = float(completion_lengths.sum().item())
        if denom > 0:
            out["train/rankgrpo/completions/clipped_ratio"] = float((overflow_counts > 0).float().mean().item())
            out["train/rankgrpo/items/overflow_token_ratio"] = float(overflow_counts.sum().item() / denom)

    return out


def compute_rank_grpo_advantage(
    data: DataProto,
    *,
    config,
    tokenizer,
    norm_adv_by_std_in_grpo: bool,
) -> DataProto:
    if tokenizer is None:
        raise ValueError("Rank-GRPO advantage computation requires the trainer tokenizer.")

    rank_cfg = _cfg_get(config, "rank_grpo", {}) or {}
    rec_num = int(_cfg_get(rank_cfg, "rec_num", 20))
    rank_separator = _cfg_get(rank_cfg, "rank_separator", "\n")
    year_tolerance = int(_cfg_get(rank_cfg, "year_tolerance", 2))
    exclude_seen = bool(_cfg_get(rank_cfg, "exclude_seen", True))
    normalize_by_std = bool(_cfg_get(rank_cfg, "normalize_by_std", norm_adv_by_std_in_grpo))
    gt_catalog_path = _cfg_get(rank_cfg, "gt_catalog_path", None)

    responses = data.batch["responses"]
    response_mask = data.batch["response_mask"]
    response_texts = _decode_response_texts(responses, response_mask, tokenizer)
    reward_models = data.non_tensor_batch.get("reward_model")
    if reward_models is None:
        raise KeyError("Rank-GRPO requires `reward_model` in data.non_tensor_batch.")

    reward_rows = [
        rank_rewards_from_text(
            text,
            reward_model,
            rec_num=rec_num,
            year_tolerance=year_tolerance,
            exclude_seen=exclude_seen,
            gt_catalog_path=gt_catalog_path,
        )
        for text, reward_model in zip(response_texts, reward_models, strict=True)
    ]
    rank_rewards = torch.tensor(reward_rows, dtype=torch.float32, device=responses.device)

    uids = data.non_tensor_batch.get("uid")
    if uids is None:
        uids = list(range(rank_rewards.size(0)))
    uid_to_indices: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[uid].append(idx)

    rank_advantages = torch.zeros_like(rank_rewards)
    for indices in uid_to_indices.values():
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=responses.device)
        group_rewards = rank_rewards.index_select(0, idx_tensor)
        centered = group_rewards - group_rewards.mean(dim=0, keepdim=True)
        if normalize_by_std:
            std = group_rewards.std(dim=0, unbiased=False, keepdim=True)
            centered = centered / (std + 1e-4)
        rank_advantages.index_copy_(0, idx_tensor, centered)

    seg_ids, rank_token_mask = _segment_rank_tokens(
        responses,
        response_mask,
        tokenizer,
        rank_separator=rank_separator,
        rec_num=rec_num,
    )
    clamped_seg_ids = seg_ids.clamp(min=0, max=rec_num - 1)
    token_advantages = rank_advantages.gather(1, clamped_seg_ids)
    token_advantages = token_advantages * rank_token_mask.float()

    data.batch["advantages"] = token_advantages
    data.batch["returns"] = token_advantages
    data.batch["rank_token_mask"] = rank_token_mask
    data.batch["rank_seg_ids"] = seg_ids
    return data
