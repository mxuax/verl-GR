"""Rank-GRPO advantage helpers."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any
from uuid import uuid4

import numpy as np
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


def _as_float_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _compute_rank_grpo_completion_stats(
    *,
    response_mask: torch.Tensor,
    rank_seg_ids: torch.Tensor,
    overflow_token_mask: torch.Tensor,
    eos_mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    completion_lengths = response_mask.sum(dim=1).float()
    terminated_with_eos = eos_mask.any(dim=1).float()
    eos_any = eos_mask.any(dim=1)
    eos_first = eos_mask.float().argmax(dim=1) + 1
    terminated_lengths = torch.where(
        eos_any,
        eos_first.float(),
        torch.zeros_like(completion_lengths),
    )

    valid_seg_ids = rank_seg_ids.masked_fill(~response_mask.bool(), -1)
    items_detected = valid_seg_ids.max(dim=1).values.add(1).clamp(min=0).float()
    overflow_token_counts = overflow_token_mask.sum(dim=1).float()

    return {
        "completion_lengths": completion_lengths.detach().cpu().numpy().astype(np.float32),
        "terminated_with_eos": terminated_with_eos.detach().cpu().numpy().astype(np.float32),
        "terminated_lengths": terminated_lengths.detach().cpu().numpy().astype(np.float32),
        "items_detected": items_detected.detach().cpu().numpy().astype(np.float32),
        "overflow_token_counts": overflow_token_counts.detach().cpu().numpy().astype(np.float32),
    }


def _rankgrpo_should_dump_debug_step(step: int | None) -> bool:
    steps_spec = os.environ.get("VERL_GR_RANKGRPO_DEBUG_STEPS", "").strip()
    if steps_spec:
        if step is None:
            return False
        requested_steps = set()
        for piece in steps_spec.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                requested_steps.add(int(piece))
            except ValueError:
                continue
        return int(step) in requested_steps
    return os.environ.get("VERL_GR_DEBUG", "0") == "1" or bool(os.environ.get("VERL_GR_RANKGRPO_DEBUG_DUMP_DIR"))


def _decode_masked_tokens(token_ids: torch.Tensor, mask: torch.Tensor, tokenizer, *, skip_special_tokens: bool) -> str:
    valid_ids = token_ids[mask.bool()].detach().cpu().tolist()
    return tokenizer.decode(valid_ids, skip_special_tokens=skip_special_tokens)


def _maybe_dump_rankgrpo_debug_samples(
    *,
    data: DataProto,
    tokenizer,
    response_texts: list[str],
    rank_rewards: torch.Tensor,
    stats: dict[str, np.ndarray],
    limit: int | None = None,
) -> None:
    step = data.meta_info.get("global_steps") if isinstance(data.meta_info, dict) else None
    try:
        step_int = int(step) if step is not None else None
    except (TypeError, ValueError):
        step_int = None
    if not _rankgrpo_should_dump_debug_step(step_int):
        return

    try:
        sample_limit = limit if limit is not None else int(os.environ.get("VERL_GR_RANKGRPO_DEBUG_SAMPLE_LIMIT", "8"))
    except ValueError:
        sample_limit = 8
    sample_limit = max(0, sample_limit)
    if sample_limit == 0:
        return

    dump_dir = os.environ.get("VERL_GR_RANKGRPO_DEBUG_DUMP_DIR") or os.path.join(
        os.getcwd(), "rankgrpo_debug_generations"
    )
    os.makedirs(dump_dir, exist_ok=True)
    step_label = step_int if step_int is not None else "unknown"
    path = os.path.join(dump_dir, f"rankgrpo_train_step_{step_label}_{uuid4().hex[:8]}.jsonl")

    prompts = data.batch.get("prompts")
    attention_mask = data.batch.get("attention_mask")
    responses = data.batch["responses"]
    response_mask = data.batch["response_mask"]
    reward_models = data.non_tensor_batch.get("reward_model")
    uids = data.non_tensor_batch.get("uid")
    prompt_width = prompts.size(1) if prompts is not None else 0

    with open(path, "w", encoding="utf-8") as f:
        for idx in range(min(sample_limit, responses.size(0))):
            prompt_text = None
            if prompts is not None:
                if attention_mask is not None:
                    prompt_mask = attention_mask[idx, :prompt_width]
                else:
                    prompt_mask = torch.ones_like(prompts[idx], dtype=torch.bool)
                prompt_text = _decode_masked_tokens(
                    prompts[idx], prompt_mask, tokenizer, skip_special_tokens=True
                )

            reward_model = reward_models[idx] if reward_models is not None else None
            if isinstance(reward_model, dict):
                ground_truth = reward_model.get("ground_truth")
            else:
                ground_truth = None

            row = {
                "step": step_int,
                "index": idx,
                "uid": str(uids[idx]) if uids is not None else None,
                "prompt": prompt_text,
                "completion": response_texts[idx],
                "completion_raw": _decode_masked_tokens(
                    responses[idx], response_mask[idx], tokenizer, skip_special_tokens=False
                ),
                "ground_truth": ground_truth,
                "rank_rewards": rank_rewards[idx].detach().cpu().tolist(),
                "rank_reward_sum": float(rank_rewards[idx].sum().detach().cpu().item()),
                "completion_length": float(stats["completion_lengths"][idx]),
                "terminated_with_eos": bool(stats["terminated_with_eos"][idx] > 0.0),
                "terminated_length": float(stats["terminated_lengths"][idx]),
                "items_detected": float(stats["items_detected"][idx]),
                "overflow_token_count": float(stats["overflow_token_counts"][idx]),
            }
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"[rankgrpo_debug] dumped {min(sample_limit, responses.size(0))} train samples to {path}")


def compute_rank_grpo_training_reward_metrics(batch_like: Any) -> dict[str, float]:
    """Expose TRL-comparable training reward scalars from Rank-GRPO rewards."""

    metrics: dict[str, float] = {}
    non_tensor = getattr(batch_like, "non_tensor_batch", {}) or {}

    rank_reward_sum = non_tensor.get("rank_reward_sum")
    rank_reward_mean = non_tensor.get("rank_reward_mean")
    if rank_reward_sum is not None and rank_reward_mean is not None:
        reward_total = _as_float_array(rank_reward_sum)
        reward = _as_float_array(rank_reward_mean)
        if reward_total.size > 0 and reward.size > 0:
            metrics.update(
                {
                    "train/rankgrpo/reward_total": float(np.mean(reward_total)),
                    "train/rankgrpo/reward": float(np.mean(reward)),
                    "train/rankgrpo/hit_any": float(np.mean(reward_total > 0.0)),
                }
            )

    completion_lengths = non_tensor.get("rankgrpo_completion_length")
    terminated_with_eos = non_tensor.get("rankgrpo_terminated_with_eos")
    terminated_lengths = non_tensor.get("rankgrpo_terminated_length")
    items_detected = non_tensor.get("rankgrpo_items_detected")
    overflow_token_counts = non_tensor.get("rankgrpo_overflow_token_count")
    if (
        completion_lengths is None
        or terminated_with_eos is None
        or terminated_lengths is None
        or items_detected is None
        or overflow_token_counts is None
    ):
        return metrics

    lengths = _as_float_array(completion_lengths)
    eos = _as_float_array(terminated_with_eos)
    term_lengths = _as_float_array(terminated_lengths)
    detected = _as_float_array(items_detected)
    overflow_counts = _as_float_array(overflow_token_counts)
    if lengths.size == 0:
        return metrics

    valid_term_lengths = term_lengths[eos > 0.0]
    if valid_term_lengths.size == 0:
        valid_term_lengths = np.asarray([0.0], dtype=np.float32)
    total_length = float(np.sum(lengths))
    metrics.update(
        {
            "train/rankgrpo/completions/mean_length": float(np.mean(lengths)),
            "train/rankgrpo/completions/min_length": float(np.min(lengths)),
            "train/rankgrpo/completions/max_length": float(np.max(lengths)),
            "train/rankgrpo/completions/clipped_ratio": float(1.0 - np.mean(eos)),
            "train/rankgrpo/completions/mean_terminated_length": float(np.mean(valid_term_lengths)),
            "train/rankgrpo/completions/min_terminated_length": float(np.min(valid_term_lengths)),
            "train/rankgrpo/completions/max_terminated_length": float(np.max(valid_term_lengths)),
            "train/rankgrpo/items/detected_mean": float(np.mean(detected)),
            "train/rankgrpo/items/detected_max": float(np.max(detected)),
            "train/rankgrpo/items/overflow_token_ratio": (
                float(np.sum(overflow_counts) / total_length) if total_length > 0.0 else 0.0
            ),
            "train/rankgrpo/items/eos_rate": float(np.mean(eos)),
        }
    )
    return metrics


def _store_rank_grpo_completion_stats(data: DataProto, stats: dict[str, np.ndarray]) -> None:
    data.non_tensor_batch["rankgrpo_completion_length"] = stats["completion_lengths"]
    data.non_tensor_batch["rankgrpo_terminated_with_eos"] = stats["terminated_with_eos"]
    data.non_tensor_batch["rankgrpo_terminated_length"] = stats["terminated_lengths"]
    data.non_tensor_batch["rankgrpo_items_detected"] = stats["items_detected"]
    data.non_tensor_batch["rankgrpo_overflow_token_count"] = stats["overflow_token_counts"]


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


def _compute_eos_mask(responses: torch.Tensor, response_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return torch.zeros_like(response_mask, dtype=torch.bool)
    return response_mask.bool() & responses.eq(int(eos_token_id))


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
    apply_extra_length_shaping = bool(_cfg_get(rank_cfg, "apply_extra_length_shaping", True))
    end_of_list_reward = float(_cfg_get(rank_cfg, "end_of_list_reward", 0.1))
    extra_token_penalty = float(_cfg_get(rank_cfg, "extra_token_penalty", -0.1))
    early_stop_penalty = float(_cfg_get(rank_cfg, "early_stop_penalty", -0.1))

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
    overflow_token_mask = response_mask.bool() & (seg_ids >= rec_num)

    eos_mask = _compute_eos_mask(responses, response_mask, tokenizer)
    token_advantages = token_advantages * (~eos_mask).float()

    if apply_extra_length_shaping:
        valid_seg_ids = seg_ids.masked_fill(~response_mask.bool(), -1)
        items_emitted = valid_seg_ids.max(dim=1).values.add(1).clamp(min=0)

        terminated_with_eos = eos_mask.any(dim=1)
        has_overflow = overflow_token_mask.any(dim=1)
        exact_len = (items_emitted >= rec_num) & (~has_overflow) & terminated_with_eos
        early_stop = (items_emitted < rec_num) & terminated_with_eos

        token_advantages = token_advantages + extra_token_penalty * overflow_token_mask.float()
        if exact_len.any():
            token_advantages = token_advantages + exact_len.float().unsqueeze(1) * (
                end_of_list_reward * eos_mask.float()
            )
        if early_stop.any():
            token_advantages = token_advantages + early_stop.float().unsqueeze(1) * (
                early_stop_penalty * eos_mask.float()
            )

    data.batch["advantages"] = token_advantages
    data.batch["returns"] = token_advantages
    data.batch["rank_token_mask"] = rank_token_mask
    data.batch["item_token_mask"] = rank_token_mask | overflow_token_mask
    data.batch["rank_seg_ids"] = seg_ids

    data.non_tensor_batch["rank_reward_sum"] = np.array(rank_rewards.sum(dim=1).cpu().tolist())
    data.non_tensor_batch["rank_reward_mean"] = np.array(rank_rewards.mean(dim=1).cpu().tolist())
    completion_stats = _compute_rank_grpo_completion_stats(
        response_mask=response_mask,
        rank_seg_ids=seg_ids,
        overflow_token_mask=overflow_token_mask,
        eos_mask=eos_mask,
    )
    _store_rank_grpo_completion_stats(data, completion_stats)
    _maybe_dump_rankgrpo_debug_samples(
        data=data,
        tokenizer=tokenizer,
        response_texts=response_texts,
        rank_rewards=rank_rewards,
        stats=completion_stats,
    )

    return data
