"""OpenOneRec TensorBoard profiling metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from verl import DataProto


def _as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float32)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            try:
                return np.asarray([float(v) for v in value.reshape(-1)], dtype=np.float32)
            except (TypeError, ValueError):
                return np.asarray([], dtype=np.float32)
        return value.astype(np.float32, copy=False).reshape(-1)
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return np.asarray([], dtype=np.float32)
    try:
        return np.asarray([float(value)], dtype=np.float32)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float32)


def _mean_metric(values: Any) -> float | None:
    arr = _as_float_array(values)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def compute_openonerec_training_reward_metrics(batch_like: Any) -> dict[str, float]:
    """Expose TRL-style training reward scalars from OpenOneRec reward extras."""

    metrics: dict[str, float] = {}
    non_tensor = getattr(batch_like, "non_tensor_batch", {}) or {}

    hit_reward = _mean_metric(non_tensor.get("hit_reward"))
    score = _mean_metric(non_tensor.get("score"))
    if score is None:
        score = _mean_metric(non_tensor.get("pass_at_1"))

    if hit_reward is not None:
        hit_arr = _as_float_array(non_tensor.get("hit_reward"))
        metrics.update(
            {
                "train/openonerec/reward_total": hit_reward,
                "train/reward_total": hit_reward,
                "train/openonerec/hit_any": float(np.mean(hit_arr > 0.0)) if hit_arr.size > 0 else 0.0,
            }
        )
    if score is not None:
        metrics["train/openonerec/reward"] = score
        metrics.setdefault("train/reward", score)

    partial_hit = _mean_metric(non_tensor.get("partial_hit_reward"))
    if partial_hit is not None:
        metrics["train/openonerec/partial_hit_reward"] = partial_hit

    pass_rate = _mean_metric(non_tensor.get("pass_rate"))
    if pass_rate is not None:
        metrics["train/openonerec/pass_rate"] = pass_rate

    format_reward = _mean_metric(non_tensor.get("format_reward"))
    if format_reward is not None:
        metrics["train/openonerec/format_reward"] = format_reward

    return metrics


def compute_openonerec_data_metrics(batch: "DataProto", use_critic: bool = True) -> dict[str, Any]:
    """Extend verl data metrics with OpenOneRec profiling scalars."""

    from verl.trainer.ppo.metric_utils import compute_data_metrics as _base_compute_data_metrics

    from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import calculate_rankgrpo_logprob_gate_metrics

    metrics = _base_compute_data_metrics(batch=batch, use_critic=use_critic)
    metrics.update(compute_openonerec_training_reward_metrics(batch))
    metrics.update(calculate_rankgrpo_logprob_gate_metrics(batch))
    return metrics


def _select_openonerec_mean_metric(
    metric_dict: dict[str, float],
    var_name: str,
    *,
    preferred_n: int | None = None,
) -> float | None:
    """Pick a validation mean for a reward variable across data sources.

    Prefers ``mean@{preferred_n}`` when present so TensorBoard ``eval/*`` aliases
    stay comparable across runs (e.g. always mean@32 after beam expansion fix).
    Falls back to the largest available ``mean@N``, then ``best@N/mean``.
    """

    mean_at: list[tuple[int, float]] = []
    best_at: list[tuple[int, float]] = []
    flat: list[float] = []
    for key, value in metric_dict.items():
        if not (key.startswith("val-aux/") or key.startswith("val-core/")):
            continue
        parts = key.split("/")
        if len(parts) < 4 or parts[2] != var_name:
            continue

        metric_name = parts[3]
        numeric = float(value)
        # best@N / mean  (len=5) — max-over-beams style; secondary preference
        if len(parts) == 5 and parts[4] == "mean" and metric_name.startswith("best@"):
            try:
                n_responses = int(metric_name.removeprefix("best@"))
            except ValueError:
                continue
            best_at.append((n_responses, numeric))
            continue

        # mean@N  (len=4) — per-response average; primary TB alias source
        if metric_name.startswith("mean@"):
            try:
                n_responses = int(metric_name.removeprefix("mean@"))
            except ValueError:
                continue
            mean_at.append((n_responses, numeric))
            continue

        # Flat mean: val-aux/.../var_name/mean
        if metric_name == "mean":
            flat.append(numeric)

    if preferred_n is not None:
        preferred_means = [value for n, value in mean_at if n == preferred_n]
        if preferred_means:
            return float(np.mean(preferred_means))

    if mean_at:
        return max(mean_at, key=lambda item: item[0])[1]
    if preferred_n is not None:
        preferred_best = [value for n, value in best_at if n == preferred_n]
        if preferred_best:
            return float(np.mean(preferred_best))
    if best_at:
        return max(best_at, key=lambda item: item[0])[1]
    if flat:
        return float(np.mean(flat))
    return None


def add_openonerec_eval_aliases(
    metric_dict: dict[str, float],
    *,
    preferred_n: int | None = None,
    n_prompts: int | None = None,
    n_responses: int | None = None,
) -> None:
    """Expose OpenOneRec validation metrics under RankGRPO TensorBoard names."""

    if n_prompts is not None:
        metric_dict["eval/n_prompts"] = float(n_prompts)
    if n_responses is not None:
        metric_dict["eval/n_responses"] = float(n_responses)
        if n_prompts and n_prompts > 0:
            metric_dict["eval/n_responses_per_prompt"] = float(n_responses) / float(n_prompts)

    reward = _select_openonerec_mean_metric(metric_dict, "score", preferred_n=preferred_n)
    if reward is None:
        reward = _select_openonerec_mean_metric(metric_dict, "pass_at_1", preferred_n=preferred_n)
    if reward is not None:
        metric_dict["eval/reward"] = reward

    reward_total = _select_openonerec_mean_metric(metric_dict, "hit_reward", preferred_n=preferred_n)
    if reward_total is None:
        reward_total = reward
    if reward_total is not None:
        metric_dict["eval/reward_total"] = reward_total
