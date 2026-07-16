"""Rank-GRPO actor loss helpers."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _compute_item_mean_log_ratio(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    rank_seg_ids: torch.Tensor,
    response_mask: torch.Tensor,
    rec_num: int,
) -> torch.Tensor:
    """Compute per-item geometric-mean log-ratio and broadcast to tokens.

    Returns item_mean_log_ratio broadcast to shape (B, T), matching TRL's
    ``log_importance_weights`` for item-level importance sampling.
    """

    log_ratio = log_prob - old_log_prob
    seg_ids = rank_seg_ids.clamp(min=0, max=rec_num).long()
    mask = response_mask.to(dtype=log_ratio.dtype)

    batch_size, _ = seg_ids.shape
    n_bins = rec_num + 1  # final bin is overflow
    sums = torch.zeros((batch_size, n_bins), dtype=log_ratio.dtype, device=log_ratio.device)
    counts = torch.zeros_like(sums)

    sums.scatter_add_(1, seg_ids, log_ratio * mask)
    counts.scatter_add_(1, seg_ids, mask)
    return (sums / counts.clamp(min=1.0)).gather(1, seg_ids)


def _item_level_log_prob(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    rank_seg_ids: torch.Tensor,
    response_mask: torch.Tensor,
    rec_num: int,
) -> torch.Tensor:
    """Replace token log-ratios with per-item geometric-mean log-ratios.

    Used by the ``verl_default`` loss path only.  The TRL-matched path uses
    ``_compute_item_mean_log_ratio`` directly to produce importance weights.
    """

    item_log_ratio = _compute_item_mean_log_ratio(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        rank_seg_ids=rank_seg_ids,
        response_mask=response_mask,
        rec_num=rec_num,
    )
    return old_log_prob + item_log_ratio


def _resolve_old_log_prob(*, log_prob: torch.Tensor, old_log_prob: torch.Tensor, rank_grpo_config) -> torch.Tensor:
    mode = str(_cfg_get(rank_grpo_config, "old_log_prob_mode", "recomputed")).lower()
    if mode in {"current", "trl", "trl_match"}:
        # TRL aligned path: when generation and update are aligned, TRL does not
        # carry a separate old_per_token_logps tensor and anchors PPO to the
        # current forward pass detached from gradient.
        return log_prob.detach()
    if mode in {"recomputed", "old", "verl"}:
        return old_log_prob
    raise ValueError(f"Unknown Rank-GRPO old_log_prob_mode: {mode}")


def _trl_clipped_pg_loss(
    log_importance_weights: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    kl_per_token: torch.Tensor | None = None,
    kl_coef: float = 0.0,
) -> torch.Tensor:
    """TRL-matched clipped policy gradient loss — per-token, *unaggregated*.

    Matches ``_compute_loss`` in rank_grpo_trainer.py (L2103-L2118):
      - Coef from ``exp(log_importance_weights)`` — no dual-clip.
      - Two-sided clamp: [1-eps_low, 1+eps_high].
      - Per-token loss = -min(coef_1*adv, coef_2*adv).
      - KL optionally added *per-token* before aggregation.
    """

    coef_1 = torch.exp(log_importance_weights)
    coef_2 = torch.clamp(coef_1, 1 - clip_ratio_low, 1 + clip_ratio_high)

    pg1 = coef_1 * advantages
    pg2 = coef_2 * advantages
    per_token_loss = -torch.min(pg1, pg2)

    if kl_per_token is not None and kl_coef != 0.0:
        per_token_loss = per_token_loss + kl_coef * kl_per_token

    return per_token_loss


def _build_debug_metrics(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    pg_loss: torch.Tensor,
    policy_loss: torch.Tensor,
    kld: torch.Tensor | None,
    ref_log_prob: torch.Tensor | None,
    use_kl: bool,
):
    """Collect per-step debug metrics.  Mirrors lines 160-204 of original."""

    metrics: dict[str, Any] = {}
    debug_mask = loss_mask.float()
    mask_sum = debug_mask.sum().clamp(min=1)

    # advantages
    adv_m = (advantages * debug_mask).sum() / mask_sum
    adv_s = ((advantages - adv_m).square() * debug_mask).sum() / mask_sum
    metrics["dbg/adv_mean"] = Metric("mean", value=adv_m.detach())
    metrics["dbg/adv_std"] = Metric("mean", value=adv_s.sqrt().detach())
    metrics["dbg/adv_min"] = Metric("min", value=advantages[loss_mask].min().detach())
    metrics["dbg/adv_max"] = Metric("max", value=advantages[loss_mask].max().detach())

    # per-token clipped-PG loss (before aggregation)
    pg_tok_mean = (pg_loss * debug_mask).sum().detach() / mask_sum
    metrics["dbg/pg_loss_tok_mean"] = Metric("mean", value=pg_tok_mean)
    pg_tok_var = ((pg_loss - pg_tok_mean).square() * debug_mask).sum().detach() / mask_sum
    metrics["dbg/pg_loss_tok_std"] = Metric("mean", value=pg_tok_var.sqrt())

    # log-probs
    metrics["dbg/logp_actor_mean"] = Metric("mean", value=(log_prob * debug_mask).sum().detach() / mask_sum)
    metrics["dbg/logp_old_mean"] = Metric("mean", value=(old_log_prob * debug_mask).sum().detach() / mask_sum)
    if use_kl and ref_log_prob is not None:
        metrics["dbg/logp_ref_mean"] = Metric("mean", value=(ref_log_prob * debug_mask).sum().detach() / mask_sum)

    # KL per-token
    if use_kl and kld is not None:
        kl_tok_mean = (kld * debug_mask).sum().detach() / mask_sum
        metrics["dbg/kl_tok_mean"] = Metric("mean", value=kl_tok_mean)

    # aggregated losses
    metrics["dbg/pg_loss_agg"] = Metric("mean", value=pg_loss.detach() if isinstance(pg_loss, torch.Tensor) else pg_loss)
    metrics["dbg/final_loss"] = Metric("mean", value=policy_loss.detach() if isinstance(policy_loss, torch.Tensor) else policy_loss)

    # per-micro-batch token count
    metrics["dbg/mask_tokens"] = Metric("sum", value=mask_sum.detach())

    # reference/actor log-prob diffs
    if ref_log_prob is not None:
        metrics["debug/logprob_diff"] = Metric("mean", value=((log_prob - ref_log_prob) * debug_mask).sum().detach() / mask_sum)
        metrics["debug/logprob_diff_abs"] = Metric("mean", value=((log_prob - ref_log_prob).abs() * debug_mask).sum().detach() / mask_sum)
        metrics["debug/ref_mean"] = Metric("mean", value=(ref_log_prob * debug_mask).sum().detach() / mask_sum)
        metrics["debug/actor_mean"] = Metric("mean", value=(log_prob * debug_mask).sum().detach() / mask_sum)

    return metrics


def rankgrpo_ppo_loss(
    config: ActorConfig,
    rank_grpo_config,
    model_output,
    data: TensorDict,
    dp_group=None,  # noqa: ARG001
):
    """PPO loss with Rank-GRPO item-level importance sampling support.

    This is the entry point called by verl's ``train_mini_batch`` loop once
    per (mini-batch, epoch).  The epoch count is controlled by
    ``actor_rollout_ref.actor.ppo_epochs`` (Hydra → ``ActorConfig`` →
    ``ray_trainer._update_actor`` → ``train_mini_batch``).

    Why ppo_epochs > 1 is safe (and helps convergence):
    ──────────────────────────────────────────────────
    The clipping ratio (coef_2 = clamp(coef_1, 1-ε, 1+ε)) in
    ``_trl_clipped_pg_loss`` acts as a per-token trust-region relative to
    π_old (the rollout policy, frozen before any updates).  In early epochs
    most tokens are within the clip window and receive full gradient.  In
    later epochs, tokens that have already reached the clip boundary produce
    ZERO gradient — they cannot drift further from π_old.  Extra epochs are
    therefore *self-limiting*: they extract residual gradient from tokens
    still within bounds without overfitting those that have converged.

    With small batch sizes (e.g. 6 prompts × 8 rollouts = 48 seq vs TRL's
    384), GRPO advantage estimates are high-variance.  A single epoch leaves
    usable gradient on the table.  ppo_epochs=12 saturates most tokens at
    the clip boundary, maximizing per-batch signal extraction while the
    trust-region prevents policy collapse.
    """

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    fields = ["response_mask", "old_log_probs", "advantages"]
    if "item_token_mask" not in data.keys():
        raise KeyError("Rank-GRPO loss requires `item_token_mask` in the batch.")
    fields.append("item_token_mask")
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")

    importance_sampling_level = _cfg_get(rank_grpo_config, "importance_sampling_level", "token")
    if importance_sampling_level == "item":
        if "rank_seg_ids" not in data.keys():
            raise KeyError("Rank-GRPO item-level importance sampling requires `rank_seg_ids` in the batch.")
        fields.append("rank_seg_ids")

    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    loss_mask = data["item_token_mask"].to(bool)
    old_log_prob = _resolve_old_log_prob(
        log_prob=log_prob,
        old_log_prob=data["old_log_probs"],
        rank_grpo_config=rank_grpo_config,
    )
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)
    ref_log_prob = data.get("ref_log_prob", None)

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    loss_agg_mode = config.loss_agg_mode
    use_kl = config.use_kl_loss
    kl_coef = config.kl_loss_coef if use_kl else 0.0
    kl_type = config.kl_loss_type if use_kl else ""

    loss_mode = _cfg_get(rank_grpo_config, "loss_mode", "verl_default")
    metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # TRL-matched path
    # ------------------------------------------------------------------
    if loss_mode == "trl_match":
        # -- importance weights (TRL-style: coef_1 = exp(log_importance_weights)) --
        if importance_sampling_level == "item":
            rec_num = int(_cfg_get(rank_grpo_config, "rec_num", 20))
            log_importance_weights = _compute_item_mean_log_ratio(
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                rank_seg_ids=data["rank_seg_ids"],
                response_mask=loss_mask,
                rec_num=rec_num,
            )
            metrics["actor/rankgrpo_importance_sampling_item"] = Metric(value=1.0, aggregation=AggregationType.MEAN)
        else:
            # token-level (or sequence-level): log_ratio per token
            log_importance_weights = log_prob - old_log_prob

        # -- KL per-token (TRL computes kld before adding to per-token loss) --
        kld: torch.Tensor | None = None
        if use_kl and ref_log_prob is not None:
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=kl_type)

        # -- per-token PG loss (TRL style) --
        pg_per_token = _trl_clipped_pg_loss(
            log_importance_weights=log_importance_weights,
            advantages=advantages,
            loss_mask=loss_mask,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            kl_per_token=kld,
            kl_coef=kl_coef,
        )

        # -- aggregate (TRL: ((loss*mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean() --
        #    which is seq-mean-token-mean, matching config.loss_agg_mode)
        pg_loss = agg_loss(
            loss_mat=pg_per_token,
            loss_mask=loss_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss = pg_loss
        metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)

        # clipfrac
        with torch.no_grad():
            coef_1 = torch.exp(log_importance_weights)
            coef_2 = torch.clamp(coef_1, 1 - clip_ratio_low, 1 + clip_ratio_high)
            is_clipped = coef_1.gt(1 + clip_ratio_high) | coef_1.lt(1 - clip_ratio_low)
            pg_clipfrac = (is_clipped.float() * loss_mask.float()).sum() / loss_mask.sum().clamp(min=1)
            metrics["actor/pg_clipfrac"] = pg_clipfrac.detach().item()

        # entropy bonus
        if entropy is not None and config.entropy_coeff != 0.0:
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=loss_mask,
                loss_agg_mode=loss_agg_mode,
                **config.global_batch_info,
            )
            policy_loss = policy_loss - config.entropy_coeff * entropy_loss
            metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

        # KL logging (already baked into pg_per_token above, here just for logging)
        if use_kl and kld is not None:
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=loss_mask,
                loss_agg_mode=loss_agg_mode,
                **config.global_batch_info,
            )
            metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
            metrics["kl_coef"] = kl_coef

        # -- debug metrics --
        with torch.no_grad():
            metrics.update(_build_debug_metrics(
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                advantages=advantages,
                loss_mask=loss_mask,
                pg_loss=pg_loss,
                policy_loss=policy_loss,
                kld=kld,
                ref_log_prob=ref_log_prob,
                use_kl=use_kl,
            ))

        return policy_loss, metrics

    # ------------------------------------------------------------------
    # verl_default path (existing behaviour, unchanged)
    # ------------------------------------------------------------------
    policy_log_prob = log_prob
    if importance_sampling_level == "item":
        rec_num = int(_cfg_get(rank_grpo_config, "rec_num", 20))
        policy_log_prob = _item_level_log_prob(
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            rank_seg_ids=data["rank_seg_ids"],
            response_mask=loss_mask,
            rec_num=rec_num,
        )
        metrics["actor/rankgrpo_importance_sampling_item"] = Metric(value=1.0, aggregation=AggregationType.MEAN)

    pg_loss, pg_metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=policy_log_prob,
        advantages=advantages,
        response_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=None,
    )

    metrics.update(Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN))
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=loss_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss -= config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    if use_kl:
        ref_log_prob = data["ref_log_prob"]
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=kl_type)
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=loss_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss += kl_loss * kl_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = kl_coef

    # ---- debug metrics ----
    with torch.no_grad():
        debug_mask = loss_mask.float()
        mask_sum = debug_mask.sum().clamp(min=1)

        adv_m = (advantages * debug_mask).sum() / mask_sum
        adv_s = ((advantages - adv_m).square() * debug_mask).sum() / mask_sum
        metrics["dbg/adv_mean"] = Metric("mean", value=adv_m.detach())
        metrics["dbg/adv_std"] = Metric("mean", value=adv_s.sqrt().detach())
        metrics["dbg/adv_min"] = Metric("min", value=advantages[loss_mask].min().detach())
        metrics["dbg/adv_max"] = Metric("max", value=advantages[loss_mask].max().detach())

        metrics["dbg/logp_actor_mean"] = Metric("mean", value=(log_prob * debug_mask).sum().detach() / mask_sum)
        metrics["dbg/logp_old_mean"]   = Metric("mean", value=(old_log_prob * debug_mask).sum().detach() / mask_sum)
        if use_kl:
            metrics["dbg/logp_ref_mean"] = Metric("mean", value=(ref_log_prob * debug_mask).sum().detach() / mask_sum)

        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(torch.clamp(negative_approx_kl, -20, 20))
        pg1 = -advantages * ratio
        pg2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
        pg_per_token = torch.maximum(pg1, pg2)
        metrics["dbg/pg_loss_tok_mean"] = Metric("mean", value=(pg_per_token * debug_mask).sum().detach() / mask_sum)
        metrics["dbg/pg_loss_tok_std"]  = Metric("mean", value=((pg_per_token - (pg_per_token*debug_mask).sum()/mask_sum).square()*debug_mask).sum().sqrt().detach() / mask_sum.sqrt())

        if use_kl:
            metrics["dbg/kl_tok_mean"] = Metric("mean", value=(kld * debug_mask).sum().detach() / mask_sum)

        metrics["dbg/pg_loss_agg"] = Metric("mean", value=pg_loss.detach())
        metrics["dbg/final_loss"]  = Metric("mean", value=policy_loss.detach())
        metrics["dbg/mask_tokens"] = Metric("sum", value=mask_sum.detach())

        if ref_log_prob is not None:
            metrics["debug/logprob_diff"] = Metric("mean", value=((log_prob - ref_log_prob) * debug_mask).sum().detach() / mask_sum)
            metrics["debug/logprob_diff_abs"] = Metric("mean", value=((log_prob - ref_log_prob).abs() * debug_mask).sum().detach() / mask_sum)
            metrics["debug/ref_mean"] = Metric("mean", value=(ref_log_prob * debug_mask).sum().detach() / mask_sum)
            metrics["debug/actor_mean"] = Metric("mean", value=(log_prob * debug_mask).sum().detach() / mask_sum)

    return policy_loss, metrics
