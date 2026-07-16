"""OpenOneRec actor loss with RankGRPO-style logprob debug metrics."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from verl.utils.metric import Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


def _collect_logprob_debug_metrics(
    *,
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, Metric]:
    debug_mask = response_mask.float()
    mask_sum = debug_mask.sum().clamp(min=1)
    return {
        "debug/logprob_diff": Metric(
            "mean",
            value=((log_prob - ref_log_prob) * debug_mask).sum().detach() / mask_sum,
        ),
        "debug/logprob_diff_abs": Metric(
            "mean",
            value=((log_prob - ref_log_prob).abs() * debug_mask).sum().detach() / mask_sum,
        ),
        "debug/ref_mean": Metric("mean", value=(ref_log_prob * debug_mask).sum().detach() / mask_sum),
        "debug/actor_mean": Metric("mean", value=(log_prob * debug_mask).sum().detach() / mask_sum),
        "dbg/logp_actor_mean": Metric("mean", value=(log_prob * debug_mask).sum().detach() / mask_sum),
        "dbg/logp_ref_mean": Metric("mean", value=(ref_log_prob * debug_mask).sum().detach() / mask_sum),
    }


def onerec_ppo_loss(
    config: ActorConfig,
    model_output,
    data: TensorDict,
    dp_group=None,
):
    """Standard verl PPO loss plus actor/ref logprob probes used in RankGRPO profiling."""

    debug_metrics: dict[str, Metric] = {}
    with torch.no_grad():
        if "ref_log_prob" in data.keys() and "response_mask" in data.keys():
            log_prob = no_padding_2_padding(model_output["log_probs"], data)
            # IMPORTANT: data.select() returns a view sharing memory with
            # the original TensorDict.  .to_padded_tensor() mutates in-place,
            # which would corrupt data before ppo_loss() runs below.  Clone
            # first so the probe doesn't destroy the real KL computation.
            probe_data = (
                data.select("response_mask", "ref_log_prob").clone().to_padded_tensor()
            )
            response_mask = probe_data["response_mask"].to(bool)
            ref_log_prob = probe_data["ref_log_prob"]
            if response_mask.any():
                debug_metrics.update(
                    _collect_logprob_debug_metrics(
                        log_prob=log_prob,
                        ref_log_prob=ref_log_prob,
                        response_mask=response_mask,
                    )
                )

    policy_loss, metrics = ppo_loss(config, model_output, data, dp_group=dp_group)
    metrics.update(debug_metrics)
    return policy_loss, metrics
