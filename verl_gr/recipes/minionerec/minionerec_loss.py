"""MiniOneRec REINFORCE-style policy loss.

Matches the original MiniOneRec ``compute_loss`` from the ReReTrainer:
  - ``exp(logp - logp.detach())`` instead of PPO's ``exp(logp - old_logp)``
  - No importance-ratio weighting → pure REINFORCE / score-function update
  - No PPO clipping
  - KL is handled by verl's existing ``use_kl_loss`` mechanism (separate term,
    identical gradient to original's embedded KL for ``seq-mean-token-mean``
    aggregation).
"""

from __future__ import annotations

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig


@register_policy_loss("minionerec_reinforce")
def compute_policy_loss_minionerec_reinforce(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: ActorConfig | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """REINFORCE-style policy loss matching original MiniOneRec.

    Uses ``exp(logp - logp.detach()) * advantages`` which evaluates to
    ``1.0 * advantages`` in forward but whose gradient is
    ``advantages * grad(logp)`` — a pure score-function estimator with
    no importance-ratio weighting.

    This eliminates the PPO-ratio multiplier ``exp(logp - old_logp)``
    that the vanilla loss uses, which is the primary mechanism-level
    difference vs. the original ReReTrainer objective.
    """
    # REINFORCE term: grad(logp) * advantages, no old_logp dependency.
    # exp(logp - logp.detach()) → forward == 1.0, grad == grad(logp)
    per_token_loss = torch.exp(log_prob - log_prob.detach()) * advantages

    pg_losses = -per_token_loss

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **(config.global_batch_info if config is not None else {}),
    )

    ppo_kl = verl_F.masked_mean(-(log_prob - old_log_prob.detach()), response_mask)

    return pg_loss, {
        "actor/pg_clipfrac": torch.tensor(0.0, device=pg_loss.device),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": torch.tensor(0.0, device=pg_loss.device),
    }
