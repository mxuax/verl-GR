"""Rank-GRPO worker customizations."""

from __future__ import annotations

from functools import partial

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

from verl_gr.recipes.rankgrpo.rankgrpo_loss import rankgrpo_ppo_loss
from verl_gr.workers.grad_hooks import install_grad_hooks


class RankGRPOActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor/rollout/ref worker that installs the Rank-GRPO PPO loss."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        # Install per-layer gradient hooks on the FSDP engine
        install_grad_hooks()

        if self._is_actor and self.actor is not None and not self.distillation_enabled:
            actor_config = getattr(getattr(self, "loss_fn", None), "keywords", {}).get("config")
            if actor_config is None or actor_config.model_config.get("model_type", "language_model") == "diffusion_model":
                return

            rank_grpo_config = self.config.get("rank_grpo", {}) or {}
            self.loss_fn = partial(rankgrpo_ppo_loss, config=actor_config, rank_grpo_config=rank_grpo_config)
            self.actor.set_loss_fn(self.loss_fn)
