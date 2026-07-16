"""Rank-GRPO task runtime wiring."""

from __future__ import annotations

from typing import Any

from omegaconf import open_dict

from verl_gr.recipes.rankgrpo.rankgrpo_worker import RankGRPOActorRolloutRefWorker
from verl_gr.recipes.task_runtime import RecipeTaskRuntime
from verl_gr.workers.rollout.registration import (
    register_rankgrpo_replica,
    register_rankgrpo_rollout_class,
)

__all__ = ["RankGRPOTask"]


class RankGRPOTask(RecipeTaskRuntime):
    """Rank-GRPO task-specific runtime preparation."""

    def prepare(self, config) -> dict[str, Any]:
        with open_dict(config.actor_rollout_ref):
            config.actor_rollout_ref.rank_grpo = config.algorithm.get("rank_grpo", {}) or {}
        return super().prepare(config)

    def configure_rollout(self, config) -> None:
        rollout_name = config.actor_rollout_ref.rollout.get("name")
        if rollout_name == "rankgrpo":
            register_rankgrpo_replica()
            register_rankgrpo_rollout_class()

    def get_actor_rollout_ref_worker(self, config):
        rollout_name = config.actor_rollout_ref.rollout.get("name")
        if rollout_name == "rankgrpo":
            return RankGRPOActorRolloutRefWorker
        return super().get_actor_rollout_ref_worker(config)
