"""Rank-GRPO task runtime wiring."""

from __future__ import annotations

from typing import Any

from verl.single_controller.ray import RayWorkerGroup
from verl.utils.fs import copy_to_local
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from verl_gr.recipes.rankgrpo.rankgrpo_tokenizer import build_rankgrpo_tokenizer_and_processor
from verl_gr.recipes.task_runtime import RecipeTaskRuntime

__all__ = ["RankGRPOTask"]


class RankGRPOTask(RecipeTaskRuntime):
    """Rank-GRPO task-specific runtime preparation."""

    def prepare(self, config) -> dict[str, Any]:
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        rank_cfg = config.data.get("rankgrpo", {}) or {}
        built = build_rankgrpo_tokenizer_and_processor(
            local_path,
            trust_remote_code=config.data.get("trust_remote_code", False),
            use_processor=rank_cfg.get("use_processor", False),
            rank_separator=rank_cfg.get("rank_separator", "\n"),
            force_pad_to_eos=rank_cfg.get("force_pad_to_eos", True),
        )

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2", "megatron"}:
            ray_worker_group_cls = RayWorkerGroup
            actor_rollout_cls = ActorRolloutRefWorker
            critic_worker = TrainingWorker
        else:
            raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")

        return {
            "tokenizer": built["tokenizer"],
            "processor": built["processor"],
            "rank_separator_token_ids": built["rank_separator_token_ids"],
            "actor_rollout_cls": actor_rollout_cls,
            "critic_worker": critic_worker,
            "reward_model_cfg": None,
            "ray_worker_group_cls": ray_worker_group_cls,
        }
