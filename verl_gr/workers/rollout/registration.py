"""Registration helpers for custom verl-GR rollout paths."""

from __future__ import annotations

from importlib import import_module

TWO_STAGE_ASYNC_ROLLOUT_PATH = "verl_gr.workers.rollout.two_stage_vllm_rollout.TwoStagevLLMRollout"
CONSTRAINED_BEAM_ASYNC_ROLLOUT_PATH = "verl_gr.workers.rollout.constrained_beam_vllm_rollout.ConstrainedBeamvLLMRollout"
RANKGRPO_ASYNC_ROLLOUT_PATH = "verl_gr.workers.rollout.rankgrpo_vllm_rollout.RankGRPOvLLMRollout"


def register_two_stage_rollout_class() -> None:
    rollout_base_mod = import_module("verl.workers.rollout.base")
    rollout_registry = getattr(rollout_base_mod, "_ROLLOUT_REGISTRY")
    rollout_registry[("two_stage", "async")] = TWO_STAGE_ASYNC_ROLLOUT_PATH


def register_constrained_beam_rollout_class() -> None:
    rollout_base_mod = import_module("verl.workers.rollout.base")
    rollout_registry = getattr(rollout_base_mod, "_ROLLOUT_REGISTRY")
    rollout_registry[("constrained_beam", "async")] = CONSTRAINED_BEAM_ASYNC_ROLLOUT_PATH


def register_rankgrpo_rollout_class() -> None:
    rollout_base_mod = import_module("verl.workers.rollout.base")
    rollout_registry = getattr(rollout_base_mod, "_ROLLOUT_REGISTRY")
    rollout_registry[("rankgrpo", "async")] = RANKGRPO_ASYNC_ROLLOUT_PATH


def register_two_stage_replica() -> None:
    rollout_replica_mod = import_module("verl.workers.rollout.replica")
    rollout_replica_registry = getattr(rollout_replica_mod, "RolloutReplicaRegistry")
    two_stage_replica = getattr(
        import_module("verl_gr.workers.rollout.two_stage_vllm_async"),
        "TwoStagevLLMReplica",
    )
    rollout_replica_registry.register("two_stage", lambda: two_stage_replica)


def register_constrained_beam_replica() -> None:
    rollout_replica_mod = import_module("verl.workers.rollout.replica")
    rollout_replica_registry = getattr(rollout_replica_mod, "RolloutReplicaRegistry")
    constrained_beam_replica = getattr(
        import_module("verl_gr.workers.rollout.constrained_beam_vllm_async"),
        "ConstrainedBeamvLLMReplica",
    )
    rollout_replica_registry.register("constrained_beam", lambda: constrained_beam_replica)


def register_rankgrpo_replica() -> None:
    rollout_replica_mod = import_module("verl.workers.rollout.replica")
    rollout_replica_registry = getattr(rollout_replica_mod, "RolloutReplicaRegistry")
    rankgrpo_replica = getattr(
        import_module("verl_gr.workers.rollout.rankgrpo_vllm_async"),
        "RankGRPOvLLMReplica",
    )
    rollout_replica_registry.register("rankgrpo", lambda: rankgrpo_replica)
