"""Shared registration helpers for the two-stage rollout path."""

from __future__ import annotations

from importlib import import_module

TWO_STAGE_ASYNC_ROLLOUT_PATH = "verl_gr.workers.rollout.two_stage_vllm_rollout.TwoStagevLLMRollout"


def register_two_stage_rollout_class() -> None:
    rollout_base_mod = import_module("verl.workers.rollout.base")
    rollout_registry = getattr(rollout_base_mod, "_ROLLOUT_REGISTRY")
    rollout_registry[("two_stage", "async")] = TWO_STAGE_ASYNC_ROLLOUT_PATH


def register_two_stage_replica() -> None:
    rollout_replica_mod = import_module("verl.workers.rollout.replica")
    rollout_replica_registry = getattr(rollout_replica_mod, "RolloutReplicaRegistry")
    two_stage_replica = getattr(
        import_module("verl_gr.workers.rollout.two_stage_vllm_async"),
        "TwoStagevLLMReplica",
    )
    rollout_replica_registry.register("two_stage", lambda: two_stage_replica)
