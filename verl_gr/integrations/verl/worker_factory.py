"""Role to worker routing bridge for verl runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkerRole(str, Enum):
    """Canonical worker roles for RL runtime wiring."""

    ACTOR_ROLLOUT = "ActorRollout"
    CRITIC = "Critic"
    REF_POLICY = "RefPolicy"
    REWARD_MODEL = "RewardModel"


@dataclass(frozen=True)
class WorkerFactoryConfig:
    """Inputs needed to build role-worker mappings."""

    actor_strategy: str = "fsdp"
    critic_enabled: bool = True
    reward_model_enabled: bool = False
    use_reference_policy: bool = False
    rollout_name: str = "vllm"
    rollout_mode: str = "sync"
    resource_pool_id: str = "global_pool"


@dataclass(frozen=True)
class WorkerRouting:
    """Resolved routing from logical roles to worker implementation IDs."""

    role_worker_mapping: dict[WorkerRole, str]
    role_pool_mapping: dict[WorkerRole, str]


def build_worker_routing(config: WorkerFactoryConfig) -> WorkerRouting:
    """Build worker routing with OneRec two-stage special-case support."""

    actor_worker = "verl.workers.fsdp_workers.ActorRolloutRefWorker"
    if config.actor_strategy not in {"fsdp", "fsdp2"}:
        actor_worker = "verl.workers.megatron_workers.ActorRolloutRefWorker"
    if config.rollout_name == "two_stage":
        actor_worker = "verl_gr.recipes.openonerec.onerec_fsdp_workers.OneRecActorRolloutRefWorker"
    if config.rollout_mode == "async":
        actor_worker = "verl.workers.fsdp_workers.AsyncActorRolloutRefWorker"

    role_worker_mapping: dict[WorkerRole, str] = {
        WorkerRole.ACTOR_ROLLOUT: actor_worker,
    }
    role_pool_mapping: dict[WorkerRole, str] = {
        WorkerRole.ACTOR_ROLLOUT: config.resource_pool_id,
    }

    if config.critic_enabled:
        role_worker_mapping[WorkerRole.CRITIC] = "verl.workers.fsdp_workers.CriticWorker"
        role_pool_mapping[WorkerRole.CRITIC] = config.resource_pool_id
    if config.reward_model_enabled:
        role_worker_mapping[WorkerRole.REWARD_MODEL] = "verl.workers.fsdp_workers.RewardModelWorker"
        role_pool_mapping[WorkerRole.REWARD_MODEL] = config.resource_pool_id
    if config.use_reference_policy:
        role_worker_mapping[WorkerRole.REF_POLICY] = actor_worker
        role_pool_mapping[WorkerRole.REF_POLICY] = config.resource_pool_id

    return WorkerRouting(role_worker_mapping=role_worker_mapping, role_pool_mapping=role_pool_mapping)

