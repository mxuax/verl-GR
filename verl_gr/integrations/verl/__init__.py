"""Thin verl runtime integration bridges."""

from verl_gr.integrations.verl.rl_runtime import RLRuntimeConfig, RuntimeTrainerHandle, VerlRLRuntime
from verl_gr.integrations.verl.worker_factory import (
    WorkerFactoryConfig,
    WorkerRole,
    WorkerRouting,
    build_worker_routing,
)

__all__ = [
    "RLRuntimeConfig",
    "RuntimeTrainerHandle",
    "VerlRLRuntime",
    "WorkerFactoryConfig",
    "WorkerRole",
    "WorkerRouting",
    "build_worker_routing",
]
