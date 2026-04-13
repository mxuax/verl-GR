"""Integration boundary for runtime bridges added in later phases."""

from verl_gr.integrations.runtime_adapter import TaskRuntime
from verl_gr.integrations.verl import (
    RLRuntimeConfig,
    RuntimeTrainerHandle,
    VerlRLRuntime,
    WorkerFactoryConfig,
    WorkerRole,
    WorkerRouting,
    build_worker_routing,
)

__all__ = [
    "TaskRuntime",
    "RLRuntimeConfig",
    "RuntimeTrainerHandle",
    "VerlRLRuntime",
    "WorkerFactoryConfig",
    "WorkerRole",
    "WorkerRouting",
    "build_worker_routing",
]

