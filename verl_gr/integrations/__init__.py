"""Integration boundary for runtime bridges added in later phases."""

from verl_gr.integrations.openonerec import (
    OpenOneRecDistillPipeline,
    OpenOneRecRLPipeline,
    OpenOneRecSFTPipeline,
)
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
    "OpenOneRecSFTPipeline",
    "OpenOneRecDistillPipeline",
    "OpenOneRecRLPipeline",
    "RLRuntimeConfig",
    "RuntimeTrainerHandle",
    "VerlRLRuntime",
    "WorkerFactoryConfig",
    "WorkerRole",
    "WorkerRouting",
    "build_worker_routing",
]

