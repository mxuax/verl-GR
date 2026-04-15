"""Thin verl runtime integration bridges."""

from verl_gr.integrations.verl.rl_runtime import RLRuntimeConfig, RuntimeTrainerHandle, VerlRLRuntime
from verl_gr.integrations.verl.ray_trainer import RayPPOTrainerRuntime

__all__ = [
    "RLRuntimeConfig",
    "RayPPOTrainerRuntime",
    "RuntimeTrainerHandle",
    "VerlRLRuntime",
]
