"""OneRec recipe adapter re-exporting trainer-level ray logic."""

from verl_gr.trainers.rl_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)

__all__ = [
    "Role",
    "ResourcePoolManager",
    "RayPPOTrainer",
    "apply_kl_penalty",
    "compute_response_mask",
    "compute_advantage",
]
