"""Objective schemas for SFT, distillation, and RL stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ObjectiveKind(str, Enum):
    """Supported objective kinds across stages."""

    SFT = "sft"
    DISTILL = "distill"
    RL = "rl"


@dataclass(frozen=True)
class ObjectiveSchema:
    """Base schema shared by all stage objectives."""

    kind: ObjectiveKind
    name: str
    reduction: str = "mean"
    weight: float = 1.0
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SFTLossSchema(ObjectiveSchema):
    """Schema for supervised training losses."""

    label_smoothing: float = 0.0
    ignore_index: int = -100


@dataclass(frozen=True)
class DistillLossSchema(ObjectiveSchema):
    """Schema for teacher-student alignment losses."""

    temperature: float = 1.0
    teacher_weight: float = 1.0


@dataclass(frozen=True)
class RewardComponent:
    """Schema for one reward component in an RL reward composition."""

    name: str
    weight: float
    enabled: bool = True
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RLRewardSchema(ObjectiveSchema):
    """Schema for RL reward composition and normalization."""

    components: tuple[RewardComponent, ...] = ()
    normalization: str = "none"
    constrained_decoding_aware: bool = False

