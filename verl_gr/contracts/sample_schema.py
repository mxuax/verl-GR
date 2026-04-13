"""Shared sample schemas for verl-GR tasks.

The goal of this module is to freeze the minimum data protocol consumed by
tokenization, SFT, distillation, and RL stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class TaskType(str, Enum):
    """Supported high-level task families."""

    OPENONEREC = "openonerec"
    MINIONEREC = "minionerec"
    RANK_GRPO = "rank_grpo"
    CUSTOM = "custom"


class RepresentationType(str, Enum):
    """Representation style used by a task path."""

    SID = "sid"
    NATURAL_LANGUAGE = "natural_language"


@dataclass(frozen=True)
class BaseSample:
    """Shared fields that every training sample must provide."""

    sample_id: str
    task_type: TaskType
    input_context: str
    target: str | Sequence[str] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SIDPathFields:
    """Fields specific to SID-based recommendation paths."""

    item_history: Sequence[str] = field(default_factory=tuple)
    sid_target: Sequence[str] = field(default_factory=tuple)
    sid_delimiter: str = ""


@dataclass(frozen=True)
class NLPathFields:
    """Fields specific to natural-language recommendation paths."""

    conversation: Sequence[Mapping[str, str]] = field(default_factory=tuple)
    nl_target: str | Sequence[str] | None = None


@dataclass(frozen=True)
class StageFieldRequirements:
    """Minimum required fields for a training stage."""

    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()


SFT_MINIMUM_FIELDS = StageFieldRequirements(
    required_fields=("sample_id", "task_type", "input_context"),
    optional_fields=("target", "metadata"),
)

DISTILL_MINIMUM_FIELDS = StageFieldRequirements(
    required_fields=("sample_id", "task_type", "input_context"),
    optional_fields=("target", "metadata"),
)

RL_MINIMUM_FIELDS = StageFieldRequirements(
    required_fields=("sample_id", "task_type", "input_context"),
    optional_fields=("target", "metadata"),
)


@dataclass(frozen=True)
class SampleSchemaBundle:
    """Container for all framework-level sample schema definitions."""

    base_sample: type[BaseSample] = BaseSample
    sid_fields: type[SIDPathFields] = SIDPathFields
    nl_fields: type[NLPathFields] = NLPathFields
    sft_requirements: StageFieldRequirements = SFT_MINIMUM_FIELDS
    distill_requirements: StageFieldRequirements = DISTILL_MINIMUM_FIELDS
    rl_requirements: StageFieldRequirements = RL_MINIMUM_FIELDS

