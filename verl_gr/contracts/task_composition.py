"""Task composition rules for verl-GR recipes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from verl_gr.contracts.sample_schema import RepresentationType, TaskType


class StageName(str, Enum):
    """Supported high-level stages in a task flow."""

    TOKENIZER = "tokenizer"
    SFT = "sft"
    DISTILL = "distill"
    RL = "rl"
    EVAL = "eval"


@dataclass(frozen=True)
class TaskComposition:
    """Recipe-level composition for one task path."""

    task_type: TaskType
    representation_type: RepresentationType
    stages: tuple[StageName, ...]

    def validate(self) -> None:
        """Validate stage order for the composed task."""

        positions = {stage: idx for idx, stage in enumerate(self.stages)}
        if StageName.TOKENIZER not in positions:
            raise ValueError("Every task composition must include the tokenizer stage.")
        if StageName.EVAL in positions and positions[StageName.TOKENIZER] > positions[StageName.EVAL]:
            raise ValueError("Tokenizer must appear before evaluation.")
        if StageName.RL in positions and StageName.TOKENIZER in positions:
            if positions[StageName.TOKENIZER] > positions[StageName.RL]:
                raise ValueError("Tokenizer must appear before RL.")


def validate_compositions(compositions: Iterable[TaskComposition]) -> None:
    """Validate a collection of task compositions."""

    for composition in compositions:
        composition.validate()

