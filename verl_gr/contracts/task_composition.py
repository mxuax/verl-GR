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


@dataclass(frozen=True)
class StageRuntimeSpec:
    """Runtime entrypoint binding for a specific stage."""

    stage: StageName
    entrypoint_module: str
    config_name: str | None = None


@dataclass(frozen=True)
class TaskRuntimeComposition:
    """Runtime-level stage bindings attached to one task composition."""

    composition: TaskComposition
    stage_runtimes: tuple[StageRuntimeSpec, ...]

    def validate(self) -> None:
        """Validate that runtime specs only target declared stages."""

        self.composition.validate()
        stage_set = set(self.composition.stages)
        for runtime_spec in self.stage_runtimes:
            if runtime_spec.stage not in stage_set:
                raise ValueError(
                    f"Runtime spec stage {runtime_spec.stage.value} is not in task composition."
                )

    def runtime_for_stage(self, stage: StageName) -> StageRuntimeSpec | None:
        """Return runtime binding for one stage."""

        for runtime_spec in self.stage_runtimes:
            if runtime_spec.stage == stage:
                return runtime_spec
        return None


def validate_compositions(compositions: Iterable[TaskComposition]) -> None:
    """Validate a collection of task compositions."""

    for composition in compositions:
        composition.validate()

