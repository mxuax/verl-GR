"""Distillation stage contract for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from verl_gr.contracts.artifact_contract import CheckpointArtifact, StageConfigArtifact, TokenizerArtifact
from verl_gr.contracts.objective_schema import DistillLossSchema
from verl_gr.contracts.tokenizer_contract import TokenizedSample


@dataclass(frozen=True)
class DistillInput:
    """Input contract for distillation stage execution."""

    tokenized_samples: Sequence[TokenizedSample]
    student_model_path: Path
    teacher_model_path: Path
    tokenizer_artifact: TokenizerArtifact
    config_artifact: StageConfigArtifact
    objective: DistillLossSchema


@dataclass(frozen=True)
class DistillOutput:
    """Output contract for distillation stage execution."""

    checkpoint: CheckpointArtifact
    metrics: Mapping[str, Any] = field(default_factory=dict)


class DistillContract(Protocol):
    """Interface that distillation trainer implementations must satisfy."""

    def run(self, distill_input: DistillInput) -> DistillOutput:
        """Run distillation and return a checkpoint plus metrics."""

