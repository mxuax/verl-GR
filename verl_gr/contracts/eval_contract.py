"""Evaluation stage contract for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from verl_gr.contracts.artifact_contract import CheckpointArtifact, StageConfigArtifact


@dataclass(frozen=True)
class EvalInput:
    """Input contract for evaluation stage execution."""

    checkpoints: Sequence[CheckpointArtifact]
    validation_data_path: Path
    config_artifact: StageConfigArtifact


@dataclass(frozen=True)
class EvalOutput:
    """Output contract for evaluation stage execution."""

    report: Mapping[str, Any] = field(default_factory=dict)


class EvalContract(Protocol):
    """Interface that evaluation implementations must satisfy."""

    def run(self, eval_input: EvalInput) -> EvalOutput:
        """Run evaluation and return a report."""

