"""SFT stage contract for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from verl_gr.contracts.artifact_contract import CheckpointArtifact, StageConfigArtifact, TokenizerArtifact
from verl_gr.contracts.objective_schema import SFTLossSchema
from verl_gr.contracts.tokenizer_contract import TokenizedSample


@dataclass(frozen=True)
class SFTInput:
    """Input contract for SFT stage execution."""

    tokenized_samples: Sequence[TokenizedSample]
    model_path: Path
    tokenizer_artifact: TokenizerArtifact
    config_artifact: StageConfigArtifact
    objective: SFTLossSchema


@dataclass(frozen=True)
class SFTOutput:
    """Output contract for SFT stage execution."""

    checkpoint: CheckpointArtifact
    metrics: Mapping[str, Any] = field(default_factory=dict)


class SFTContract(Protocol):
    """Interface that SFT trainer implementations must satisfy."""

    def run(self, sft_input: SFTInput) -> SFTOutput:
        """Run the SFT stage and return a checkpoint plus metrics."""

