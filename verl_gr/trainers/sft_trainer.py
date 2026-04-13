"""Skeleton SFT trainer for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from verl_gr.contracts.sft_contract import SFTInput, SFTOutput


@dataclass
class SFTTrainer:
    """Stage-level orchestrator skeleton for SFT."""

    name: str = "sft_trainer"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_input(self, sft_input: SFTInput) -> None:
        """Validate the minimal SFT input contract."""

        if not sft_input.tokenized_samples:
            raise ValueError("SFTInput.tokenized_samples must not be empty.")

    def run(self, sft_input: SFTInput) -> SFTOutput:
        """Execute SFT training.

        Phase A only provides the interface skeleton. Concrete runtime-backed
        implementations are expected to be added through later integrations.
        """

        self.validate_input(sft_input)
        raise NotImplementedError("SFTTrainer.run is intentionally deferred to later phases.")

