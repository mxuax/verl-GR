"""Skeleton distillation trainer for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from verl_gr.contracts.distill_contract import DistillInput, DistillOutput


@dataclass
class DistillTrainer:
    """Stage-level orchestrator skeleton for distillation."""

    name: str = "distill_trainer"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_input(self, distill_input: DistillInput) -> None:
        """Validate the minimal distillation input contract."""

        if not distill_input.tokenized_samples:
            raise ValueError("DistillInput.tokenized_samples must not be empty.")

    def run(self, distill_input: DistillInput) -> DistillOutput:
        """Execute distillation.

        Phase A only provides the interface skeleton. Concrete runtime-backed
        implementations are expected to be added through later integrations.
        """

        self.validate_input(distill_input)
        raise NotImplementedError("DistillTrainer.run is intentionally deferred to later phases.")

