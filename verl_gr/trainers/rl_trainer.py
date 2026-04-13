"""Skeleton RL trainer for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from verl_gr.contracts.rl_contract import RLInput, RLOutput


@dataclass
class RLTrainer:
    """Stage-level orchestrator skeleton for RL."""

    name: str = "rl_trainer"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_bridge: Any | None = None

    def validate_input(self, rl_input: RLInput) -> None:
        """Validate the minimal RL input contract."""

        if not rl_input.tokenized_samples:
            raise ValueError("RLInput.tokenized_samples must not be empty.")

    def run(self, rl_input: RLInput) -> RLOutput:
        """Execute RL training.

        Phase A only provides the interface skeleton. Concrete runtime-backed
        implementations are expected to be added through later integrations.
        """

        self.validate_input(rl_input)
        if self.runtime_bridge is not None and hasattr(self.runtime_bridge, "run"):
            return self.runtime_bridge.run(rl_input)
        raise NotImplementedError("RLTrainer.run is intentionally deferred to later phases.")

