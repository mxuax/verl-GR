"""RL stage contract for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from verl_gr.contracts.artifact_contract import CheckpointArtifact, RewardOrDecodingArtifact, StageConfigArtifact, TokenizerArtifact
from verl_gr.contracts.objective_schema import RLRewardSchema
from verl_gr.contracts.tokenizer_contract import TokenizedSample


@dataclass(frozen=True)
class RLInput:
    """Input contract for RL stage execution."""

    tokenized_samples: Sequence[TokenizedSample]
    policy_model_path: Path
    tokenizer_artifact: TokenizerArtifact
    config_artifact: StageConfigArtifact
    reward_schema: RLRewardSchema
    reference_model_path: Path | None = None
    reward_or_decoding_artifact: RewardOrDecodingArtifact | None = None


@dataclass(frozen=True)
class RLOutput:
    """Output contract for RL stage execution."""

    checkpoint: CheckpointArtifact
    metrics: Mapping[str, Any] = field(default_factory=dict)
    traces: Mapping[str, Any] = field(default_factory=dict)


class RLContract(Protocol):
    """Interface that RL trainer implementations must satisfy."""

    def run(self, rl_input: RLInput) -> RLOutput:
        """Run the RL stage and return a checkpoint plus metrics."""


@dataclass(frozen=True)
class RLWorkloadProfile:
    """Execution profile resolved from trainer config for RL workload wiring."""

    actor_strategy: str
    rollout_name: str
    rollout_mode: str
    use_legacy_worker_impl: str = "auto"

    def requires_onerec_actor_worker(self) -> bool:
        """Whether OneRec-specific actor/rollout worker should be used."""

        return (
            self.use_legacy_worker_impl != "disable"
            and self.rollout_name == "two_stage"
            and self.actor_strategy in {"fsdp", "fsdp2"}
            and self.rollout_mode != "async"
        )

