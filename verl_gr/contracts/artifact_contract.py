"""Artifact contracts shared across verl-GR stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from verl_gr.contracts.sample_schema import RepresentationType


@dataclass(frozen=True)
class TokenizerArtifact:
    """Artifacts emitted by the tokenization stage."""

    tokenizer_root: Path
    representation_type: RepresentationType
    special_token_file: Path | None = None
    representation_schema_file: Path | None = None


@dataclass(frozen=True)
class StageConfigArtifact:
    """Configuration artifact consumed by stages."""

    task_config_path: Path
    stage_config_path: Path | None = None
    objective_config_path: Path | None = None


@dataclass(frozen=True)
class CheckpointArtifact:
    """Checkpoint artifact produced by a training stage."""

    stage_name: str
    checkpoint_root: Path
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardOrDecodingArtifact:
    """Auxiliary artifact tied to RL reward or decoding behavior."""

    reward_schema_path: Path | None = None
    decoding_policy_path: Path | None = None


@dataclass(frozen=True)
class ArtifactBundle:
    """Top-level artifact bundle passed between phases."""

    tokenizer: TokenizerArtifact
    config: StageConfigArtifact
    checkpoints: tuple[CheckpointArtifact, ...] = ()
    reward_or_decoding: RewardOrDecodingArtifact | None = None

