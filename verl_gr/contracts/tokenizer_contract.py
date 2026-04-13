"""Tokenizer stage contract for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from verl_gr.contracts.artifact_contract import TokenizerArtifact
from verl_gr.contracts.sample_schema import BaseSample, RepresentationType


class TokenizerKind(str, Enum):
    """Supported tokenizer families."""

    SID = "sid"
    NATURAL_LANGUAGE = "natural_language"


@dataclass(frozen=True)
class TokenizerInput:
    """Input contract consumed by tokenizer implementations."""

    sample: BaseSample
    raw_payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizedSample:
    """Output contract emitted by tokenizer implementations."""

    sample_id: str
    representation_type: RepresentationType
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...] | None = None
    special_token_map: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizerSpec:
    """Static description of a tokenizer implementation."""

    name: str
    kind: TokenizerKind
    required_sample_fields: tuple[str, ...]
    optional_sample_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatMessage:
    """Normalized chat message used across tokenizer/RL handoff."""

    role: str
    content: str


@dataclass(frozen=True)
class PromptPackage:
    """Prompt payload plus reward annotation extracted from chat input."""

    prompt_messages: tuple[ChatMessage, ...]
    reward_payload: Mapping[str, Any]


class TokenizerContract(Protocol):
    """Interface that every framework tokenizer implementation must satisfy."""

    spec: TokenizerSpec

    def tokenize(self, batch: Sequence[TokenizerInput]) -> Sequence[TokenizedSample]:
        """Convert raw samples into tokenized samples."""

    def build_artifact(self) -> TokenizerArtifact:
        """Emit tokenizer artifacts consumed by downstream stages."""

