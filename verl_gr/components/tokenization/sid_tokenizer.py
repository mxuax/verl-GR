"""SID tokenizer skeleton for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from verl_gr.contracts.artifact_contract import TokenizerArtifact
from verl_gr.contracts.sample_schema import RepresentationType
from verl_gr.contracts.tokenizer_contract import (
    TokenizedSample,
    TokenizerInput,
    TokenizerKind,
    TokenizerSpec,
)


@dataclass
class SIDTokenizer:
    """Placeholder SID tokenizer implementation.

    Concrete tokenization logic is intentionally deferred to later phases.
    """

    artifact_root: Path
    spec: TokenizerSpec = field(
        default_factory=lambda: TokenizerSpec(
            name="sid_tokenizer",
            kind=TokenizerKind.SID,
            required_sample_fields=("sample_id", "task_type", "input_context"),
            optional_sample_fields=("target", "metadata"),
        )
    )

    def tokenize(self, batch: Sequence[TokenizerInput]) -> Sequence[TokenizedSample]:
        """Tokenize a batch of SID-path samples."""

        raise NotImplementedError("SID tokenization is intentionally deferred to later phases.")

    def build_artifact(self) -> TokenizerArtifact:
        """Return the tokenizer artifact declaration for SID paths."""

        return TokenizerArtifact(
            tokenizer_root=self.artifact_root,
            representation_type=RepresentationType.SID,
            special_token_file=self.artifact_root / "special_tokens.json",
            representation_schema_file=self.artifact_root / "sid_schema.json",
        )
