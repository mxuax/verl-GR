"""Natural-language tokenizer skeleton for verl-GR."""

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
class NLTokenizer:
    """Placeholder natural-language tokenizer implementation."""

    artifact_root: Path
    spec: TokenizerSpec = field(
        default_factory=lambda: TokenizerSpec(
            name="nl_tokenizer",
            kind=TokenizerKind.NATURAL_LANGUAGE,
            required_sample_fields=("sample_id", "task_type", "input_context"),
            optional_sample_fields=("target", "metadata"),
        )
    )

    def tokenize(self, batch: Sequence[TokenizerInput]) -> Sequence[TokenizedSample]:
        """Tokenize a batch of natural-language path samples."""

        raise NotImplementedError("NL tokenization is intentionally deferred to later phases.")

    def build_artifact(self) -> TokenizerArtifact:
        """Return the tokenizer artifact declaration for NL paths."""

        return TokenizerArtifact(
            tokenizer_root=self.artifact_root,
            representation_type=RepresentationType.NATURAL_LANGUAGE,
            special_token_file=self.artifact_root / "special_tokens.json",
            representation_schema_file=self.artifact_root / "nl_schema.json",
        )
