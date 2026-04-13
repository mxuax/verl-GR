"""OpenOneRec SFT adapter owned by recipe layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verl_gr.contracts.artifact_contract import CheckpointArtifact
from verl_gr.contracts.sft_contract import SFTInput, SFTOutput


@dataclass
class OpenOneRecSFTPipeline:
    """Translate SFT contract input to OpenOneRec runtime arguments."""

    entrypoint: str = "OpenOneRec.verl_rl.recipe.sft.main_sft"

    def build_runtime_args(self, sft_input: SFTInput) -> dict[str, Any]:
        return {
            "model_path": str(sft_input.model_path),
            "tokenizer_root": str(sft_input.tokenizer_artifact.tokenizer_root),
            "task_config_path": str(sft_input.config_artifact.task_config_path),
            "stage_config_path": (
                str(sft_input.config_artifact.stage_config_path)
                if sft_input.config_artifact.stage_config_path
                else None
            ),
            "objective_name": sft_input.objective.name,
            "tokenized_sample_count": len(sft_input.tokenized_samples),
        }

    def run(self, sft_input: SFTInput, dry_run: bool = True) -> SFTOutput:
        _ = self.build_runtime_args(sft_input)
        if not dry_run:
            raise NotImplementedError(
                "OpenOneRec SFT runtime execution is not wired in Phase B. "
                "Use dry_run=True for contract-level integration checks."
            )
        checkpoint_root = Path(sft_input.model_path).parent / "openonerec_sft_ckpt"
        return SFTOutput(
            checkpoint=CheckpointArtifact(
                stage_name="sft",
                checkpoint_root=checkpoint_root,
                metadata={"entrypoint": self.entrypoint, "mode": "dry_run"},
            ),
            metrics={"status": "initialized", "pipeline": "openonerec_sft"},
        )

