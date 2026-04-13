"""OpenOneRec distill adapter owned by recipe layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verl_gr.contracts.artifact_contract import CheckpointArtifact
from verl_gr.contracts.distill_contract import DistillInput, DistillOutput


@dataclass
class OpenOneRecDistillPipeline:
    """Translate distillation contract input to OpenOneRec runtime arguments."""

    entrypoint: str = "OpenOneRec.verl_rl.recipe.distill.main_distill"

    def build_runtime_args(self, distill_input: DistillInput) -> dict[str, Any]:
        return {
            "student_model_path": str(distill_input.student_model_path),
            "teacher_model_path": str(distill_input.teacher_model_path),
            "tokenizer_root": str(distill_input.tokenizer_artifact.tokenizer_root),
            "task_config_path": str(distill_input.config_artifact.task_config_path),
            "stage_config_path": (
                str(distill_input.config_artifact.stage_config_path)
                if distill_input.config_artifact.stage_config_path
                else None
            ),
            "objective_name": distill_input.objective.name,
            "tokenized_sample_count": len(distill_input.tokenized_samples),
        }

    def run(self, distill_input: DistillInput, dry_run: bool = True) -> DistillOutput:
        _ = self.build_runtime_args(distill_input)
        if not dry_run:
            raise NotImplementedError(
                "OpenOneRec distill runtime execution is not wired in Phase B. "
                "Use dry_run=True for contract-level integration checks."
            )
        checkpoint_root = Path(distill_input.student_model_path).parent / "openonerec_distill_ckpt"
        return DistillOutput(
            checkpoint=CheckpointArtifact(
                stage_name="distill",
                checkpoint_root=checkpoint_root,
                metadata={"entrypoint": self.entrypoint, "mode": "dry_run"},
            ),
            metrics={"status": "initialized", "pipeline": "openonerec_distill"},
        )

