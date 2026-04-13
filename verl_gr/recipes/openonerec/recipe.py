"""OpenOneRec task-level recipe wiring."""

from __future__ import annotations

from dataclasses import dataclass, field

from verl_gr.contracts.sample_schema import RepresentationType, TaskType
from verl_gr.contracts.task_composition import (
    StageName,
    StageRuntimeSpec,
    TaskComposition,
    TaskRuntimeComposition,
)
from verl_gr.recipes.openonerec.distill_pipeline import OpenOneRecDistillPipeline
from verl_gr.recipes.openonerec.rl_pipeline import OpenOneRecRLPipeline
from verl_gr.recipes.openonerec.sft_pipeline import OpenOneRecSFTPipeline


@dataclass
class OpenOneRecRecipe:
    """Task-level wiring for OpenOneRec stage orchestration adapters."""

    sft_pipeline: OpenOneRecSFTPipeline = field(default_factory=OpenOneRecSFTPipeline)
    distill_pipeline: OpenOneRecDistillPipeline = field(default_factory=OpenOneRecDistillPipeline)
    rl_pipeline: OpenOneRecRLPipeline = field(default_factory=OpenOneRecRLPipeline)

    @property
    def composition(self) -> TaskComposition:
        return TaskComposition(
            task_type=TaskType.OPENONEREC,
            representation_type=RepresentationType.SID,
            stages=(
                StageName.TOKENIZER,
                StageName.SFT,
                StageName.DISTILL,
                StageName.RL,
                StageName.EVAL,
            ),
        )

    @property
    def runtime_composition(self) -> TaskRuntimeComposition:
        runtime_composition = TaskRuntimeComposition(
            composition=self.composition,
            stage_runtimes=(
                StageRuntimeSpec(
                    stage=StageName.RL,
                    entrypoint_module="verl_gr.recipes.openonerec.main_onerec_ppo",
                    config_name="grpo_trainer",
                ),
            ),
        )
        runtime_composition.validate()
        return runtime_composition

    def select_pipeline(self, stage: StageName) -> object:
        if stage == StageName.SFT:
            return self.sft_pipeline
        if stage == StageName.DISTILL:
            return self.distill_pipeline
        if stage == StageName.RL:
            return self.rl_pipeline
        raise ValueError(f"Unsupported stage for OpenOneRec integration: {stage.value}")

