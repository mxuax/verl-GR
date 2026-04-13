"""OpenOneRec recipe wiring for verl-GR."""

from __future__ import annotations

from dataclasses import dataclass, field

from verl_gr.contracts.sample_schema import RepresentationType, TaskType
from verl_gr.contracts.task_composition import StageName, TaskComposition
from verl_gr.integrations.openonerec.distill_pipeline import OpenOneRecDistillPipeline
from verl_gr.integrations.openonerec.rl_pipeline import OpenOneRecRLPipeline
from verl_gr.integrations.openonerec.sft_pipeline import OpenOneRecSFTPipeline
from verl_gr.recipes.recipe_registry import RecipeSpec


@dataclass
class OpenOneRecRecipe:
    """Task-level wiring for OpenOneRec stage orchestration adapters."""

    sft_pipeline: OpenOneRecSFTPipeline = field(default_factory=OpenOneRecSFTPipeline)
    distill_pipeline: OpenOneRecDistillPipeline = field(default_factory=OpenOneRecDistillPipeline)
    rl_pipeline: OpenOneRecRLPipeline = field(default_factory=OpenOneRecRLPipeline)

    @property
    def composition(self) -> TaskComposition:
        """Return canonical OpenOneRec stage composition."""

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

    def select_pipeline(self, stage: StageName) -> object:
        """Return the integration adapter for one stage."""

        if stage == StageName.SFT:
            return self.sft_pipeline
        if stage == StageName.DISTILL:
            return self.distill_pipeline
        if stage == StageName.RL:
            return self.rl_pipeline
        raise ValueError(f"Unsupported stage for OpenOneRec integration: {stage.value}")


def create_openonerec_recipe_spec() -> RecipeSpec:
    """Create registry spec for the OpenOneRec recipe."""

    recipe = OpenOneRecRecipe()
    return RecipeSpec(
        name="openonerec",
        composition=recipe.composition,
        factory=OpenOneRecRecipe,
    )

