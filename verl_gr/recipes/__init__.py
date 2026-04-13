"""Recipes package for task-level wiring."""

from verl_gr.recipes.openonerec import (
    OpenOneRecDistillPipeline,
    OpenOneRecGRPORuntime,
    OpenOneRecRecipe,
    OpenOneRecRLPipeline,
    OpenOneRecSFTPipeline,
)
from verl_gr.recipes.recipe_registry import RecipeRegistry

__all__ = [
    "OpenOneRecRecipe",
    "OpenOneRecSFTPipeline",
    "OpenOneRecDistillPipeline",
    "OpenOneRecRLPipeline",
    "OpenOneRecGRPORuntime",
    "RecipeRegistry",
]

