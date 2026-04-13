"""Recipes package for task-level wiring."""

from verl_gr.recipes.openonerec_recipe import OpenOneRecRecipe, create_openonerec_recipe_spec
from verl_gr.recipes.recipe_registry import RecipeRegistry, build_default_registry, register_builtin_recipes

__all__ = [
    "OpenOneRecRecipe",
    "RecipeRegistry",
    "build_default_registry",
    "create_openonerec_recipe_spec",
    "register_builtin_recipes",
]

