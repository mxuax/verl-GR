"""Registry for task recipes in verl-GR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from verl_gr.contracts.task_composition import TaskComposition


@dataclass(frozen=True)
class RecipeSpec:
    """Metadata describing a recipe entrypoint."""

    name: str
    composition: TaskComposition
    factory: Callable[[], object]


class RecipeRegistry:
    """Simple in-memory registry for recipe specifications."""

    def __init__(self) -> None:
        self._recipes: dict[str, RecipeSpec] = {}

    def register(self, spec: RecipeSpec) -> None:
        """Register a recipe specification."""

        spec.composition.validate()
        if spec.name in self._recipes:
            raise ValueError(f"Recipe '{spec.name}' is already registered.")
        self._recipes[spec.name] = spec

    def get(self, name: str) -> RecipeSpec:
        """Return a recipe specification by name."""

        try:
            return self._recipes[name]
        except KeyError as exc:
            raise KeyError(f"Recipe '{name}' is not registered.") from exc

    def list_names(self) -> tuple[str, ...]:
        """List all registered recipe names."""

        return tuple(sorted(self._recipes))

