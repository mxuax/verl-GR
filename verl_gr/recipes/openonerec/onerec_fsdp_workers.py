"""OpenOneRec worker shim for two-stage rollout registration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker


def register_two_stage_rollout_classes() -> None:
    """Register OpenOneRec two-stage rollout in the local worker process."""
    rollout_base_mod = import_module("verl.workers.rollout.base")
    rollout_registry = getattr(rollout_base_mod, "_ROLLOUT_REGISTRY")
    rollout_registry[("two_stage", "async")] = "verl_gr.workers.rollout.two_stage_vllm_rollout.TwoStagevLLMRollout"


class OneRecActorRolloutRefWorker(ActorRolloutRefWorker):
    """Model-engine worker with local two-stage rollout registration."""

    @staticmethod
    def _normalize_wrap_targets(value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            normalized: list[str] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append(item)
                elif hasattr(item, "__name__"):
                    normalized.append(str(item.__name__))
                else:
                    normalized.append(str(item))
            return sorted(set(normalized))
        return value

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.rollout.name == "two_stage" and self.config.rollout.mode == "async":
            register_two_stage_rollout_classes()
        return super().init_model()
