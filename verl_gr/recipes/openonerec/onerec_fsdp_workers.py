"""Legacy compatibility shim for old OpenOneRec worker import paths.

The current `verl` model-engine path uses `verl.workers.engine_workers.ActorRolloutRefWorker`
directly; OpenOneRec two-stage routing is now injected through rollout/agent-loop registries.
"""

from typing import Any

from verl.workers.engine_workers import ActorRolloutRefWorker


class OneRecActorRolloutRefWorker(ActorRolloutRefWorker):
    """Backward-compatible alias kept for old imports inside `verl_gr`."""

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
