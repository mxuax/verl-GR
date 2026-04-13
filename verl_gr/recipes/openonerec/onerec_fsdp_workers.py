"""Local OneRec worker hooks for two-stage rollout."""

from __future__ import annotations

from importlib import import_module
from typing import Any

try:
    ActorRolloutRefWorker = getattr(import_module("verl.workers.fsdp_workers"), "ActorRolloutRefWorker")
except Exception:  # pragma: no cover - fallback for environments without verl deps
    ActorRolloutRefWorker = object


class OneRecActorRolloutRefWorker(ActorRolloutRefWorker):
    """Compatibility worker hook for OneRec two-stage rollout.

    Phase B keeps this as a compatibility shim and delegates to base rollout
    building unless a deeper custom rollout implementation is added.
    """

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

    def _normalize_fsdp_wrap_policy(self, fsdp_config: Any) -> None:
        wrap_policy = fsdp_config.get("wrap_policy", None)
        if wrap_policy is None:
            return
        current = wrap_policy.get("transformer_layer_cls_to_wrap", None)
        normalized = self._normalize_wrap_targets(current)
        if normalized is None:
            return
        wrap_policy["transformer_layer_cls_to_wrap"] = normalized

    def _build_model_optimizer(self, *args, **kwargs):
        fsdp_config = kwargs.get("fsdp_config")
        if fsdp_config is None and len(args) >= 2:
            fsdp_config = args[1]
        if fsdp_config is not None:
            self._normalize_fsdp_wrap_policy(fsdp_config)
        return super()._build_model_optimizer(*args, **kwargs)

