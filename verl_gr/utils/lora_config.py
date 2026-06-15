"""LoRA configuration helpers for verl-GR recipes.

All helpers are no-ops when ``lora_rank == 0`` and ``lora_adapter_path`` is unset,
so existing full-parameter training paths remain unchanged.
"""

from __future__ import annotations

from typing import Any


def _lora_dict(model_config: Any) -> dict[str, Any]:
    lora = getattr(model_config, "lora", None)
    if lora is None:
        return {}
    if isinstance(lora, dict):
        return lora
    if hasattr(lora, "items"):
        return dict(lora.items())
    return {}


def resolve_lora_rank(model_config: Any) -> int:
    """Return effective LoRA rank from ``model.lora.rank`` or ``model.lora_rank``."""
    rank = int(_lora_dict(model_config).get("rank", 0) or 0)
    if rank <= 0:
        rank = int(getattr(model_config, "lora_rank", 0) or 0)
    return rank


def is_lora_enabled(model_config: Any) -> bool:
    """True when LoRA training or a pre-trained adapter path is configured."""
    if getattr(model_config, "lora_adapter_path", None):
        return True
    return resolve_lora_rank(model_config) > 0


def should_merge_lora(model_config: Any) -> bool:
    """Whether to merge LoRA into base weights before export / vLLM sync."""
    return bool(_lora_dict(model_config).get("merge", False))


def normalize_lora_config(config) -> None:
    """Fill missing ``lora_rank`` from adapter metadata when adapter path is set.

    Only mutates config when ``lora_adapter_path`` is explicitly provided.
    """
    from omegaconf import open_dict

    model_cfg = config.actor_rollout_ref.model
    adapter_path = model_cfg.get("lora_adapter_path")
    if not adapter_path:
        return
    if resolve_lora_rank(model_cfg) > 0:
        return

    from verl.utils.fs import copy_to_local
    from verl.utils.model import get_lora_rank_from_adapter

    local_path = copy_to_local(adapter_path, use_shm=model_cfg.get("use_shm", False))
    inferred_rank = get_lora_rank_from_adapter(local_path)
    with open_dict(model_cfg):
        model_cfg.lora_rank = inferred_rank


def trainable_parameters(module) -> list:
    """Parameters with ``requires_grad=True`` (LoRA adapters only when PEFT is active)."""
    return [param for param in module.parameters() if param.requires_grad]
