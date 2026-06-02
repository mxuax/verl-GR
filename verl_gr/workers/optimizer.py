"""Optimizer builders for verl-GR (MiniOneRec alignment)."""

from __future__ import annotations

from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer


def build_actor_optimizer(parameters, config: FSDPOptimizerConfig):
    """Build optimizer; ``paged_adamw_32bit`` matches MiniOneRec/TRL + bitsandbytes."""
    optim_name = str(getattr(config, "optimizer", "AdamW")).lower()

    if optim_name in {"paged_adamw_32bit", "pagedadamw32bit"}:
        try:
            from bitsandbytes.optim import AdamW as BnbAdamW
        except ImportError as exc:
            raise ImportError(
                "optimizer='paged_adamw_32bit' requires bitsandbytes. "
                "Install with: pip install bitsandbytes"
            ) from exc

        kwargs = {
            "lr": config.lr,
            "betas": config.betas,
            "weight_decay": config.weight_decay,
            "optim_bits": 32,
            "is_paged": True,
        }
        if config.override_optimizer_config:
            kwargs.update(config.override_optimizer_config)
        return BnbAdamW(parameters, **kwargs)

    return build_optimizer(parameters, config)
