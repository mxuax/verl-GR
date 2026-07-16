"""Optimizer builders for verl-GR (MiniOneRec alignment)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer


class FP32MasterOptimizer(torch.optim.Optimizer):
    """Update fp32 master weights, then write them back to visible parameters.

    MiniOneRec's original DeepSpeed bf16 path keeps partitioned fp32 master
    weights even though the visible model parameters are bf16. DDP has no ZeRO
    partitioning, so this wrapper keeps full fp32 masters locally and delegates
    the actual Adam update to the wrapped optimizer.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], inner_cls, inner_kwargs: dict):
        visible_params = [param for param in params if param.requires_grad]
        if not visible_params:
            raise ValueError("FP32MasterOptimizer received no trainable parameters")

        master_params = [
            torch.nn.Parameter(param.detach().float().clone(), requires_grad=True)
            for param in visible_params
        ]
        super().__init__(master_params, defaults={})
        self.visible_params = visible_params
        self.master_params = master_params
        self.inner_optimizer = inner_cls(master_params, **inner_kwargs)
        self.param_groups = self.inner_optimizer.param_groups
        self.state = self.inner_optimizer.state
        self.defaults = self.inner_optimizer.defaults

    def master_param_for_visible(self, visible_param: torch.nn.Parameter):
        for visible, master in zip(self.visible_params, self.master_params, strict=True):
            if visible is visible_param:
                return master
        return None

    @torch.no_grad()
    def _copy_visible_grads_to_master(self):
        for visible, master in zip(self.visible_params, self.master_params, strict=True):
            if visible.grad is None:
                master.grad = None
                continue
            if master.grad is None:
                master.grad = torch.empty_like(master.data)
            master.grad.copy_(visible.grad.detach().float())

    @torch.no_grad()
    def _copy_master_to_visible(self):
        for visible, master in zip(self.visible_params, self.master_params, strict=True):
            visible.data.copy_(master.data.to(dtype=visible.dtype))

    def step(self, closure=None):
        self._copy_visible_grads_to_master()
        result = self.inner_optimizer.step(closure=closure)
        self._copy_master_to_visible()
        return result

    def zero_grad(self, set_to_none: bool = True):
        self.inner_optimizer.zero_grad(set_to_none=set_to_none)
        for visible in self.visible_params:
            if visible.grad is None:
                continue
            if set_to_none:
                visible.grad = None
            else:
                visible.grad.zero_()

    def state_dict(self):
        state = self.inner_optimizer.state_dict()
        state["_fp32_master_params"] = [param.detach().cpu() for param in self.master_params]
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        master_state = state_dict.pop("_fp32_master_params", None)
        result = self.inner_optimizer.load_state_dict(state_dict)
        if master_state is not None:
            for master, saved in zip(self.master_params, master_state, strict=True):
                master.data.copy_(saved.to(device=master.device, dtype=master.dtype))
            self._copy_master_to_visible()
        self.param_groups = self.inner_optimizer.param_groups
        self.state = self.inner_optimizer.state
        self.defaults = self.inner_optimizer.defaults
        return result


def build_actor_optimizer(parameters, config: FSDPOptimizerConfig):
    """Build optimizer; ``paged_adamw_32bit`` matches MiniOneRec/TRL + bitsandbytes."""
    optim_name = str(getattr(config, "optimizer", "AdamW")).lower()
    override_kwargs = dict(config.override_optimizer_config or {})
    use_fp32_master = bool(override_kwargs.pop("use_fp32_master", False))

    if optim_name in {"adamw_torch_fp32_master", "adamw_fp32_master"}:
        kwargs = {
            "lr": config.lr,
            "betas": config.betas,
            "eps": override_kwargs.pop("eps", getattr(config, "eps", 1e-8)),
            "weight_decay": config.weight_decay,
        }
        kwargs.update(override_kwargs)
        return FP32MasterOptimizer(parameters, torch.optim.AdamW, kwargs)

    if optim_name in {"adamw", "adamw_torch"} and use_fp32_master:
        kwargs = {
            "lr": config.lr,
            "betas": config.betas,
            "eps": override_kwargs.pop("eps", getattr(config, "eps", 1e-8)),
            "weight_decay": config.weight_decay,
        }
        kwargs.update(override_kwargs)
        return FP32MasterOptimizer(parameters, torch.optim.AdamW, kwargs)

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
        kwargs.update(override_kwargs)
        if use_fp32_master:
            return FP32MasterOptimizer(parameters, BnbAdamW, kwargs)
        return BnbAdamW(parameters, **kwargs)

    return build_optimizer(parameters, config)
