"""Reference model periodic weight synchronization mixin.

Provides ``sync_ref_weights()`` which synchronizes the actor weights into
the frozen ref engine so that the KL reference tracks the current policy.

When ``mixup_alpha > 0``, the update follows the TRL ``SyncRefModelCallback``
EMA convention::

    ref = (1 - alpha) * ref + alpha * actor

where ``alpha = ref_model_mixup_alpha`` (default 0.6 in TRL → ref gets 60% actor).

* alpha = 0.0 → ref frozen
* alpha = 0.6 → soft tracking (TRL default)
* alpha = 1.0 → ref = actor (hard copy)
"""

from __future__ import annotations

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.fsdp_utils import (
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_full_state_dict,
)


def _ema_update(ref_module, actor_module, alpha: float):
    """In-place EMA matching TRL ``SyncRefModelCallback._sync_target_model``.

    TRL formula:  ref = (1 - alpha) * ref + alpha * actor

    * alpha = 0.0  → ref unchanged (frozen)
    * alpha = 0.6  → ref = 0.4·ref + 0.6·actor  (soft tracking, TRL default)
    * alpha = 1.0  → ref = actor  (hard copy)
    """
    # Unwrap DDP / FSDP to access raw parameters
    ref_params = (
        ref_module.module.parameters() if isinstance(ref_module, (DDP, FSDP))
        else ref_module.parameters()
    )
    actor_params = (
        actor_module.module.parameters() if isinstance(actor_module, (DDP, FSDP))
        else actor_module.parameters()
    )
    with torch.no_grad():
        for rp, ap in zip(ref_params, actor_params):
            rp.data.mul_(1.0 - alpha).add_(ap.data, alpha=alpha)


class RefSyncMixin:
    """Mixin for ActorRolloutRefWorker that periodically syncs actor weights to ref."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_ref_weights(self, mixup_alpha: float = 0.0):
        if not hasattr(self, "actor") or not hasattr(self, "ref"):
            return

        actor_module = self.actor.engine.module
        ref_module = self.ref.engine.module

        # DDP: full parameters on every rank — EMA or direct copy.
        if isinstance(actor_module, DDP) and isinstance(ref_module, DDP):
            if mixup_alpha > 0:
                _ema_update(ref_module, actor_module, mixup_alpha)
            else:
                ref_module.module.load_state_dict(actor_module.module.state_dict())
            return

        ref_fsdp_ver = fsdp_version(ref_module)

        # FSDP with EMA: gather both state dicts, mix on CPU, load back.
        # TRL formula: ref = (1 - alpha) * ref + alpha * actor
        if mixup_alpha > 0:
            rank0_only = ref_fsdp_ver == 2
            actor_sd = get_fsdp_full_state_dict(actor_module, offload_to_cpu=True, rank0_only=rank0_only)
            ref_sd = get_fsdp_full_state_dict(ref_module, offload_to_cpu=True, rank0_only=rank0_only)
            for key in actor_sd:
                ref_sd[key] = (1.0 - mixup_alpha) * ref_sd[key] + mixup_alpha * actor_sd[key]
            if ref_fsdp_ver == 1:
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                with FSDP.state_dict_type(ref_module, StateDictType.FULL_STATE_DICT, state_dict_config):
                    ref_module.load_state_dict(ref_sd)
            elif ref_fsdp_ver == 2:
                fsdp2_load_full_state_dict(ref_module, ref_sd)
            return

        # FSDP hard copy (original path, mixup_alpha == 0)
        rank0_only = ref_fsdp_ver == 2
        full_state_dict = get_fsdp_full_state_dict(actor_module, offload_to_cpu=True, rank0_only=rank0_only)

        if ref_fsdp_ver == 1:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(ref_module, StateDictType.FULL_STATE_DICT, state_dict_config):
                ref_module.load_state_dict(full_state_dict)
        elif ref_fsdp_ver == 2:
            fsdp2_load_full_state_dict(ref_module, full_state_dict)
