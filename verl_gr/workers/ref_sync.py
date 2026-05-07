"""Reference model periodic weight synchronization mixin.

Provides ``sync_ref_weights()`` which extracts the full state dict from the
actor engine and loads it into the frozen ref engine so that the KL reference
tracks the current policy rather than the stale SFT checkpoint.
"""

from __future__ import annotations

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.fsdp_utils import (
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_full_state_dict,
)


class RefSyncMixin:
    """Mixin for ActorRolloutRefWorker that periodically syncs actor weights to ref."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_ref_weights(self):
        if not hasattr(self, "actor") or not hasattr(self, "ref"):
            return

        actor_module = self.actor.engine.module
        ref_module = self.ref.engine.module
        ref_fsdp_ver = fsdp_version(ref_module)

        # FSDP2: only rank 0 gathers the full state dict; fsdp2_load_full_state_dict
        #        broadcasts from rank 0 automatically.
        # FSDP1: every rank must independently produce the full state dict because
        #        load_state_dict(rank0_only=False) distributes from each local copy.
        rank0_only = ref_fsdp_ver == 2
        full_state_dict = get_fsdp_full_state_dict(actor_module, offload_to_cpu=True, rank0_only=rank0_only)

        if ref_fsdp_ver == 1:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(ref_module, StateDictType.FULL_STATE_DICT, state_dict_config):
                ref_module.load_state_dict(full_state_dict)
        elif ref_fsdp_ver == 2:
            fsdp2_load_full_state_dict(ref_module, full_state_dict)
