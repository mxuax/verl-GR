"""MiniOneRec worker shim for constrained-beam rollout registration."""

from __future__ import annotations

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl_gr.workers.ref_sync import RefSyncMixin
from verl_gr.workers.rollout.registration import register_constrained_beam_rollout_class


class MiniOneRecActorRolloutRefWorker(RefSyncMixin, ActorRolloutRefWorker):
    """Model-engine worker with local constrained-beam rollout registration."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.rollout.name == "constrained_beam" and self.config.rollout.mode == "async":
            register_constrained_beam_rollout_class()
        return super().init_model()
