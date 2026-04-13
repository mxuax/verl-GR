"""Local OneRec worker hooks for two-stage rollout."""

from __future__ import annotations

from verl.workers.fsdp_workers import ActorRolloutRefWorker


class OneRecActorRolloutRefWorker(ActorRolloutRefWorker):
    """Compatibility worker hook for OneRec two-stage rollout.

    Phase B keeps this as a compatibility shim and delegates to base rollout
    building unless a deeper custom rollout implementation is added.
    """

