"""Single-stage constrained-beam vLLM rollout adapter."""

from __future__ import annotations

from importlib import import_module
import os

import ray
from verl_gr.workers.rollout.zmq_utils import build_zmq_handle

ServerAdapter = getattr(import_module("verl.workers.rollout.vllm_rollout.vllm_rollout"), "ServerAdapter")


class ConstrainedBeamvLLMRollout(ServerAdapter):
    """Async server adapter for MiniOneRec-style constrained beam generation."""

    def __init__(self, *args, **kwargs):
        if {"config", "model_config", "device_mesh"}.issubset(kwargs):
            super().__init__(
                config=kwargs["config"],
                model_config=kwargs["model_config"],
                device_mesh=kwargs["device_mesh"],
                replica_rank=kwargs.get("replica_rank", -1),
            )
            local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
            local_rank = self.rollout_rank % local_world_size
            namespace = os.environ.get("VERL_ROLLOUT_ZMQ_NAMESPACE", "constrained-beam")
            self.zmq_handle = build_zmq_handle(
                namespace=namespace,
                replica_rank=self.replica_rank,
                local_rank=local_rank,
            )
            return
        raise RuntimeError(
            "ConstrainedBeamvLLMRollout async adapter requires kwargs: config, model_config, device_mesh."
        )

    async def resume(self, tags: list[str]):
        await super().resume(tags=tags)

    async def update_weights(self, weights, global_steps: int = None, **kwargs):
        await self._execute_server_method("abort_all_requests", reset_prefix_cache=True)
        try:
            await super().update_weights(weights=weights, global_steps=global_steps, **kwargs)
        finally:
            await self._execute_server_method("resume_generation")

    async def release(self):
        await super().release()

    async def _execute_server_method(self, method: str, **kwargs):
        if self.rollout_rank != 0:
            return None
        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            self.server_handle = ray.get_actor(f"{prefix}server_{self.replica_rank}_{self.node_rank}")
        return await getattr(self.server_handle, method).remote(**kwargs)
