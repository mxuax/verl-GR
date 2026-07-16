"""Rank-GRPO vLLM rollout adapter with per-job ZMQ weight-sync sockets."""

from __future__ import annotations

from importlib import import_module
import os

from verl_gr.workers.rollout.zmq_utils import build_zmq_handle

ServerAdapter = getattr(import_module("verl.workers.rollout.vllm_rollout.vllm_rollout"), "ServerAdapter")


class RankGRPOvLLMRollout(ServerAdapter):
    """Async server adapter for Rank-GRPO single-turn vLLM rollouts."""

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
            namespace = os.environ.get("VERL_ROLLOUT_ZMQ_NAMESPACE", "rankgrpo")
            self.zmq_handle = build_zmq_handle(
                namespace=namespace,
                replica_rank=self.replica_rank,
                local_rank=local_rank,
            )
            return
        raise RuntimeError(
            "RankGRPOvLLMRollout async adapter requires kwargs: config, model_config, device_mesh."
        )
