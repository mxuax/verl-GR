"""Async vLLM rollout server for Rank-GRPO with isolated ZMQ weight sync."""

from __future__ import annotations

import os

import ray
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


class RankGRPOvLLMHttpServer(vLLMHttpServer):
    """vLLM HTTP server using VerlGR ZMQ namespace for actor→rollout weight sync."""

    def __init__(self, *args, **kwargs):
        os.environ.setdefault("VERL_ROLLOUT_ZMQ_NAMESPACE", "rankgrpo")
        os.environ.setdefault("VERL_ZMQ_SOCKET_PREFIX", "verl-gr-rankgrpo")
        super().__init__(*args, **kwargs)

    def _get_worker_extension_cls(self) -> str:
        return "verl_gr.workers.rollout.zmq_utils.VerlGRVLLMColocateWorkerExtension"


class RankGRPOvLLMReplica(vLLMReplica):
    """vLLM replica that launches the Rank-GRPO HTTP server."""

    def __init__(
        self,
        replica_rank: int,
        config,
        model_config,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model,
            is_teacher_model,
            name_suffix,
        )
        self.server_class = ray.remote(RankGRPOvLLMHttpServer)
