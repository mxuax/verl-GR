"""OneRec custom actor worker for two-stage rollout."""

import logging
from importlib import import_module
from typing import Any

import torch
from torch.distributed.device_mesh import init_device_mesh

from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__name__)

class OneRecActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor worker that swaps in OneRec two-stage vLLM rollout."""

    @staticmethod
    def _normalize_wrap_targets(value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            normalized: list[str] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append(item)
                elif hasattr(item, "__name__"):
                    normalized.append(str(item.__name__))
                else:
                    normalized.append(str(item))
            return sorted(set(normalized))
        return value

    def _normalize_fsdp_wrap_policy(self, fsdp_config: Any) -> None:
        wrap_policy = fsdp_config.get("wrap_policy", None)
        if wrap_policy is None:
            return
        current = wrap_policy.get("transformer_layer_cls_to_wrap", None)
        normalized = self._normalize_wrap_targets(current)
        if normalized is not None:
            wrap_policy["transformer_layer_cls_to_wrap"] = normalized

    def _build_model_optimizer(self, *args, **kwargs):
        fsdp_config = kwargs.get("fsdp_config")
        if fsdp_config is None and len(args) >= 2:
            fsdp_config = args[1]
        if fsdp_config is not None:
            self._normalize_fsdp_wrap_policy(fsdp_config)
        return super()._build_model_optimizer(*args, **kwargs)

    def _build_rollout(self, trust_remote_code: bool = False):
        if self.config.rollout.name != "two_stage":
            return super()._build_rollout(trust_remote_code)

        if self.config.rollout.mode == "async":
            logger.warning("OneRec two-stage rollout currently supports sync mode only; using base async rollout.")
            return super()._build_rollout(trust_remote_code)

        TwoStagevLLMRollout = getattr(
            import_module("verl_gr.workers.rollout.two_stage_vllm_rollout"),
            "TwoStagevLLMRollout",
        )

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        if self.world_size % infer_tp != 0:
            raise ValueError(f"rollout world_size {self.world_size} not divisible by infer_tp {infer_tp}")

        device_name = get_device_name()
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])

        log_gpu_memory_usage("Before building OneRec vllm rollout", logger=logger)
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        lora_kwargs = (
            {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}}
            if self._is_lora
            else {}
        )

        rollout = TwoStagevLLMRollout(
            model_path=local_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
            device_mesh=rollout_device_mesh,
            trust_remote_code=trust_remote_code,
            **lora_kwargs,
        )
        log_gpu_memory_usage("After building OneRec vllm rollout", logger=logger)

        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

        full_params = torch.distributed.get_world_size() == 1
        rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            model_config=self.actor_model_config,
            rollout_config=self.config.rollout,
            full_params=full_params,
            device_mesh=rollout_device_mesh,
            offload_param=self._is_offload_param,
            load_format=self.config.rollout.load_format,
            layered_summon=self.config.rollout.get("layered_summon", False),
        )
        log_gpu_memory_usage("After building sharding manager", logger=logger)
        return rollout, rollout_sharding_manager

