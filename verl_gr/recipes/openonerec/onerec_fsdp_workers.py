"""OneRec custom actor worker for two-stage rollout."""

import logging
from importlib import import_module
from typing import Any

import torch
from verl_gr.integrations.verl.openonerec_bridge import (
    get_actor_rollout_ref_worker,
    get_copy_to_local,
    get_device_name,
    get_fsdp_vllm_sharding_manager,
    get_log_gpu_memory_usage,
)

logger = logging.getLogger(__name__)

try:
    ActorRolloutRefWorker = get_actor_rollout_ref_worker()
except Exception:  # pragma: no cover - fallback when verl isn't installed
    ActorRolloutRefWorker = object


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

        FSDPVLLMShardingManager = get_fsdp_vllm_sharding_manager()
        log_gpu_memory_usage = get_log_gpu_memory_usage()
        copy_to_local = get_copy_to_local()
        init_device_mesh = getattr(import_module("torch.distributed.device_mesh"), "init_device_mesh")
        device_name_fn = get_device_name()
        OneRecvLLMRollout = getattr(
            import_module("verl_gr.components.rollout.onerec_vllm_rollout"),
            "OneRecvLLMRollout",
        )

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        if self.world_size % infer_tp != 0:
            raise ValueError(f"rollout world_size {self.world_size} not divisible by infer_tp {infer_tp}")

        device_name = device_name_fn()
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])

        log_gpu_memory_usage("Before building OneRec vllm rollout", logger=logger)
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        lora_kwargs = (
            {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}}
            if self._is_lora
            else {}
        )

        rollout = OneRecvLLMRollout(
            model_path=local_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
            device_mesh=rollout_device_mesh,
            trust_remote_code=trust_remote_code,
            **lora_kwargs,
        )
        log_gpu_memory_usage("After building OneRec vllm rollout", logger=logger)

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

