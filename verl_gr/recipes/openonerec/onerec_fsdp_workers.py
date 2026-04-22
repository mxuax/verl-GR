"""OneRec custom actor worker for two-stage rollout."""

import logging
from importlib import import_module
from typing import Any

from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from verl.utils.device import get_device_name
from verl.utils.profiler import log_gpu_memory_usage
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

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

        rollout_module = import_module("verl_gr.workers.rollout.two_stage_vllm_rollout")
        TwoStagevLLMRollout = getattr(rollout_module, "TwoStagevLLMRollout")

        logger.warning("Two-stage rollout selected: %s", TwoStagevLLMRollout.__name__)

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        if self.world_size % infer_tp != 0:
            raise ValueError(f"rollout world_size {self.world_size} not divisible by infer_tp {infer_tp}")

        device_name = get_device_name()
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])

        log_gpu_memory_usage("Before building OneRec vllm rollout", logger=logger)
        rollout_cfg_node = OmegaConf.create(OmegaConf.to_container(self.config.rollout, resolve=True))
        rollout_cfg: RolloutConfig = omega_conf_to_dataclass(rollout_cfg_node)
        model_cfg: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        self.model_config = model_cfg
        rollout = TwoStagevLLMRollout(
            config=rollout_cfg,
            model_config=model_cfg,
            device_mesh=rollout_device_mesh,
        )
        self.rollout = rollout
        self.rollout_device_mesh = rollout_device_mesh
        log_gpu_memory_usage("After building OneRec vllm rollout (async adapter)", logger=logger)

        # Keep parity with upstream ActorRolloutRefWorker._build_rollout(), where
        # rollout_mode()/update_weights rely on these lifecycle flags.
        self.base_sync_done = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # Upstream also stores model_config during rollout construction.
        self.model_config = self.actor_model_config
        return self.rollout

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        _ = global_steps
        await self.rollout_mode()
        return True


class OneRecAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Async worker that ensures two-stage async rollout is registry-visible."""

    def _build_rollout(self, trust_remote_code: bool = False):
        if self.config.rollout.name == "two_stage":
            from verl.workers.rollout import base as rollout_base

            rollout_base._ROLLOUT_REGISTRY[("two_stage", "async")] = (
                "verl.workers.rollout.vllm_rollout.vllm_rollout.ServerAdapter"
            )
        return super()._build_rollout(trust_remote_code=trust_remote_code)
