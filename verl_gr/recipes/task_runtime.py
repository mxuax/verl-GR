"""Shared task runtime wiring for verl-GR recipes."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf, open_dict
from transformers import AutoConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

MODEL_TYPE_TO_TRANSFORMER_LAYER = {
    "qwen2": "Qwen2DecoderLayer",
    "qwen3": "Qwen3DecoderLayer",
    "llama": "LlamaDecoderLayer",
    "mistral": "MistralDecoderLayer",
    "gemma": "GemmaDecoderLayer",
    "gemma2": "Gemma2DecoderLayer",
}


def build_hf_tokenizer_and_processor(
    model_path: str,
    *,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    """Build HuggingFace tokenizer and processor for recipe runtimes."""

    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(model_path, trust_remote_code=trust_remote_code, use_fast=True)
    return tokenizer, processor


class RecipeTaskRuntime:
    """Runtime preparation shared by independent recipe task implementations."""

    def __init__(self) -> None:
        self._rollout_counts_expanded = False

    @staticmethod
    def _normalize_layer_wrap_value(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, set):
            normalized: list[str] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append(item)
                elif hasattr(item, "__name__"):
                    normalized.append(str(item.__name__))
                else:
                    normalized.append(str(item))
            return sorted(normalized)
        if isinstance(value, tuple):
            return list(value)
        if value is None:
            return None
        return value

    @staticmethod
    def _infer_role_strategy(role_cfg) -> str:
        """Infer backend strategy from explicit field, engine config, or target."""
        if role_cfg is None:
            return ""
        strategy = str(role_cfg.get("strategy", "") or "").lower()
        if strategy:
            return strategy

        engine_cfg = role_cfg.get("engine_config") or role_cfg.get("engine") or {}
        strategy = str(engine_cfg.get("strategy", "") or "").lower()
        if strategy:
            return strategy

        engine_target = str(engine_cfg.get("_target_", "") or "").lower()
        if "ddpengineconfig" in engine_target:
            return "ddp"
        if "fsdpengineconfig" in engine_target:
            return "fsdp"

        target = str(role_cfg.get("_target_", "") or "").lower()
        if "ddpactorconfig" in target:
            return "ddp"
        if "fsdpactorconfig" in target:
            return "fsdp"
        if "mcoreactorconfig" in target or "megatron" in target:
            return "megatron"
        return ""

    def _ensure_role_strategy(self, config, role_name: str) -> str:
        actor_rollout_ref = config.get("actor_rollout_ref")
        if actor_rollout_ref is None:
            return ""
        role_cfg = actor_rollout_ref.get(role_name)
        strategy = self._infer_role_strategy(role_cfg)
        if role_cfg is not None and strategy and role_cfg.get("strategy") is None:
            with open_dict(role_cfg):
                role_cfg.strategy = strategy
        return strategy

    def sanitize_fsdp2_wrap_policy(self, config) -> None:
        actor_rollout_ref = config.get("actor_rollout_ref")
        if actor_rollout_ref is None:
            return
        for role_name in ("actor", "ref"):
            role_cfg = actor_rollout_ref.get(role_name)
            if role_cfg is None or self._ensure_role_strategy(config, role_name) != "fsdp2":
                continue
            fsdp_cfg = role_cfg.get("fsdp_config")
            if fsdp_cfg is None:
                continue
            wrap_policy = fsdp_cfg.get("wrap_policy")
            if wrap_policy is None:
                continue
            normalized = self._normalize_layer_wrap_value(wrap_policy.get("transformer_layer_cls_to_wrap"))
            if normalized is not None:
                wrap_policy["transformer_layer_cls_to_wrap"] = normalized

    @staticmethod
    def get_reward_model_cfg(config):
        reward_root = config.get("reward")
        if reward_root is not None and reward_root.get("reward_model") is not None:
            return reward_root.reward_model
        legacy_cfg = config.get("reward_model")
        if legacy_cfg is not None:
            return legacy_cfg
        return None

    def expand_rollout_counts(self, config) -> None:
        """Recipe-specific rollout count expansion hook."""

    def configure_rollout(self, config) -> None:
        """Recipe-specific rollout registration/configuration hook."""

    def get_actor_rollout_ref_worker(self, config):
        return ActorRolloutRefWorker

    def configure_fsdp_wrap_policy(self, config, model_path: str, *, trust_remote_code: bool) -> None:
        """Align FSDP wrap target with the actual HF model family."""

        try:
            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        except Exception:
            return
        layer_cls = MODEL_TYPE_TO_TRANSFORMER_LAYER.get(str(getattr(hf_config, "model_type", "")).lower())
        if layer_cls is None:
            return

        actor_rollout_ref = config.get("actor_rollout_ref")
        if actor_rollout_ref is None:
            return
        for role_name in ("actor", "ref"):
            role_cfg = actor_rollout_ref.get(role_name)
            if role_cfg is None or self._ensure_role_strategy(config, role_name) not in {"fsdp", "fsdp2"}:
                continue
            with open_dict(role_cfg):
                fsdp_cfg = role_cfg.get("fsdp_config")
                if fsdp_cfg is None:
                    role_cfg.fsdp_config = OmegaConf.create({})
                    fsdp_cfg = role_cfg.fsdp_config
                wrap_policy = fsdp_cfg.get("wrap_policy")
                if wrap_policy is None:
                    fsdp_cfg.wrap_policy = OmegaConf.create({})
                    wrap_policy = fsdp_cfg.wrap_policy
                wrap_policy.transformer_layer_cls_to_wrap = [layer_cls]

    def prepare(self, config) -> dict[str, Any]:
        if not self._rollout_counts_expanded:
            self.expand_rollout_counts(config)
            self._rollout_counts_expanded = True
        self.configure_rollout(config)

        reward_model_cfg = self.get_reward_model_cfg(config)
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        self.configure_fsdp_wrap_policy(config, local_path, trust_remote_code=trust_remote_code)
        tokenizer, processor = build_hf_tokenizer_and_processor(
            local_path,
            trust_remote_code=trust_remote_code,
        )

        actor_strategy = self._ensure_role_strategy(config, "actor")
        if actor_strategy in {"fsdp", "fsdp2", "ddp"}:
            # Side-effect: register DDP engine with verl's EngineRegistry
            if actor_strategy == "ddp":
                import verl_gr.workers.engine.ddp  # noqa: F401
            ray_worker_group_cls = RayWorkerGroup
            actor_rollout_cls = self.get_actor_rollout_ref_worker(config)
            critic_worker = TrainingWorker
        elif actor_strategy == "megatron":
            ray_worker_group_cls = RayWorkerGroup
            actor_rollout_cls = self.get_actor_rollout_ref_worker(config)
            critic_worker = TrainingWorker
        else:
            raise NotImplementedError(f"Unknown strategy: {actor_strategy or '<missing>'}")

        return {
            "tokenizer": tokenizer,
            "processor": processor,
            "actor_rollout_cls": actor_rollout_cls,
            "critic_worker": critic_worker,
            "reward_model_cfg": reward_model_cfg,
            "ray_worker_group_cls": ray_worker_group_cls,
        }
