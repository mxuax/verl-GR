"""Bridge helpers that isolate verl imports for OpenOneRec recipes."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def get_data_proto_cls() -> Any:
    return getattr(import_module("verl"), "DataProto")


def get_torch_functional_module() -> Any:
    return import_module("verl.utils.torch_functional")


def get_compute_position_id_with_mask() -> Any:
    return getattr(import_module("verl.utils.model"), "compute_position_id_with_mask")


def get_copy_to_local() -> Any:
    return getattr(import_module("verl.utils.fs"), "copy_to_local")


def get_process_image_video() -> tuple[Any, Any]:
    module = import_module("verl.utils.dataset.vision_utils")
    return getattr(module, "process_image"), getattr(module, "process_video")


def get_qwen2_vl_rope_index() -> Any:
    return getattr(import_module("verl.models.transformers.qwen2_vl"), "get_rope_index")


def get_actor_rollout_ref_worker() -> Any:
    return getattr(import_module("verl.workers.fsdp_workers"), "ActorRolloutRefWorker")


def get_async_actor_rollout_ref_worker() -> Any:
    return getattr(import_module("verl.workers.fsdp_workers"), "AsyncActorRolloutRefWorker")


def get_critic_worker(use_legacy_worker_impl: str) -> Any:
    if use_legacy_worker_impl in {"auto", "enable"}:
        return getattr(import_module("verl.workers.fsdp_workers"), "CriticWorker")
    if use_legacy_worker_impl == "disable":
        return getattr(import_module("verl.workers.roles"), "CriticWorker")
    raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")


def get_fsdp_reward_model_worker() -> Any:
    return getattr(import_module("verl.workers.fsdp_workers"), "RewardModelWorker")


def get_megatron_reward_model_worker() -> Any:
    return getattr(import_module("verl.workers.megatron_workers"), "RewardModelWorker")


def get_fsdp_vllm_sharding_manager() -> Any:
    return getattr(import_module("verl.workers.sharding_manager.fsdp_vllm"), "FSDPVLLMShardingManager")


def get_log_gpu_memory_usage() -> Any:
    return getattr(import_module("verl.utils.profiler"), "log_gpu_memory_usage")


def get_device_name() -> Any:
    return getattr(import_module("verl.utils.device"), "get_device_name")


def get_vllm_rollout_spmd_symbols() -> tuple[Any, Any]:
    module = import_module("verl.workers.rollout.vllm_rollout.vllm_rollout_spmd")
    return getattr(module, "vLLMRollout"), getattr(module, "_pre_process_inputs")


def get_ppo_runtime_symbols() -> dict[str, Any]:
    constants_module = import_module("verl.trainer.constants_ppo")
    main_ppo_module = import_module("verl.trainer.main_ppo")
    reward_module = import_module("verl.trainer.ppo.reward")
    device_module = import_module("verl.utils.device")
    fs_module = import_module("verl.utils.fs")
    utils_module = import_module("verl.utils")
    dataset_module = import_module("verl.utils.dataset.rl_dataset")
    return {
        "get_ppo_ray_runtime_env": getattr(constants_module, "get_ppo_ray_runtime_env"),
        "create_rl_dataset": getattr(main_ppo_module, "create_rl_dataset"),
        "create_rl_sampler": getattr(main_ppo_module, "create_rl_sampler"),
        "is_cuda_available": getattr(device_module, "is_cuda_available"),
        "load_reward_manager": getattr(reward_module, "load_reward_manager"),
        "copy_to_local": getattr(fs_module, "copy_to_local"),
        "hf_tokenizer": getattr(utils_module, "hf_tokenizer"),
        "hf_processor": getattr(utils_module, "hf_processor"),
        "collate_fn": getattr(dataset_module, "collate_fn"),
    }


def get_ray_worker_group() -> Any:
    return getattr(import_module("verl.single_controller.ray"), "RayWorkerGroup")


def get_megatron_ray_worker_group() -> Any:
    return getattr(import_module("verl.single_controller.ray.megatron"), "NVMegatronRayWorkerGroup")


def get_megatron_worker_symbols() -> dict[str, Any]:
    module = import_module("verl.workers.megatron_workers")
    return {
        "ActorRolloutRefWorker": getattr(module, "ActorRolloutRefWorker"),
        "AsyncActorRolloutRefWorker": getattr(module, "AsyncActorRolloutRefWorker"),
        "CriticWorker": getattr(module, "CriticWorker"),
        "RewardModelWorker": getattr(module, "RewardModelWorker"),
    }
