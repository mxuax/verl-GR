"""Task-aware PPO entrypoint for verl-gr recipes."""

import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Callable

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import (
    TaskRunner as BaseTaskRunner,
    auto_set_device,
    create_rl_dataset,
    create_rl_sampler,
    migrate_legacy_reward_impl,
    run_ppo as base_run_ppo,
)
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.dataset.rl_dataset import collate_fn
from verl_gr.recipes.minionerec.minionerec_recipe import MiniOneRecTask
from verl_gr.recipes.openonerec.onerec_recipe import OneRecTask
from verl_gr.recipes.rankgrpo.rankgrpo_task import RankGRPOTask
from verl_gr.trainers.rl_trainer import RLTrainer

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "verl_gr"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    factory: Callable[[], object]


TASK_REGISTRY = {
    "openonerec": TaskSpec(name="openonerec", factory=OneRecTask),
    "minionerec": TaskSpec(name="minionerec", factory=MiniOneRecTask),
    "rankgrpo": TaskSpec(name="rankgrpo", factory=RankGRPOTask),
}


def _infer_legacy_task_name(config) -> str:
    task_cfg = config.get("task", {})
    task_class_path = str(task_cfg.get("class_path", "")).lower()
    if "minionerec" in task_class_path:
        return "minionerec"
    if "rankgrpo" in task_class_path:
        return "rankgrpo"
    if "openonerec" in task_class_path:
        return "openonerec"

    rollout_name = str(config.actor_rollout_ref.rollout.get("name", "")).lower()
    if rollout_name == "constrained_beam":
        return "minionerec"
    if rollout_name == "two_stage":
        return "openonerec"

    custom_cls_name = config.data.get("custom_cls", {}).get("name", "")
    custom_cls_path = config.data.get("custom_cls", {}).get("path", "")
    reward_path = config.get("custom_reward_function", {}).get("path", "") or config.get("reward", {}).get(
        "custom_reward_function", {}
    ).get("path", "")
    if (
        "minionerec" in str(custom_cls_name).lower()
        or "minionerec" in str(custom_cls_path).lower()
        or "minionerec" in str(reward_path).lower()
    ):
        return "minionerec"
    if (
        custom_cls_name == "RankGRPODataset"
        or "rankgrpo" in str(custom_cls_path).lower()
        or "rankgrpo" in str(reward_path).lower()
        or config.algorithm.get("rank_grpo", {}).get("enable", False)
    ):
        return "rankgrpo"
    return "openonerec"


def _select_task(config):
    task_cfg = config.get("task", {})
    task_class_path = str(task_cfg.get("class_path", "")).lower()
    if "minionerec" in task_class_path:
        task_name = "minionerec"
    elif "rankgrpo" in task_class_path:
        task_name = "rankgrpo"
    elif "openonerec" in task_class_path:
        task_name = "openonerec"
    else:
        task_name = str(task_cfg.get("name", "") or _infer_legacy_task_name(config)).lower()
    try:
        return TASK_REGISTRY[task_name].factory()
    except KeyError as exc:
        valid = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(f"Unknown verl-gr task '{task_name}'. Expected one of: {valid}") from exc


def _inject_legacy_reward_placeholders(config) -> None:
    """Provide root-level legacy reward keys expected by verl's migration hook."""
    placeholders = (
        (
            "reward_model",
            {
                "num_workers": None,
                "reward_manager": None,
                "enable": None,
                "enable_resource_pool": None,
                "n_gpus_per_node": None,
                "nnodes": None,
                "reward_loop_source": None,
                "reward_loop_module_path": None,
                "reward_loop_class_name": None,
                "reward_kwargs": None,
                "model": {
                    "path": None,
                    "external_lib": None,
                    "trust_remote_code": None,
                },
                "rollout": {
                    "name": None,
                    "dtype": None,
                    "gpu_memory_utilization": None,
                    "enforce_eager": None,
                    "cudagraph_capture_sizes": None,
                    "free_cache_engine": None,
                    "data_parallel_size": None,
                    "expert_parallel_size": None,
                    "tensor_model_parallel_size": None,
                    "max_num_batched_tokens": None,
                    "max_model_len": None,
                    "max_num_seqs": None,
                    "load_format": None,
                    "engine_kwargs": None,
                    "limit_images": None,
                    "enable_chunked_prefill": None,
                    "enable_prefix_caching": None,
                    "disable_log_stats": None,
                    "skip_tokenizer_init": None,
                    "prompt_length": None,
                    "response_length": None,
                },
            },
        ),
        ("custom_reward_function", {"path": None, "name": None}),
        ("sandbox_fusion", {"url": None, "max_concurrent": None, "memory_limit_mb": None}),
    )
    for key, value in placeholders:
        if OmegaConf.select(config, key) is None:
            OmegaConf.update(config, key, value, force_add=True)


def _cfg_get(node, key: str, default=None):
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _cfg_keys(node) -> list[str]:
    if node is None:
        return []
    if hasattr(node, "keys"):
        try:
            return sorted(str(key) for key in node.keys())
        except Exception:
            return []
    return []


def _strategy_debug_snapshot(config, role_name: str) -> dict:
    actor_rollout_ref = _cfg_get(config, "actor_rollout_ref")
    role_cfg = _cfg_get(actor_rollout_ref, role_name)
    engine_cfg = _cfg_get(role_cfg, "engine_config") or _cfg_get(role_cfg, "engine")
    return {
        "role": role_name,
        "role_path": f"actor_rollout_ref.{role_name}",
        "role_target": _cfg_get(role_cfg, "_target_"),
        "role_strategy": _cfg_get(role_cfg, "strategy"),
        "engine_target": _cfg_get(engine_cfg, "_target_"),
        "engine_strategy": _cfg_get(engine_cfg, "strategy"),
        "role_keys": _cfg_keys(role_cfg),
        "engine_keys": _cfg_keys(engine_cfg),
    }


def _model_debug_snapshot(config) -> dict:
    actor_rollout_ref = _cfg_get(config, "actor_rollout_ref")
    model_cfg = _cfg_get(actor_rollout_ref, "model")
    return {
        "path": "actor_rollout_ref.model",
        "model_target": _cfg_get(model_cfg, "_target_"),
        "model_path": _cfg_get(model_cfg, "path"),
        "tokenizer_path": _cfg_get(model_cfg, "tokenizer_path"),
        "use_remove_padding": _cfg_get(model_cfg, "use_remove_padding"),
        "model_keys": _cfg_keys(model_cfg),
    }


def _validate_strategy_signals(config, task_impl, role_name: str, stage: str) -> str:
    strategy = task_impl._ensure_role_strategy(config, role_name)
    snapshot = _strategy_debug_snapshot(config, role_name)
    print(f"[verl-gr] strategy-signals[{stage}] {snapshot}")
    if not strategy:
        raise ValueError(
            f"Missing backend strategy for {snapshot['role_path']} at stage '{stage}'. "
            f"Signals={snapshot}"
        )
    return strategy


def _validate_model_signal(config, stage: str) -> None:
    snapshot = _model_debug_snapshot(config)
    print(f"[verl-gr] model-signals[{stage}] {snapshot}")
    if not snapshot["model_target"]:
        raise ValueError(f"Missing model _target_ at stage '{stage}'. Signals={snapshot}")


def _ensure_runtime_root_blocks(config) -> None:
    """Backfill root runtime blocks that base verl dereferences before TaskRunner."""
    def merge_missing(base_path: str, value) -> None:
        if isinstance(value, dict):
            if base_path and OmegaConf.select(config, base_path) is None:
                OmegaConf.update(config, base_path, {}, force_add=True)
            for sub_key, sub_value in value.items():
                sub_path = f"{base_path}.{sub_key}" if base_path else sub_key
                merge_missing(sub_path, sub_value)
            return
        if OmegaConf.select(config, base_path) is None:
            OmegaConf.update(config, base_path, value, force_add=True)

    placeholders = (
        (
            "transfer_queue",
            {
                "enable": False,
                "backend": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {
                        "total_storage_size": 100000,
                        "num_data_storage_units": 8,
                    },
                },
            },
        ),
        (
            "ray_kwargs",
            {
                "ray_init": {"num_cpus": None},
                "timeline_json_file": None,
            },
        ),
        (
            "global_profiler",
            {
                "_target_": "verl.utils.profiler.ProfilerConfig",
                "tool": None,
                "steps": None,
                "profile_continuous_steps": False,
                "save_path": "outputs/profile",
                "global_tool_config": {
                    "nsys": {
                        "_target_": "verl.utils.profiler.config.NsightToolConfig",
                        "discrete": False,
                        "controller_nsight_options": {
                            "trace": "cuda,nvtx,cublas,ucx",
                            "cuda-memory-usage": "true",
                            "cuda-graph-trace": "graph",
                        },
                        "worker_nsight_options": {
                            "trace": "cuda,nvtx,cublas,ucx",
                            "cuda-memory-usage": "true",
                            "cuda-graph-trace": "graph",
                            "capture-range": "cudaProfilerApi",
                            "capture-range-end": None,
                            "kill": "none",
                        },
                    },
                    "torch_memory": {
                        "trace_alloc_max_entries": 100000,
                        "stack_depth": 32,
                        "context": "all",
                        "stacks": "all",
                        "kw_args": {},
                    },
                    "precision_debugger": {
                        "_target_": "verl.utils.profiler.config.PrecisionDebuggerToolConfig",
                        "config_path": None,
                        "steps": None,
                        "stages": None,
                        "strict": False,
                    },
                },
            },
        ),
        (
            "actor_rollout_ref",
            {
                "hybrid_engine": True,
                "nccl_timeout": 600,
                "model": {
                    "_target_": "verl.workers.config.HFModelConfig",
                    "path": None,
                    "hf_config_path": None,
                    "tokenizer_path": None,
                    "use_shm": False,
                    "trust_remote_code": False,
                    "custom_chat_template": None,
                    "external_lib": None,
                    "override_config": {},
                    "enable_gradient_checkpointing": True,
                    "enable_activation_offload": False,
                    "use_remove_padding": True,
                    "lora_rank": 0,
                    "lora_alpha": 16,
                    "target_modules": "all-linear",
                    "exclude_modules": None,
                    "lora": {},
                    "lora_adapter_path": None,
                    "use_liger": False,
                    "use_fused_kernels": False,
                    "fused_kernel_options": {"impl_backend": "torch"},
                },
                "rollout": {
                    "_target_": "verl.workers.config.RolloutConfig",
                    "name": None,
                    "mode": "async",
                    "nnodes": 0,
                    "n_gpus_per_node": "${oc.select:trainer.n_gpus_per_node,8}",
                    "temperature": 1.0,
                    "top_k": -1,
                    "top_p": 1.0,
                    "prompt_length": "${oc.select:data.max_prompt_length,512}",
                    "response_length": "${oc.select:data.max_response_length,512}",
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.5,
                    "ignore_eos": False,
                    "enforce_eager": False,
                    "cudagraph_capture_sizes": None,
                    "free_cache_engine": True,
                    "tensor_model_parallel_size": 2,
                    "data_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "max_num_batched_tokens": 8192,
                    "max_model_len": None,
                    "max_num_seqs": 1024,
                    "enable_chunked_prefill": True,
                    "enable_prefix_caching": True,
                    "logprobs_mode": "processed_logprobs",
                    "scheduling_policy": "fcfs",
                    "load_format": "dummy",
                    "layered_summon": False,
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": None,
                    "log_prob_use_dynamic_bsz": False,
                    "log_prob_max_token_len_per_gpu": 16384,
                    "disable_log_stats": True,
                    "do_sample": True,
                    "n": 1,
                    "over_sample_rate": 0.0,
                    "engine_kwargs": {
                        "vllm": {},
                        "sglang": {},
                        "trtllm": {},
                    },
                    "val_kwargs": {
                        "_target_": "verl.workers.config.SamplingConfig",
                        "top_k": -1,
                        "top_p": 1.0,
                        "temperature": 0,
                        "n": 1,
                        "do_sample": False,
                    },
                    "multi_turn": {
                        "_target_": "verl.workers.config.MultiTurnConfig",
                        "enable": False,
                        "max_assistant_turns": None,
                        "tool_config_path": None,
                        "max_user_turns": None,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 256,
                        "tool_response_truncate_side": "middle",
                        "use_inference_chat_template": False,
                        "tokenization_sanity_check_mode": "strict",
                        "format": "hermes",
                        "num_repeat_rollouts": None,
                    },
                    "calculate_log_probs": False,
                    "agent": {
                        "_target_": "verl.workers.config.AgentLoopConfig",
                        "num_workers": 1,
                    },
                    "custom": {},
                    "checkpoint_manager_class": None,
                    "checkpoint_engine": {
                        "_target_": "verl.workers.config.CheckpointEngineConfig",
                        "backend": "naive",
                        "wait_timeout": 300.0,
                        "retry_max_attempts": 10,
                        "retry_backoff_factor": 2.0,
                        "engine_kwargs": {},
                        "custom_backend_module": None,
                    },
                    "skip": {
                        "_target_": "verl.workers.config.SkipConfig",
                        "enable": False,
                        "action": "dump",
                        "dump_dir": "rollout_skip_dump",
                        "load_dir": None,
                        "max_dump_step": None,
                    },
                },
            },
        ),
        (
            "data",
            {
                "tokenizer": None,
                "use_shm": False,
                "train_files": None,
                "val_files": None,
                "train_max_samples": -1,
                "val_max_samples": -1,
                "prompt_key": "prompt",
                "reward_fn_key": "data_source",
                "max_prompt_length": 512,
                "max_response_length": 512,
                "train_batch_size": 1024,
                "val_batch_size": None,
                "tool_config_path": "${oc.select:actor_rollout_ref.rollout.multi_turn.tool_config_path, null}",
                "return_raw_input_ids": False,
                "return_raw_chat": True,
                "return_full_prompt": False,
                "shuffle": True,
                "seed": None,
                "dataloader_num_workers": 8,
                "image_patch_size": 14,
                "validation_shuffle": False,
                "filter_overlong_prompts": False,
                "filter_overlong_prompts_workers": 1,
                "truncation": "error",
                "image_key": "images",
                "video_key": "videos",
                "trust_remote_code": False,
                "custom_cls": {
                    "path": None,
                    "name": None,
                },
                "return_multi_modal_inputs": True,
                "sampler": {
                    "class_path": None,
                    "class_name": None,
                },
                "datagen": {
                    "path": None,
                    "name": None,
                },
                "apply_chat_template_kwargs": {},
            },
        ),
        (
            "algorithm",
            {
                "_target_": "verl.trainer.config.AlgoConfig",
                "gamma": 1.0,
                "lam": 1.0,
                "adv_estimator": "grpo",
                "norm_adv_by_std_in_grpo": True,
                "use_kl_in_reward": False,
                "kl_penalty": "kl",
                "kl_ctrl": {
                    "_target_": "verl.trainer.config.KLControlConfig",
                    "type": "fixed",
                    "kl_coef": 0.001,
                    "horizon": 10000,
                    "target_kl": 0.1,
                },
                "use_pf_ppo": False,
                "pf_ppo": {
                    "reweight_method": "pow",
                    "weight_pow": 2.0,
                },
                "rollout_correction": {
                    "rollout_is": None,
                    "rollout_is_threshold": 2.0,
                    "rollout_rs": None,
                    "rollout_rs_threshold": None,
                    "bypass_mode": False,
                    "loss_type": "ppo_clip",
                    "rollout_is_batch_normalize": False,
                },
            },
        ),
        (
            "trainer",
            {
                "balance_batch": True,
                "total_epochs": 2,
                "total_training_steps": None,
                "project_name": "MiniOneRec_RL",
                "experiment_name": "minionerec_grpo_ddp",
                "logger": ["tensorboard"],
                "log_val_generations": 0,
                "rollout_data_dir": None,
                "validation_data_dir": None,
                "best_ckpt_metric": None,
                "default_hdfs_dir": None,
                "nnodes": 1,
                "n_gpus_per_node": 1,
                "save_freq": -1,
                "test_freq": -1,
                "esi_redundant_time": 0,
                "resume_mode": "auto",
                "resume_from_path": None,
                "val_before_train": True,
                "val_only": False,
                "critic_warmup": 0,
                "del_local_ckpt_after_load": False,
                "default_local_dir": "checkpoints/${trainer.project_name}/${trainer.experiment_name}",
                "max_actor_ckpt_to_keep": None,
                "max_critic_ckpt_to_keep": None,
                "ray_wait_register_center_timeout": 300,
                "device": "cuda",
                "remove_previous_ckpt_in_save": False,
            },
        ),
        (
            "reward",
            {
                "_target_": "verl.trainer.config.RewardConfig",
                "num_workers": 1,
                "launch_reward_fn_async": False,
                "reward_type": "function",
                "reward_manager": "naive",
                "reward_model": {
                    "enable": False,
                    "enable_resource_pool": False,
                    "rm_coef": 1.0,
                    "n_gpus_per_node": 1,
                    "nnodes": 0,
                    "model_path": None,
                    "model": {
                        "path": None,
                        "tokenizer_path": "${actor_rollout_ref.model.path}",
                        "use_shared_fs": False,
                        "trust_remote_code": False,
                        "external_lib": None,
                        "override_config": {},
                        "enable_gradient_checkpointing": False,
                    },
                    "micro_batch_size_per_gpu": None,
                    "log_prob_micro_batch_size_per_gpu": None,
                    "max_length": None,
                    "use_dynamic_batch": "${oc.select:critic.use_dynamic_bsz,false}",
                    "max_num_tokens": 16384,
                    "use_remove_padding": False,
                    "rollout": {
                        "name": None,
                        "dtype": None,
                        "gpu_memory_utilization": None,
                        "enforce_eager": None,
                        "cudagraph_capture_sizes": None,
                        "free_cache_engine": None,
                        "data_parallel_size": None,
                        "expert_parallel_size": None,
                        "tensor_model_parallel_size": None,
                        "max_num_batched_tokens": None,
                        "max_model_len": None,
                        "max_num_seqs": None,
                        "load_format": None,
                        "engine_kwargs": None,
                        "limit_images": None,
                        "enable_chunked_prefill": None,
                        "enable_prefix_caching": None,
                        "disable_log_stats": None,
                        "skip_tokenizer_init": None,
                        "prompt_length": None,
                        "response_length": None,
                    },
                },
                "custom_reward_function": {"path": None, "name": None},
                "verifier": {
                    "reward_coef": 1.0,
                    "tokenizer_path": "${actor_rollout_ref.model.path}",
                    "model": {
                        "path": None,
                        "tokenizer_path": "${actor_rollout_ref.model.path}",
                        "trust_remote_code": False,
                        "override_config": {},
                        "use_remove_padding": False,
                        "use_shm": False,
                        "external_lib": None,
                    },
                    "micro_batch_size_per_gpu": None,
                    "max_length": None,
                    "use_dynamic_batch": "${oc.select:critic.use_dynamic_bsz,false}",
                    "max_num_tokens": 16384,
                    "param_offload": False,
                    "forward_max_token_len_per_gpu": "${oc.select:.max_num_tokens,null}",
                    "norm_batch_size": 256,
                    "verifier_prompt_template": None,
                },
            },
        ),
        (
            "distillation",
            {
                "_target_": "verl.workers.config.DistillationConfig",
                "enabled": False,
                "distillation_loss": {
                    "_target_": "verl.workers.config.DistillationLossConfig",
                    "loss_mode": "k3",
                    "topk": 32,
                    "use_task_rewards": True,
                    "distillation_loss_coef": 1.0,
                    "loss_max_clamp": None,
                    "log_prob_min_clamp": None,
                    "use_policy_gradient": False,
                    "policy_loss_mode": "vanilla",
                    "clip_ratio": 0.2,
                    "clip_ratio_low": 0.2,
                    "clip_ratio_high": 0.2,
                },
                "n_gpus_per_node": 0,
                "nnodes": 0,
                "teacher_models": {
                    "teacher_model": {
                        "_target_": "verl.workers.config.DistillationTeacherModelConfig",
                        "key": None,
                        "model_path": None,
                        "num_replicas": 0,
                        "inference": {
                            "_target_": "verl.workers.config.RolloutConfig",
                            "name": "${oc.select:actor_rollout_ref.rollout.name}",
                            "dtype": "${oc.select:actor_rollout_ref.rollout.dtype}",
                            "gpu_memory_utilization": 0.5,
                            "enforce_eager": True,
                            "cudagraph_capture_sizes": None,
                            "free_cache_engine": True,
                            "data_parallel_size": 1,
                            "expert_parallel_size": 1,
                            "tensor_model_parallel_size": 2,
                            "max_num_batched_tokens": 8192,
                            "max_model_len": None,
                            "max_num_seqs": 1024,
                            "load_format": "auto",
                            "engine_kwargs": {},
                            "limit_images": None,
                            "enable_chunked_prefill": True,
                            "enable_prefix_caching": True,
                            "disable_log_stats": True,
                            "skip_tokenizer_init": False,
                            "prompt_length": "${oc.select:actor_rollout_ref.rollout.prompt_length}",
                            "response_length": "${oc.select:actor_rollout_ref.rollout.response_length}",
                            "temperature": "${oc.select:actor_rollout_ref.rollout.temperature}",
                        },
                    }
                },
                "teacher_key": "data_source",
            },
        ),
    )
    for key, value in placeholders:
        merge_missing(key, value)


def _build_main():
    @ray.remote(num_cpus=1)
    class TaskRunner(BaseTaskRunner):
        def __init__(self):
            super().__init__()

        def run(self, config):
            task_impl = _select_task(config)
            task_impl.sanitize_fsdp2_wrap_policy(config)
            _validate_strategy_signals(config, task_impl, "actor", "task_runner_pre_prepare")
            _validate_strategy_signals(config, task_impl, "ref", "task_runner_pre_prepare")
            _validate_model_signal(config, "task_runner_pre_prepare")
            pprint(OmegaConf.to_container(config, resolve=False))
            prepared = task_impl.prepare(config)
            tokenizer = prepared["tokenizer"]
            processor = prepared["processor"]
            actor_rollout_cls = prepared["actor_rollout_cls"]
            ray_worker_group_cls = prepared["ray_worker_group_cls"]

            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            if need_reference_policy(config) and not ref_in_actor:
                actor_role = Role.ActorRolloutRef
            else:
                actor_role = Role.ActorRollout
            self.role_worker_mapping[actor_role] = ray.remote(actor_rollout_cls)
            self.mapping[actor_role] = "global_pool"

            if need_critic(config):
                self.add_critic_worker(config)

            self.add_reward_model_resource_pool(config)
            self.add_teacher_model_resource_pool(config)
            self.add_ref_policy_worker(config, actor_rollout_cls)

            resource_pool_manager = self.init_resource_pool_mgr(config)

            train_dataset = create_rl_dataset(
                config.data.train_files,
                config.data,
                tokenizer,
                processor,
                is_train=True,
                max_samples=config.data.get("train_max_samples", -1),
            )
            val_dataset = create_rl_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                is_train=False,
                max_samples=config.data.get("val_max_samples", -1),
            )
            train_sampler = create_rl_sampler(config.data, train_dataset)

            trainer = RLTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )
            trainer.init_workers()
            trainer.fit()

    def run_ppo(config) -> None:
        task_impl = _select_task(config)
        task_impl.sanitize_fsdp2_wrap_policy(config)
        auto_set_device(config)
        _inject_legacy_reward_placeholders(config)
        config = migrate_legacy_reward_impl(config)
        _ensure_runtime_root_blocks(config)
        _validate_strategy_signals(config, task_impl, "actor", "driver_pre_base_run_ppo")
        _validate_strategy_signals(config, task_impl, "ref", "driver_pre_base_run_ppo")
        _validate_model_signal(config, "driver_pre_base_run_ppo")
        base_run_ppo(config, task_runner_class=TaskRunner)

    @hydra.main(config_path=str(_CONFIG_ROOT), config_name="openonerec/grpo_trainer", version_base=None)
    def main(config):
        run_ppo(config)

    return main


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _build_main()()
