"""Hydra config backfill for verl 0.7.1 compatibility."""

from omegaconf import OmegaConf


def inject_legacy_reward_placeholders(config) -> None:
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


def ensure_runtime_root_blocks(config) -> None:
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
                        "update_weights_bucket_megabytes": 2048,
                        "engine_kwargs": {},
                        "custom_backend_module": None,
                    },
                    "skip": {
                        "_target_": "verl.workers.config.SkipConfig",
                        "enable": False,
                        "action": "cache",
                        "dump_dir": "rollout_skip_dump",
                        "max_dump_step": 1,
                    },
                },
                "actor": {
                    "rollout_n": 1,
                    "ppo_mini_batch_size": 256,
                    "ppo_micro_batch_size": None,
                    "ppo_micro_batch_size_per_gpu": None,
                    "use_dynamic_bsz": False,
                    "ppo_max_token_len_per_gpu": 16384,
                    "clip_ratio": 0.2,
                    "clip_ratio_low": 0.2,
                    "clip_ratio_high": 0.2,
                    "tau_pos": 1.0,
                    "tau_neg": 1.05,
                    "freeze_vision_tower": False,
                    "policy_loss": {
                        "_target_": "verl.workers.config.PolicyLossConfig",
                        "loss_mode": "vanilla",
                        "clip_cov_ratio": 0.0002,
                        "clip_cov_lb": 1.0,
                        "clip_cov_ub": 5.0,
                        "kl_cov_ratio": 0.0002,
                        "ppo_kl_coef": 0.1,
                    },
                    "clip_ratio_c": 3.0,
                    "loss_agg_mode": "token-mean",
                    "loss_scale_factor": None,
                    "entropy_coeff": 0,
                    "calculate_entropy": False,
                    "use_kl_loss": False,
                    "use_prefix_grouper": False,
                    "use_torch_compile": True,
                    "kl_loss_coef": 0.001,
                    "kl_loss_type": "low_var_kl",
                    "ppo_epochs": 1,
                    "shuffle": False,
                    "data_loader_seed": 42,
                    "checkpoint": {
                        "_target_": "verl.trainer.config.CheckpointConfig",
                        "save_contents": ["model", "optimizer", "extra"],
                        "load_contents": "${.save_contents}",
                        "async_save": False,
                        "mbridge_config": {},
                    },
                    "optim": {
                        "_target_": "verl.workers.config.FSDPOptimizerConfig",
                        "optimizer": "AdamW",
                        "optimizer_impl": "torch.optim",
                        "lr": 1e-6,
                        "lr_warmup_steps_ratio": 0.0,
                        "total_training_steps": -1,
                        "weight_decay": 0.01,
                        "lr_warmup_steps": -1,
                        "betas": [0.9, 0.999],
                        "clip_grad": 1.0,
                        "min_lr_ratio": 0.0,
                        "num_cycles": 0.5,
                        "lr_scheduler_type": "constant",
                        "zero_indexed_step": True,
                        "warmup_style": None,
                        "override_optimizer_config": None,
                    },
                    "use_fused_kernels": False,
                    "profiler": {
                        "_target_": "verl.utils.profiler.ProfilerConfig",
                        "tool": None,
                        "enable": False,
                        "all_ranks": False,
                        "ranks": [],
                        "save_path": "${oc.select:global_profiler.save_path,null}",
                        "tool_config": {
                            "nsys": {
                                "_target_": "verl.utils.profiler.config.NsightToolConfig",
                                "discrete": "${oc.select:global_profiler.global_tool_config.nsys.discrete}",
                            },
                            "npu": {
                                "_target_": "verl.utils.profiler.config.NPUToolConfig",
                                "contents": [],
                                "level": "level0",
                                "analysis": True,
                                "discrete": False,
                            },
                            "torch": {
                                "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                                "contents": [],
                                "discrete": False,
                            },
                            "torch_memory": {
                                "_target_": "verl.utils.profiler.config.TorchMemoryToolConfig",
                                "trace_alloc_max_entries": "${oc.select:global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries,100000}",
                                "stack_depth": "${oc.select:global_profiler.global_tool_config.torch_memory.stack_depth,32}",
                            },
                            "precision_debugger": {
                                "_target_": "verl.utils.profiler.config.PrecisionDebuggerToolConfig",
                                "config_path": "${oc.select:global_profiler.global_tool_config.precision_debugger.config_path,null}",
                                "steps": None,
                                "stages": "${oc.select:global_profiler.global_tool_config.precision_debugger.stages,null}",
                                "strict": "${oc.select:global_profiler.global_tool_config.precision_debugger.strict,False}",
                            },
                        },
                    },
                },
                "ref": {
                    "rollout_n": 1,
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": None,
                    "log_prob_use_dynamic_bsz": False,
                    "log_prob_max_token_len_per_gpu": 16384,
                    "profiler": {
                        "_target_": "verl.utils.profiler.ProfilerConfig",
                        "tool": None,
                        "enable": False,
                        "all_ranks": False,
                        "ranks": [],
                        "save_path": "${oc.select:global_profiler.save_path,null}",
                        "tool_config": {
                            "nsys": {
                                "_target_": "verl.utils.profiler.config.NsightToolConfig",
                                "discrete": "${oc.select:global_profiler.global_tool_config.nsys.discrete}",
                            },
                            "npu": {
                                "_target_": "verl.utils.profiler.config.NPUToolConfig",
                                "contents": [],
                                "level": "level0",
                                "analysis": True,
                                "discrete": False,
                            },
                            "torch": {
                                "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                                "contents": [],
                                "discrete": False,
                            },
                            "torch_memory": {
                                "_target_": "verl.utils.profiler.config.TorchMemoryToolConfig",
                                "trace_alloc_max_entries": "${oc.select:global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries,100000}",
                                "stack_depth": "${oc.select:global_profiler.global_tool_config.torch_memory.stack_depth,32}",
                            },
                            "precision_debugger": {
                                "_target_": "verl.utils.profiler.config.PrecisionDebuggerToolConfig",
                                "config_path": "${oc.select:global_profiler.global_tool_config.precision_debugger.config_path,null}",
                                "steps": None,
                                "stages": "${oc.select:global_profiler.global_tool_config.precision_debugger.stages,null}",
                                "strict": "${oc.select:global_profiler.global_tool_config.precision_debugger.strict,False}",
                            },
                        },
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
                "reward_manager": {
                    "_target_": "verl.workers.config.reward_model.RewardManagerConfig",
                    "source": "register",
                    "name": "naive",
                    "module": {
                        "_target_": "verl.trainer.config.config.ModuleConfig",
                        "path": None,
                        "name": "custom_reward_manager",
                    },
                },
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

