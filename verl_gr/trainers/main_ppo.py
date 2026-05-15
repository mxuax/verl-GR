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
        base_run_ppo(config, task_runner_class=TaskRunner)

    @hydra.main(config_path=str(_CONFIG_ROOT), config_name="openonerec/grpo_trainer", version_base=None)
    def main(config):
        run_ppo(config)

    return main


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _build_main()()
