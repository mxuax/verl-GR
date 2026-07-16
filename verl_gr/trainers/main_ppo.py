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


def _build_main():
    @ray.remote(num_cpus=1)
    class TaskRunner(BaseTaskRunner):
        def __init__(self):
            super().__init__()

        def run(self, config):
            task_impl = _select_task(config)
            task_impl.sanitize_fsdp2_wrap_policy(config)
            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)
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
        base_run_ppo(config, task_runner_class=TaskRunner)

    @hydra.main(config_path=str(_CONFIG_ROOT), config_name="openonerec/grpo_trainer", version_base=None)
    def main(config):
        run_ppo(config)

    return main


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _build_main()()
