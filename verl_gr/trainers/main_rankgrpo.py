"""Rank-GRPO entrypoint using the verl/verl_gr trainer stack."""

from __future__ import annotations

import os
from pathlib import Path
from pprint import pprint

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

from verl_gr.recipes.rankgrpo.rankgrpo_recipe import RankGRPOTask
from verl_gr.trainers.rl_trainer import RLTrainer

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "verl_gr" / "rankgrpo"


def _build_main():
    task_impl = RankGRPOTask()

    @ray.remote(num_cpus=1)
    class TaskRunner(BaseTaskRunner):
        def __init__(self):
            super().__init__()

        def run(self, config):
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
        task_impl.sanitize_fsdp2_wrap_policy(config)
        auto_set_device(config)
        config = migrate_legacy_reward_impl(config)
        base_run_ppo(config, task_runner_class=TaskRunner)

    @hydra.main(config_path=str(_CONFIG_ROOT), config_name="rankgrpo_trainer", version_base=None)
    def main(config):
        run_ppo(config)

    return main


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _build_main()()

