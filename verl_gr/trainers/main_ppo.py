"""Local PPO entrypoint with customize verl-gr trainer."""

import os
from importlib import import_module
from pathlib import Path

from verl_gr.integrations.verl.bridge import (
    get_ppo_runtime_symbols,
)
from verl_gr.recipes.openonerec.onerec_recipe import OneRecTask

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "verl_gr" / "openonerec"


def _build_main():
    hydra = import_module("hydra")
    ray = import_module("ray")
    OmegaConf = getattr(import_module("omegaconf"), "OmegaConf")
    runtime_symbols = get_ppo_runtime_symbols()
    base_run_ppo = runtime_symbols["run_ppo"]
    create_rl_dataset = runtime_symbols["create_rl_dataset"]
    create_rl_sampler = runtime_symbols["create_rl_sampler"]
    auto_set_device = runtime_symbols["auto_set_device"]
    migrate_legacy_reward_impl = runtime_symbols["migrate_legacy_reward_impl"]
    collate_fn = runtime_symbols["collate_fn"]

    # Import trainer symbols lazily to avoid importing heavy verl stack
    # during module import/inspection time.
    rl_trainer_mod = import_module("verl_gr.trainers.rl_trainer")
    Role = getattr(rl_trainer_mod, "Role")
    ResourcePoolManager = getattr(rl_trainer_mod, "ResourcePoolManager")
    RLTrainer = getattr(rl_trainer_mod, "RLTrainer")
    task_impl = OneRecTask()

    @ray.remote(num_cpus=1)
    class TaskRunner:
        def run(self, config):
            task_impl.sanitize_fsdp2_wrap_policy(config)
            OmegaConf.resolve(config)
            prepared = task_impl.prepare(config, runtime_symbols=runtime_symbols)
            tokenizer = prepared["tokenizer"]
            processor = prepared["processor"]
            actor_rollout_cls = prepared["actor_rollout_cls"]
            critic_worker = prepared["critic_worker"]
            reward_model_worker = prepared["reward_model_worker"]
            reward_model_cfg = prepared["reward_model_cfg"]
            ray_worker_group_cls = prepared["ray_worker_group_cls"]

            n_gpus_per_node = config.trainer.n_gpus_per_node
            nnodes = config.trainer.nnodes
            global_pool_id = "global_pool"
            resource_pool_spec = {global_pool_id: [n_gpus_per_node] * nnodes}
            role_worker_mapping = {Role.ActorRollout: ray.remote(actor_rollout_cls)}
            mapping = {Role.ActorRollout: global_pool_id}
            if config.critic.get("enable", True):
                role_worker_mapping[Role.Critic] = ray.remote(critic_worker)
                mapping[Role.Critic] = global_pool_id
            if reward_model_cfg is not None and reward_model_cfg.get("enable", False) and reward_model_worker is not None:
                role_worker_mapping[Role.RewardModel] = ray.remote(reward_model_worker)
                mapping[Role.RewardModel] = global_pool_id
            if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
                role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
                mapping[Role.RefPolicy] = global_pool_id

            resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

            train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
            train_sampler = create_rl_sampler(config.data, train_dataset)

            trainer = RLTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
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

    @hydra.main(config_path=str(_CONFIG_ROOT), config_name="grpo_trainer", version_base=None)
    def main(config):
        run_ppo(config)

    return main


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _build_main()()
