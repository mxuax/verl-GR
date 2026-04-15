"""OpenOneRec local PPO entrypoint with custom OneRec trainer."""

import os
from importlib import import_module
from pathlib import Path

from verl_gr.components.tokenization.sid_tokenizer import build_hf_tokenizer_and_processor
from verl_gr.integrations.verl.openonerec_bridge import (
    get_async_actor_rollout_ref_worker,
    get_critic_worker,
    get_fsdp_reward_model_worker,
    get_megatron_ray_worker_group,
    get_megatron_reward_model_worker,
    get_megatron_worker_symbols,
    get_ppo_runtime_symbols,
    get_ray_worker_group,
)

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "verl_gr" / "openonerec"


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


def _sanitize_fsdp2_wrap_policy(config) -> None:
    actor_rollout_ref = config.get("actor_rollout_ref")
    if actor_rollout_ref is None:
        return
    for role_name in ("actor", "ref"):
        role_cfg = actor_rollout_ref.get(role_name)
        if role_cfg is None or str(role_cfg.get("strategy", "")) != "fsdp2":
            continue
        fsdp_cfg = role_cfg.get("fsdp_config")
        if fsdp_cfg is None:
            continue
        wrap_policy = fsdp_cfg.get("wrap_policy")
        if wrap_policy is None:
            continue
        normalized = _normalize_layer_wrap_value(wrap_policy.get("transformer_layer_cls_to_wrap"))
        if normalized is not None:
            wrap_policy["transformer_layer_cls_to_wrap"] = normalized


def _get_reward_model_cfg(config):
    reward_root = config.get("reward")
    if reward_root is not None and reward_root.get("reward_model") is not None:
        return reward_root.reward_model
    legacy_cfg = config.get("reward_model")
    if legacy_cfg is not None:
        return legacy_cfg
    return None


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
    copy_to_local = runtime_symbols["copy_to_local"]
    collate_fn = runtime_symbols["collate_fn"]

    # Import trainer symbols lazily to avoid importing heavy verl stack
    # during module import/inspection time.
    rl_trainer_mod = import_module("verl_gr.trainers.rl_trainer")
    Role = getattr(rl_trainer_mod, "Role")
    ResourcePoolManager = getattr(rl_trainer_mod, "ResourcePoolManager")
    RayPPOTrainer = getattr(rl_trainer_mod, "RayPPOTrainer")

    @ray.remote(num_cpus=1)
    class OneRecTaskRunner:
        def run(self, config):
            _sanitize_fsdp2_wrap_policy(config)
            OmegaConf.resolve(config)
            reward_model_cfg = _get_reward_model_cfg(config)
            local_path = copy_to_local(
                config.actor_rollout_ref.model.path,
                use_shm=config.actor_rollout_ref.model.get("use_shm", False),
            )
            trust_remote_code = config.data.get("trust_remote_code", False)
            tokenizer, processor = build_hf_tokenizer_and_processor(
                local_path,
                trust_remote_code=trust_remote_code,
                hf_tokenizer_loader=runtime_symbols["hf_tokenizer"],
                hf_processor_loader=runtime_symbols["hf_processor"],
            )

            if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
                RayWorkerGroup = get_ray_worker_group()
                OneRecActorRolloutRefWorker = getattr(
                    import_module("verl_gr.recipes.openonerec.onerec_fsdp_workers"),
                    "OneRecActorRolloutRefWorker",
                )
                AsyncActorRolloutRefWorker = get_async_actor_rollout_ref_worker()
                use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
                CriticWorker = get_critic_worker(use_legacy_worker_impl)
                actor_rollout_cls = (
                    AsyncActorRolloutRefWorker
                    if config.actor_rollout_ref.rollout.mode == "async"
                    else OneRecActorRolloutRefWorker
                )
                ray_worker_group_cls = RayWorkerGroup
            elif config.actor_rollout_ref.actor.strategy == "megatron":
                NVMegatronRayWorkerGroup = get_megatron_ray_worker_group()
                megatron_workers = get_megatron_worker_symbols()
                ActorRolloutRefWorker = megatron_workers["ActorRolloutRefWorker"]
                AsyncActorRolloutRefWorker = megatron_workers["AsyncActorRolloutRefWorker"]
                CriticWorker = megatron_workers["CriticWorker"]
                actor_rollout_cls = (
                    AsyncActorRolloutRefWorker
                    if config.actor_rollout_ref.rollout.mode == "async"
                    else ActorRolloutRefWorker
                )
                ray_worker_group_cls = NVMegatronRayWorkerGroup
            else:
                raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")

            if reward_model_cfg is not None and reward_model_cfg.get("enable", False):
                if reward_model_cfg.strategy in {"fsdp", "fsdp2"}:
                    RewardModelWorker = get_fsdp_reward_model_worker()
                elif reward_model_cfg.strategy == "megatron":
                    RewardModelWorker = get_megatron_reward_model_worker()
                else:
                    raise NotImplementedError(f"Unknown reward model strategy: {reward_model_cfg.strategy}")
            else:
                RewardModelWorker = None

            n_gpus_per_node = config.trainer.n_gpus_per_node
            nnodes = config.trainer.nnodes
            global_pool_id = "global_pool"
            resource_pool_spec = {global_pool_id: [n_gpus_per_node] * nnodes}
            role_worker_mapping = {Role.ActorRollout: ray.remote(actor_rollout_cls)}
            mapping = {Role.ActorRollout: global_pool_id}
            if config.critic.get("enable", True):
                role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
                mapping[Role.Critic] = global_pool_id
            if reward_model_cfg is not None and reward_model_cfg.get("enable", False) and RewardModelWorker is not None:
                role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
                mapping[Role.RewardModel] = global_pool_id
            if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
                role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
                mapping[Role.RefPolicy] = global_pool_id

            resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

            train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
            train_sampler = create_rl_sampler(config.data, train_dataset)

            trainer = RayPPOTrainer(
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

    def run_onerec_ppo(config) -> None:
        _sanitize_fsdp2_wrap_policy(config)
        auto_set_device(config)
        config = migrate_legacy_reward_impl(config)
        base_run_ppo(config, task_runner_class=OneRecTaskRunner)

    @hydra.main(config_path=str(_CONFIG_ROOT), config_name="grpo_trainer", version_base=None)
    def main(config):
        run_onerec_ppo(config)

    return main


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _build_main()()
