"""OpenOneRec local PPO entrypoint with custom OneRec trainer."""

import os
from importlib import import_module
from pathlib import Path

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

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "verl_gr" / "openonerec"


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


def _build_main():
    hydra = import_module("hydra")
    ray = import_module("ray")
    OmegaConf = getattr(import_module("omegaconf"), "OmegaConf")
    runtime_symbols = get_ppo_runtime_symbols()
    get_ppo_ray_runtime_env = runtime_symbols["get_ppo_ray_runtime_env"]
    create_rl_dataset = runtime_symbols["create_rl_dataset"]
    create_rl_sampler = runtime_symbols["create_rl_sampler"]
    is_cuda_available = runtime_symbols["is_cuda_available"]
    load_reward_manager = runtime_symbols["load_reward_manager"]
    copy_to_local = runtime_symbols["copy_to_local"]
    hf_tokenizer = runtime_symbols["hf_tokenizer"]
    hf_processor = runtime_symbols["hf_processor"]
    collate_fn = runtime_symbols["collate_fn"]

    onerec_trainer_mod = import_module("verl_gr.recipes.openonerec.onerec_ray_trainer")
    Role = getattr(onerec_trainer_mod, "Role")
    ResourcePoolManager = getattr(onerec_trainer_mod, "ResourcePoolManager")
    RayPPOTrainer = getattr(onerec_trainer_mod, "RayPPOTrainer")

    @ray.remote(num_cpus=1)
    class OneRecTaskRunner:
        def run(self, config):
            _sanitize_fsdp2_wrap_policy(config)
            OmegaConf.resolve(config)
            local_path = copy_to_local(
                config.actor_rollout_ref.model.path,
                use_shm=config.actor_rollout_ref.model.get("use_shm", False),
            )
            trust_remote_code = config.data.get("trust_remote_code", False)
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

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

            if config.reward_model.get("enable", False):
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    RewardModelWorker = get_fsdp_reward_model_worker()
                elif config.reward_model.strategy == "megatron":
                    RewardModelWorker = get_megatron_reward_model_worker()
                else:
                    raise NotImplementedError(f"Unknown reward model strategy: {config.reward_model.strategy}")
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
            if config.reward_model.get("enable", False) and RewardModelWorker is not None:
                role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
                mapping[Role.RewardModel] = global_pool_id
            if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
                role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
                mapping[Role.RefPolicy] = global_pool_id

            reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
            val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
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
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )
            trainer.init_workers()
            trainer.fit()

    def run_ppo(config) -> None:
        _sanitize_fsdp2_wrap_policy(config)
        if not ray.is_initialized():
            ray.init(runtime_env=get_ppo_ray_runtime_env(), num_cpus=config.ray_init.num_cpus)
        if (
            is_cuda_available
            and config.trainer.get("profile_steps") is not None
            and len(config.trainer.get("profile_steps", [])) > 0
        ):
            nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
            runner = OneRecTaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = OneRecTaskRunner.remote()
        ray.get(runner.run.remote(config))
        timeline_json_file = config.ray_init.get("timeline_json_file", None)
        if timeline_json_file:
            ray.timeline(filename=timeline_json_file)

    @hydra.main(config_path=str(_CONFIG_ROOT), config_name="grpo_trainer", version_base=None)
    def main(config):
        run_ppo(config)

    return main


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _build_main()()

