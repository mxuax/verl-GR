"""Local OpenOneRec GRPO entrypoint with OneRec TaskRunner behavior."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from verl_gr.contracts.rl_contract import RLWorkloadProfile

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "verl_gr" / "openonerec"


def _build_main():
    hydra = import_module("hydra")
    ray = import_module("ray")
    trainer_main = import_module("verl.trainer.main_ppo")
    BaseTaskRunner = getattr(trainer_main, "TaskRunner")
    run_verl_ppo = getattr(trainer_main, "run_ppo")
    OneRecActorRolloutRefWorker = getattr(
        import_module("verl_gr.recipes.openonerec.onerec_fsdp_workers"),
        "OneRecActorRolloutRefWorker",
    )

    class OneRecTaskRunner(BaseTaskRunner):
        """TaskRunner override that routes two-stage rollout to OneRec worker."""

        def add_actor_rollout_worker(self, config):
            workload_profile = RLWorkloadProfile(
                actor_strategy=str(config.actor_rollout_ref.actor.strategy),
                rollout_name=str(config.actor_rollout_ref.rollout.name),
                rollout_mode=str(config.actor_rollout_ref.rollout.mode),
                use_legacy_worker_impl=str(config.trainer.get("use_legacy_worker_impl", "auto")),
            )
            if workload_profile.requires_onerec_actor_worker():
                RayWorkerGroup = getattr(import_module("verl.single_controller.ray"), "RayWorkerGroup")
                Role = getattr(import_module("verl.trainer.ppo.ray_trainer"), "Role")
                actor_rollout_cls = OneRecActorRolloutRefWorker
                self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
                self.mapping[Role.ActorRollout] = "global_pool"
                return actor_rollout_cls, RayWorkerGroup
            return super().add_actor_rollout_worker(config)

    def run_ppo(config) -> None:
        task_runner_class = ray.remote(num_cpus=1)(OneRecTaskRunner)
        run_verl_ppo(config, task_runner_class=task_runner_class)

    @hydra.main(config_path=str(_CONFIG_ROOT), config_name="grpo_trainer", version_base=None)
    def hydra_entry(config):
        run_ppo(config)

    return hydra_entry


if __name__ == "__main__":
    _build_main()()

