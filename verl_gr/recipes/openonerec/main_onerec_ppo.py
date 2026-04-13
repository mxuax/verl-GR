"""Local OpenOneRec GRPO entrypoint.

This module is intentionally hosted in verl-GR so launchers do not import recipe
code from the external OpenOneRec package path.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

_CONFIG_ROOT = (
    Path(__file__).resolve().parents[4] / "OpenOneRec" / "verl_rl" / "verl" / "trainer" / "config"
)


def _build_main():
    hydra = import_module("hydra")
    ray = import_module("ray")
    constants = import_module("verl.trainer.constants_ppo")
    trainer_main = import_module("verl.trainer.main_ppo")
    get_ppo_ray_runtime_env = getattr(constants, "get_ppo_ray_runtime_env")
    run_verl_ppo = getattr(trainer_main, "run_ppo")

    def run_ppo(config) -> None:
        if not ray.is_initialized():
            ray.init(
                runtime_env=get_ppo_ray_runtime_env(),
                num_cpus=config.ray_init.num_cpus,
            )
        run_verl_ppo(config)

    @hydra.main(config_path=str(_CONFIG_ROOT), config_name="ppo_trainer", version_base=None)
    def hydra_entry(config):
        run_ppo(config)

    return hydra_entry


if __name__ == "__main__":
    _build_main()()

