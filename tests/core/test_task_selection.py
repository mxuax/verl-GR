"""Runtime task selection behavior tests."""

from __future__ import annotations

from omegaconf import OmegaConf

from verl_gr.trainers.main_ppo import TASK_REGISTRY, _infer_legacy_task_name, _select_task


def test_task_registry_includes_all_recipes():
    assert set(TASK_REGISTRY) == {"openonerec", "minionerec", "rankgrpo"}


def test_infer_legacy_task_name_from_rollout():
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "constrained_beam"}}, "data": {}, "algorithm": {}})
    assert _infer_legacy_task_name(config) == "minionerec"

    config.actor_rollout_ref.rollout.name = "two_stage"
    assert _infer_legacy_task_name(config) == "openonerec"


def test_select_task_prefers_class_path_over_task_name():
    config = OmegaConf.create(
        {
            "task": {
                "name": "openonerec",
                "class_path": "verl_gr.recipes.minionerec.minionerec_recipe.MiniOneRecTask",
            },
            "actor_rollout_ref": {"rollout": {"name": "vllm"}},
            "data": {},
            "algorithm": {},
        }
    )
    task = _select_task(config)
    assert task.__class__.__name__ == "MiniOneRecTask"
