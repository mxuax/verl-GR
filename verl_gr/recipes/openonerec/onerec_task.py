"""OpenOneRec task runtime."""

from importlib import import_module

from omegaconf import OmegaConf, open_dict

from verl_gr.recipes.task_runtime import RecipeTaskRuntime
from verl_gr.workers.rollout.beam_config import BEAM_WIDTH_KEY
from verl_gr.workers.rollout.registration import (
    register_two_stage_replica,
    register_two_stage_rollout_class,
)

__all__ = ["OneRecTask"]

class OneRecTask(RecipeTaskRuntime):
    """OpenOneRec task-specific runtime preparation logic."""

    def expand_rollout_counts(self, config) -> None:
        rollout_cfg = config.actor_rollout_ref.rollout
        if rollout_cfg.get("name") != "two_stage":
            return

        custom_cfg = rollout_cfg.get("custom")
        if custom_cfg is None:
            with open_dict(rollout_cfg):
                rollout_cfg.custom = OmegaConf.create({})
            custom_cfg = rollout_cfg.custom

        beam_size = int(custom_cfg.get(BEAM_WIDTH_KEY, custom_cfg.get("stage2_beam_size", 32)))
        base_train_n = int(rollout_cfg.get("n", 1))
        rollout_cfg["n"] = base_train_n * beam_size

        val_kwargs = rollout_cfg.get("val_kwargs")
        if val_kwargs is not None:
            base_val_n = int(val_kwargs.get("n", 1))
            val_kwargs["n"] = base_val_n * beam_size

    def configure_rollout(self, config) -> None:
        if config.actor_rollout_ref.rollout.get("name") != "two_stage":
            return
        register_two_stage_replica()
        register_two_stage_rollout_class()
        OmegaConf.update(
            config,
            "data.return_raw_chat",
            True,
            force_add=True,
        )
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
            "verl_gr.recipes.openonerec.two_stage_agent_loop.OpenOneRecAgentLoopManager",
            force_add=True,
        )
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.agent.default_agent_loop",
            "openonerec_two_stage_agent",
            force_add=True,
        )

    def get_actor_rollout_ref_worker(self, config):
        if config.actor_rollout_ref.rollout.get("name") == "two_stage":
            return getattr(
                import_module("verl_gr.recipes.openonerec.onerec_fsdp_workers"),
                "OneRecActorRolloutRefWorker",
            )
        return super().get_actor_rollout_ref_worker(config)


