"""MiniOneRec task wiring for verl-GR."""

from __future__ import annotations

from omegaconf import OmegaConf
from omegaconf import open_dict

from verl_gr.recipes.minionerec.minionerec_dataset import MiniOneRecDataset
from verl_gr.recipes.minionerec.minionerec_loss import (  # noqa: F401  # register REINFORCE policy loss
    compute_policy_loss_minionerec_reinforce,
)
from verl_gr.recipes.minionerec.minionerec_reward import compute_score
from verl_gr.recipes.task_runtime import RecipeTaskRuntime
from verl_gr.workers.rollout.registration import (
    register_constrained_beam_replica,
    register_constrained_beam_rollout_class,
)

__all__ = ["MiniOneRecDataset", "MiniOneRecTask", "compute_score"]


class MiniOneRecTask(RecipeTaskRuntime):
    """Task runtime for MiniOneRec single-stage constrained beam rollout."""

    def expand_rollout_counts(self, config) -> None:
        rollout_cfg = config.actor_rollout_ref.rollout
        if rollout_cfg.get("name") != "constrained_beam":
            return
        custom_cfg = rollout_cfg.get("custom")
        if custom_cfg is None:
            with open_dict(rollout_cfg):
                rollout_cfg.custom = OmegaConf.create({})
            custom_cfg = rollout_cfg.custom
        beam_size = int(custom_cfg.get("beam_width", custom_cfg.get("beam_size", 20)))
        # MiniOneRec's group size semantics are independent from OpenOneRec two-stage:
        # - base_generations_per_prompt: how many constrained-beam groups per prompt
        # - rollout.n: total rollout requests per prompt seen by trainer/advantage code
        #   (base_generations_per_prompt * beam_width)
        base_generations_per_prompt = int(custom_cfg.get("num_generations_per_prompt", rollout_cfg.get("n", 1)))
        base_generations_per_prompt = max(1, base_generations_per_prompt)
        # Struct-mode compatibility: some remote configs may not define this key yet.
        with open_dict(custom_cfg):
            custom_cfg["num_generations_per_prompt"] = base_generations_per_prompt
        rollout_cfg["n"] = base_generations_per_prompt * max(1, beam_size)

    def configure_rollout(self, config) -> None:
        if config.actor_rollout_ref.rollout.get("name") != "constrained_beam":
            return
        register_constrained_beam_replica()
        register_constrained_beam_rollout_class()
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
            "verl_gr.recipes.minionerec.constrained_beam_agent_loop.MiniOneRecConstrainedBeamAgentLoopManager",
            force_add=True,
        )
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.agent.default_agent_loop",
            "minionerec_constrained_beam_agent",
            force_add=True,
        )

    def get_actor_rollout_ref_worker(self, config):
        if config.actor_rollout_ref.rollout.get("name") == "constrained_beam":
            return __import__(
                "verl_gr.recipes.minionerec.minionerec_fsdp_workers",
                fromlist=["MiniOneRecActorRolloutRefWorker"],
            ).MiniOneRecActorRolloutRefWorker
        return super().get_actor_rollout_ref_worker(config)
