"""OpenOneRec rollout wiring behavior tests."""

from __future__ import annotations

from omegaconf import OmegaConf

from verl_gr.recipes.openonerec.onerec_task import OneRecTask


def test_configure_rollout_sets_two_stage_agent_loop():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "two_stage",
                    "agent": {},
                }
            },
            "data": {},
        }
    )
    OneRecTask().configure_rollout(config)
    assert config.data.return_raw_chat is True
    assert config.actor_rollout_ref.rollout.agent.default_agent_loop == "openonerec_two_stage_agent"
    assert (
        config.actor_rollout_ref.rollout.agent.agent_loop_manager_class
        == "verl_gr.recipes.openonerec.two_stage_agent_loop.OpenOneRecAgentLoopManager"
    )


def test_configure_rollout_noop_for_non_two_stage():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"name": "vllm", "agent": {}}},
            "data": {},
        }
    )
    OneRecTask().configure_rollout(config)
    assert config.data.get("return_raw_chat") is None
