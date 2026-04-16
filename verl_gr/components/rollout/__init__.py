"""Rollout components for verl-GR."""

__all__ = ["TwoStagevLLMRollout"]


def __getattr__(name: str):
    if name == "TwoStagevLLMRollout":
        from verl_gr.components.rollout.two_stage_vllm_rollout import TwoStagevLLMRollout

        return TwoStagevLLMRollout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
