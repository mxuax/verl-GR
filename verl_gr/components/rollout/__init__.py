"""Rollout components for verl-GR."""

__all__ = ["OneRecvLLMRollout"]


def __getattr__(name: str):
    if name == "OneRecvLLMRollout":
        from verl_gr.components.rollout.onerec_vllm_rollout import OneRecvLLMRollout

        return OneRecvLLMRollout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
