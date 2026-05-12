"""OpenOneRec vLLM native beam search adapter.

Provides a bridge between verl-GR's two-stage rollout and vLLM's native
``BeamSearchParams`` / ``LLM.beam_search()`` API.

Status: STUB — vLLM V1 ``AsyncLLM`` (used by verl's async rollout path)
does not expose ``beam_search()``.  This adapter will be filled in once
one of the following approaches is validated:

- Access the underlying synchronous ``LLM`` instance from the vLLM server.
- Use a vLLM engine callback to inject beam search via ``LLM.beam_search()``.
- Fall back to HF deterministic beam (reuse the MiniOneRec HF generator).

For now, the OpenOneRec stage-2 beam search continues to use the legacy
Python beam backend (``beam_backend.py``) via ``_run_stage2_beam_search()``.
"""

from __future__ import annotations


class VLLMNativeBeamAdapter:
    """Stub adapter — will be implemented when vLLM beam API is accessible."""

    def __init__(self):
        raise NotImplementedError(
            "VLLMNativeBeamAdapter is a stub. "
            "vLLM V1 AsyncLLM does not expose beam_search(). "
            "See module docstring for planned approaches."
        )
