"""Small compatibility helpers for vLLM imports."""

from __future__ import annotations

from vllm import SamplingParams
from vllm.lora.request import LoRARequest

try:
    from vllm.sampling_params import BeamSearchParams
except Exception:  # pragma: no cover - optional across vLLM versions
    BeamSearchParams = None

__all__ = [
    "BeamSearchParams",
    "LoRARequest",
    "SamplingParams",
]
