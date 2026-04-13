"""Bridge helpers that isolate vllm imports for OpenOneRec recipes."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def get_sampling_params_cls() -> Any:
    return getattr(import_module("vllm"), "SamplingParams")


def get_lora_request_cls() -> Any:
    return getattr(import_module("vllm.lora.request"), "LoRARequest")


def get_beam_search_params_cls_or_none() -> Any:
    try:
        return getattr(import_module("vllm.sampling_params"), "BeamSearchParams")
    except Exception:  # pragma: no cover - optional based on vllm version
        return None
