"""Native rollout router for verl-GR recommendation tasks.

Dispatches generation requests to the appropriate rollout backend
without going through the per-token Python beam loop in ``beam_backend.py``.

Semantic branches:
- ``mini_hf_constrained_beam_sample``  → MiniOneRec training (HF beam sampling)
- ``mini_hf_constrained_beam_eval``    → MiniOneRec validation (HF deterministic beam)
- ``open_vllm_native_beam``            → OpenOneRec stage-2 (vLLM native beam)
- ``legacy_python_beam``               → fallback (existing per-token beam backend)

The router is invoked from custom agent-loop workers and does NOT modify
``verl/`` source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

DECODE_MODE_TRAIN_KEY = "mini_hf_constrained_beam_sample"
DECODE_MODE_EVAL_KEY = "mini_hf_constrained_beam_eval"
DECODE_MODE_OPEN_VLLM = "open_vllm_native_beam"
DECODE_MODE_LEGACY = "legacy_python_beam"

VALID_DECODE_MODES = frozenset({
    DECODE_MODE_TRAIN_KEY,
    DECODE_MODE_EVAL_KEY,
    DECODE_MODE_OPEN_VLLM,
    DECODE_MODE_LEGACY,
})


@dataclass
class NativeRolloutRequest:
    """Normalized request for native rollout generation."""

    prompt_token_ids: list[int]
    request_id: str
    decode_mode: str
    beam_width: int
    beam_index: int = 0
    beam_group_id: int = 0
    temperature: float = 1.0
    max_tokens: int = 16
    top_p: float = 1.0
    top_k: int = -1
    length_penalty: float = 0.0
    ignore_eos: bool = False
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    is_validate: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class NativeRolloutRouter:
    """Dispatch native rollout generation per task + decode mode.

    Does NOT depend on ``verl/`` internals or the async agent-loop
    ``server_manager.generate()`` path.  Each branch is implemented as a
    standalone callable that consumes ``NativeRolloutRequest`` and
    returns ``NativeRolloutOutput``.
    """

    def __init__(self, *, default_branch: str | None = None):
        self._branches: dict[str, Any] = {}
        self._default_branch = default_branch or DECODE_MODE_LEGACY

    def register(self, name: str, handler: Any) -> None:
        if name not in VALID_DECODE_MODES:
            raise ValueError(f"Unknown decode mode: {name}. Valid: {sorted(VALID_DECODE_MODES)}")
        self._branches[name] = handler

    def resolve(self, decode_mode: str | None) -> Any:
        mode = decode_mode or self._default_branch
        if mode not in self._branches and mode == DECODE_MODE_LEGACY:
            return None  # caller should fall back to legacy agent-loop path
        handler = self._branches.get(mode)
        if handler is None:
            raise RuntimeError(
                f"Decode mode '{mode}' is not registered. "
                f"Registered: {sorted(self._branches.keys())}"
            )
        return handler


def resolve_decode_mode_train(rollout_config: Any) -> str:
    """Extract training decode mode from rollout custom config."""
    custom = getattr(rollout_config, "custom", None) or {}
    if isinstance(custom, dict):
        return str(custom.get("decode_mode_train", DECODE_MODE_TRAIN_KEY))
    return str(getattr(custom, "decode_mode_train", DECODE_MODE_TRAIN_KEY))


def resolve_decode_mode_val(rollout_config: Any) -> str:
    """Extract validation decode mode from rollout custom config."""
    custom = getattr(rollout_config, "custom", None) or {}
    if isinstance(custom, dict):
        return str(custom.get("decode_mode_val", DECODE_MODE_EVAL_KEY))
    return str(getattr(custom, "decode_mode_val", DECODE_MODE_EVAL_KEY))


def resolve_decode_mode_stage2(rollout_config: Any) -> str:
    """Extract OpenOneRec stage-2 decode mode from rollout custom config."""
    custom = getattr(rollout_config, "custom", None) or {}
    if isinstance(custom, dict):
        return str(custom.get("stage2_decode_mode", DECODE_MODE_OPEN_VLLM))
    return str(getattr(custom, "stage2_decode_mode", DECODE_MODE_OPEN_VLLM))


# ---------------------------------------------------------------------------
# TokenOutput → DataProto helpers
# ---------------------------------------------------------------------------

@dataclass
class NativeRolloutOutput:
    token_ids: list[int]
    log_probs: list[float] | None = None
    stop_reason: str | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": self.token_ids,
            "log_probs": self.log_probs or [],
            "stop_reason": self.stop_reason or "length",
            "finish_reason": self.finish_reason,
        }


def build_response_tensors(
    prompt_length: int,
    response_ids_batch: list[list[int]],
    pad_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert a batch of response token lists into DataProto-compatible tensors.

    Args:
        prompt_length: number of prompt tokens (assumed same for all in batch).
        response_ids_batch: one sub-list per returned sequence.
        pad_token_id / eos_token_id: token IDs for padding / EOS.
        device: target torch device.

    Returns dict with keys: ``prompts``, ``responses``, ``input_ids``,
    ``attention_mask``, ``position_ids``.
    """
    max_resp_len = max(len(r) for r in response_ids_batch)
    n_total = len(response_ids_batch)

    responses = torch.full((n_total, max_resp_len), pad_token_id, dtype=torch.long, device=device)
    for i, r in enumerate(response_ids_batch):
        if r:
            responses[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)

    # build response mask: 1 for valid tokens (non-pad, pre-EOS), 0 for pad
    resp_mask = torch.ones(n_total, max_resp_len, dtype=torch.long, device=device)
    for i, r in enumerate(response_ids_batch):
        for j, tid in enumerate(r):
            if tid == eos_token_id:
                resp_mask[i, j+1:] = 0
                break
        resp_mask[i, len(r):] = 0

    prompts = torch.full((n_total, prompt_length), pad_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prompts, responses], dim=1)
    attn = torch.cat([torch.ones(n_total, prompt_length, dtype=torch.long, device=device), resp_mask], dim=1)
    pos_ids = _build_position_ids(attn, device)

    return {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attn,
        "position_ids": pos_ids,
    }


def _build_position_ids(attention_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute position_ids from attention_mask (cumsum-style, equivalent to
    ``verl.utils.model.compute_position_id_with_mask``)."""
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(device)
