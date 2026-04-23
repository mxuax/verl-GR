"""Thin rollout helpers shared by recommendation-style workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

from verl_gr.third_party.vllm import SamplingParams


@dataclass
class PreparedPromptInputs:
    """Prepared vLLM inputs plus a mutable copy of non-tensor metadata."""

    vllm_inputs: list[dict[str, Any]]
    non_tensor_batch: dict[str, Any]


@dataclass
class CandidateExpansion:
    """Expanded response candidates aligned with prompt-side tensors."""

    responses: list[list[int]]
    idx: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    non_tensor_batch: dict[str, Any]
    batch_size: int


def prepare_prompt_token_inputs(
    prompts: DataProto,
    *,
    pad_token_id: int,
    preprocess_inputs: Callable[[int, torch.Tensor], Any],
) -> PreparedPromptInputs:
    """Convert prompt-side `DataProto` tensors into vLLM token inputs."""

    idx = prompts.batch["input_ids"]
    batch_size = idx.size(0)
    non_tensor_batch = dict(prompts.non_tensor_batch)

    if "raw_prompt_ids" not in non_tensor_batch:
        non_tensor_batch["raw_prompt_ids"] = np.array(
            [preprocess_inputs(pad_token_id, idx[i]) for i in range(batch_size)],
            dtype=object,
        )

    raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
    multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
    vllm_inputs: list[dict[str, Any]] = []

    if multi_modal_data is not None:
        iterator = zip(raw_prompt_ids, multi_modal_data, strict=True)
        for prompt_token_ids, sample_multi_modal_data in iterator:
            vllm_inputs.append(
                {
                    "prompt_token_ids": _to_token_list(prompt_token_ids),
                    "multi_modal_data": sample_multi_modal_data,
                }
            )
    else:
        for prompt_token_ids in raw_prompt_ids:
            vllm_inputs.append({"prompt_token_ids": _to_token_list(prompt_token_ids)})

    return PreparedPromptInputs(vllm_inputs=vllm_inputs, non_tensor_batch=non_tensor_batch)


def build_lora_requests(
    inference_engine: Any,
    *,
    lora_kwargs: Any,
    lora_request_cls: type[Any],
    batch_size: int,
) -> list[Any] | None:
    """Build a shared LoRA request list when rollout uses adapter weights."""

    if not lora_kwargs:
        return None

    lora_int_ids = list(inference_engine.llm_engine.list_loras())
    if not lora_int_ids:
        return None

    lora_int_id = lora_int_ids[0]
    return [
        lora_request_cls(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
    ] * batch_size


def build_sampling_params(
    *,
    max_tokens: int,
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    stop: list[str] | None = None,
    include_stop_str_in_output: bool | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> SamplingParams:
    """Build a small, explicit SamplingParams object for rollout code."""

    sampling_kwargs: dict[str, Any] = {
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }
    if stop is not None:
        sampling_kwargs["stop"] = stop
    if include_stop_str_in_output is not None:
        sampling_kwargs["include_stop_str_in_output"] = include_stop_str_in_output
    if extra_kwargs:
        sampling_kwargs.update(extra_kwargs)
    return SamplingParams(**sampling_kwargs)


def expand_beam_candidates(
    *,
    item_outputs: list[Any],
    stage_inputs: list[dict[str, Any]],
    idx: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    non_tensor_batch: dict[str, Any],
    beam_width: int,
    beam_return_mode: str = "all_beams",
    beam_indices: Any = None,
) -> CandidateExpansion:
    """Expand beam outputs back into prompt-aligned tensors and metadata."""

    responses: list[list[int]] = []
    return_all_beams = beam_return_mode == "all_beams"

    if return_all_beams:
        expanded_idx: list[int] = []
        expanded_beam_indices: list[int] = []
        for sample_idx, output in enumerate(item_outputs):
            original_prompt_len = len(stage_inputs[sample_idx]["prompt_token_ids"])
            num_seqs = len(output.sequences)
            for seq_idx in range(beam_width):
                best_seq = output.sequences[seq_idx] if seq_idx < num_seqs else output.sequences[0]
                responses.append(best_seq.tokens[original_prompt_len:])
                expanded_idx.append(sample_idx)
                expanded_beam_indices.append(seq_idx)

        idx = idx[expanded_idx]
        attention_mask = attention_mask[expanded_idx]
        position_ids = position_ids[expanded_idx]
        non_tensor_batch = _expand_non_tensor_batch(non_tensor_batch, expanded_idx)
        non_tensor_batch["_beam_indices"] = np.array(expanded_beam_indices, dtype=np.int64)
        batch_size = len(responses)
    else:
        for sample_idx, output in enumerate(item_outputs):
            original_prompt_len = len(stage_inputs[sample_idx]["prompt_token_ids"])
            seq_idx = int(beam_indices[sample_idx]) if beam_indices is not None else 0
            if seq_idx >= len(output.sequences):
                seq_idx = 0
            responses.append(output.sequences[seq_idx].tokens[original_prompt_len:])
        batch_size = len(responses)

    return CandidateExpansion(
        responses=responses,
        idx=idx,
        attention_mask=attention_mask,
        position_ids=position_ids,
        non_tensor_batch=non_tensor_batch,
        batch_size=batch_size,
    )


def pack_rollout_batch(
    *,
    idx: torch.Tensor,
    responses: list[list[int]] | torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int,
    response_length: int,
    calculate_log_probs: bool,
    non_tensor_batch: dict[str, Any],
) -> DataProto:
    """Pack responses back into a `DataProto` using verl-compatible fields."""

    if not isinstance(responses, torch.Tensor):
        response = pad_2d_list_to_length(responses, pad_token_id, max_length=response_length).to(idx.device)
    else:
        response = responses

    seq = torch.cat([idx, response], dim=-1)
    batch_size = response.size(0)
    delta_position_id = torch.arange(1, response.size(1) + 1, device=position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
    if position_ids.dim() == 3:
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

    response_position_ids = position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
    response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
    attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

    batch = TensorDict(
        {
            "prompts": idx,
            "responses": response,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )
    if calculate_log_probs:
        batch["rollout_log_probs"] = torch.zeros_like(response, dtype=torch.float32)

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def _expand_non_tensor_batch(non_tensor_batch: dict[str, Any], expanded_idx: list[int]) -> dict[str, Any]:
    expanded_non_tensor_batch: dict[str, Any] = {}
    for key, val in non_tensor_batch.items():
        if isinstance(val, np.ndarray):
            expanded_non_tensor_batch[key] = val[expanded_idx]
        elif isinstance(val, list):
            expanded_non_tensor_batch[key] = [val[idx] for idx in expanded_idx]
        else:
            expanded_non_tensor_batch[key] = val
    return expanded_non_tensor_batch


def _to_token_list(prompt_token_ids: Any) -> list[int]:
    if isinstance(prompt_token_ids, np.ndarray):
        return prompt_token_ids.tolist()
    if isinstance(prompt_token_ids, list):
        return prompt_token_ids
    return list(prompt_token_ids)
