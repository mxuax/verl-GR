"""Rank-GRPO async rollout fast path.

The upstream single-turn agent loop issues one vLLM request per repeated rollout.
Rank-GRPO's prompts are text-only and repeated contiguously, so we fire each
prompt group as *n* independent ``n=1`` vLLM requests concurrently via
``asyncio.gather`` — matching TRL's colocated generation behaviour (one
independent generation per completion).  The results are expanded back into
verl's normal DataProto layout.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import numpy as np
import ray
import torch

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
)
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout.replica import TokenOutput
from verl_gr.workers.rollout.rankgrpo_vllm_async import RankGRPOvLLMReplica


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _build_rankgrpo_sampling_params(config, *, validate: bool) -> dict[str, Any]:
    params = {
        # Match TRL's colocated vLLM path: one independent sample per request.
        "n": 1,
        "repetition_penalty": 1.0,
        "temperature": _cfg_get(config, "temperature", 1.0),
        "top_p": _cfg_get(config, "top_p", 1.0),
        "top_k": _cfg_get(config, "top_k", -1),
        "min_p": _cfg_get(config, "min_p", 0.0),
        "max_tokens": _cfg_get(config, "response_length", None),
        "logprobs": _cfg_get(config, "calculate_log_probs", False),
    }
    if validate:
        val_kwargs = _cfg_get(config, "val_kwargs", None)
        params["temperature"] = _cfg_get(val_kwargs, "temperature", params["temperature"])
        params["top_p"] = _cfg_get(val_kwargs, "top_p", params["top_p"])
        params["top_k"] = _cfg_get(val_kwargs, "top_k", params["top_k"])
        params["min_p"] = _cfg_get(val_kwargs, "min_p", params["min_p"])
    if params["max_tokens"] is None:
        params.pop("max_tokens")
    return params


def _resolve_eos_token_id(tokenizer) -> int | list[int] | None:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, (list, tuple)):
        return [int(token_id) for token_id in eos_token_id]
    return int(eos_token_id)


def build_trl_completion_mask(response_ids: list[int], eos_token_id: int | list[int] | None) -> list[int]:
    """Build TRL ``completion_mask``: 1 through first EOS inclusive, 0 after.

    Matches ``Rank-GRPO/libs/trl/rank_grpo_trainer.py`` when
    ``mask_truncated_completions`` is disabled.
    """
    if not response_ids:
        return []
    if eos_token_id is None:
        return [1] * len(response_ids)

    response_tensor = torch.tensor([response_ids], dtype=torch.long)
    mask = get_response_mask(response_tensor, eos_token=eos_token_id, dtype=torch.int64)
    return mask[0].tolist()


from verl_gr.recipes.rankgrpo.rankgrpo_rollout_utils import maybe_truncate_rankgrpo_response


def _mask_rollout_logprobs(
    response_logprobs: list[float] | None,
    response_mask: list[int],
) -> list[float] | None:
    if response_logprobs is None:
        return None
    masked: list[float] = []
    for idx, mask_value in enumerate(response_mask):
        if idx < len(response_logprobs):
            masked.append(response_logprobs[idx] if mask_value else 0.0)
        else:
            masked.append(0.0)
    return masked


class RankGRPOAgentLoopWorker(AgentLoopWorker):
    """Batch repeated Rank-GRPO single-turn rollouts before calling vLLM.

    Each prompt group is identified by identical ``raw_prompt_ids`` and gets
    ``n`` independent ``n=1`` vLLM requests fired concurrently via
    ``asyncio.gather``, matching TRL's one-independent-generation-per-completion
    behaviour.
    """

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        if not self._can_use_rankgrpo_fast_path(batch):
            return await super().generate_sequences(batch)

        sampling_params = self._build_sampling_params(batch)
        groups = self._group_repeated_prompts(batch)

        tasks = [
            asyncio.create_task(self._generate_group(batch, positions, prompt_ids, sampling_params))
            for positions, prompt_ids in groups
        ]
        group_results = await asyncio.gather(*tasks)

        outputs_by_position: dict[int, Any] = {}
        for positions, outputs in group_results:
            if len(outputs) != len(positions):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} completions for {len(positions)} Rank-GRPO rollouts."
                )
            for position, output in zip(positions, outputs, strict=True):
                outputs_by_position[position] = output

        outputs = [outputs_by_position[position] for position in range(len(batch))]
        return self._postprocess(
            outputs,
            input_non_tensor_batch=batch.non_tensor_batch,
            validate=batch.meta_info.get("validate", False),
        )

    def _can_use_rankgrpo_fast_path(self, batch: DataProto) -> bool:
        if self.processor is not None or self.reward_loop_worker_handles is not None or self.distillation_enabled:
            return False
        if "raw_prompt_ids" not in batch.non_tensor_batch:
            return False
        agent_names = batch.non_tensor_batch.get("agent_name")
        if agent_names is None:
            return True
        return all(str(name) == "single_turn_agent" for name in agent_names)

    def _build_sampling_params(self, batch: DataProto) -> dict[str, Any]:
        return _build_rankgrpo_sampling_params(
            self.rollout_config,
            validate=batch.meta_info.get("validate", False),
        )

    def _group_repeated_prompts(self, batch: DataProto) -> list[tuple[list[int], list[int]]]:
        groups: list[tuple[list[int], list[int]]] = []
        current_positions: list[int] = []
        current_key = None
        current_prompt_ids: list[int] | None = None

        for position, prompt_ids_value in enumerate(batch.non_tensor_batch["raw_prompt_ids"]):
            prompt_ids = normalize_token_ids(prompt_ids_value)
            key = tuple(prompt_ids)
            if current_key is not None and key != current_key:
                assert current_prompt_ids is not None
                groups.append((current_positions, current_prompt_ids))
                current_positions = []
            current_key = key
            current_prompt_ids = prompt_ids
            current_positions.append(position)

        if current_positions:
            assert current_prompt_ids is not None
            groups.append((current_positions, current_prompt_ids))
        return groups

    async def _generate_group(
        self,
        batch: DataProto,
        positions: list[int],
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> tuple[list[int], list[Any]]:
        n = len(positions)
        metrics: dict[str, float] = {}
        with simple_timer("generate_sequences", metrics):
            # Match TRL: n independent n=1 vLLM requests, sent concurrently.
            tasks = [
                self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
                for _ in range(n)
            ]
            token_outputs: list[TokenOutput] = await asyncio.gather(*tasks)

        eos_token_id = _resolve_eos_token_id(self.tokenizer)
        outputs = []
        for token_output in token_outputs:
            response_ids = token_output.token_ids[: self.rollout_config.response_length]
            response_ids = maybe_truncate_rankgrpo_response(
                response_ids,
                self.tokenizer,
                eos_token_id=eos_token_id,
            )
            response_ids = response_ids[: self.rollout_config.response_length]
            response_mask = build_trl_completion_mask(response_ids, eos_token_id)
            response_logprobs = _mask_rollout_logprobs(
                (
                    token_output.log_probs[: self.rollout_config.response_length]
                    if token_output.log_probs is not None
                    else None
                ),
                response_mask,
            )
            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                routed_experts=token_output.routed_experts,
                multi_modal_data={},
                num_turns=2,
                metrics=AgentLoopMetrics(
                    generate_sequences=metrics["generate_sequences"],
                    num_preempted=token_output.num_preempted if token_output.num_preempted is not None else -1,
                ),
                extra_fields=dict(token_output.extra_fields or {}),
            )
            output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
            outputs.append(
                await self._agent_loop_postprocess(
                    output,
                    batch.meta_info.get("validate", False),
                    **{key: value[positions[0]] for key, value in batch.non_tensor_batch.items()},
                )
            )
        return positions, outputs


class RankGRPOAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that keeps repeated Rank-GRPO rollout groups colocated."""

    def __init__(self, *args, **kwargs):
        self.rollout_replica_class = RankGRPOvLLMReplica
        self.agent_loop_workers_class = ray.remote(RankGRPOAgentLoopWorker)
        super().__init__(*args, **kwargs)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        if "raw_prompt_ids" not in prompts.non_tensor_batch:
            return await super().generate_sequences(prompts)

        chunk_indices = self._grouped_worker_indices(prompts)
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(prompts.select_idxs(indices))
                for worker, indices in zip(self.agent_loop_workers, chunk_indices, strict=False)
                if len(indices) > 0
            ]
        )
        output = DataProto.concat(outputs)

        metrics = [worker_output.meta_info.pop("metrics") for worker_output in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _grouped_worker_indices(self, prompts: DataProto) -> list[list[int]]:
        groups: list[list[int]] = []
        current_group: list[int] = []
        current_key = None
        for idx, prompt_ids_value in enumerate(prompts.non_tensor_batch.get("raw_prompt_ids", [])):
            key = tuple(normalize_token_ids(prompt_ids_value))
            if current_key is not None and key != current_key:
                groups.append(current_group)
                current_group = []
            current_key = key
            current_group.append(idx)
        if current_group:
            groups.append(current_group)

        chunks = [[] for _ in self.agent_loop_workers]
        if not groups:
            return chunks

        # Preserve global sample order so DataProto.concat(worker_outputs) remains
        # aligned with the input batch without relying on an auxiliary field to
        # survive recipe-specific postprocessing.
        target_size = max(1, int(np.ceil(len(prompts) / max(1, len(chunks)))))
        chunk_idx = 0
        for group in groups:
            if chunks[chunk_idx] and len(chunks[chunk_idx]) + len(group) > target_size and chunk_idx < len(chunks) - 1:
                chunk_idx += 1
            chunks[chunk_idx].extend(group)
        return chunks
