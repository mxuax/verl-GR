"""Async vLLM rollout server for single-stage constrained beam generation."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any, Optional

import ray

from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    LoRARequest,
    RequestOutput,
    SamplingParams,
    TokensPrompt,
    vLLMHttpServer,
    vLLMReplica,
)
from verl_gr.workers.rollout.beam_backend import run_async_beam_search
from verl_gr.workers.rollout.beam_config import BeamSearchConfig, get_rollout_custom_value, resolve_beam_search_config
from verl_gr.workers.rollout.constraints import build_constraint_from_config

logger = logging.getLogger(__name__)


async def _drain_final_request_output(generator: Any) -> Optional[RequestOutput]:
    final_res: Optional[RequestOutput] = None
    while True:
        try:
            final_res = await generator.__anext__()
        except StopAsyncIteration:
            return final_res


class ConstrainedBeamvLLMHttpServer(vLLMHttpServer):
    """Serve one prompt and reuse its constrained beams across async calls."""

    _MAX_CONSTRAINED_BEAM_CACHE_SIZE = 1024

    def __init__(self, *args, **kwargs):
        os.environ.setdefault("VERL_ROLLOUT_ZMQ_NAMESPACE", "constrained-beam")
        os.environ.setdefault("VERL_ZMQ_SOCKET_PREFIX", "verl-gr-constrained-beam")
        super().__init__(*args, **kwargs)
        self._constrained_beam_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._constrained_beam_build_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        max_inflight_requests = self._resolve_max_inflight_requests()
        self._constrained_beam_max_inflight_requests = max(1, max_inflight_requests)
        self._constrained_beam_engine_request_semaphore = asyncio.Semaphore(self._constrained_beam_max_inflight_requests)

    def _resolve_max_inflight_requests(self) -> int:
        custom = getattr(self.config, "custom", None)
        if custom is None:
            custom_mapping = {}
        elif isinstance(custom, dict):
            custom_mapping = custom
        elif hasattr(custom, "items"):
            custom_mapping = dict(custom.items())
        else:
            custom_mapping = {}

        explicit = custom_mapping.get("constrained_beam_max_inflight_requests")
        legacy = custom_mapping.get("beam_subrequest_parallelism")
        if explicit is not None:
            return int(explicit)
        if legacy is not None:
            return int(legacy)

        # Constrained beam emits many max_tokens=1 requests; 8 inflight requests
        # heavily underutilizes vLLM and becomes the dominant bottleneck.
        max_num_seqs = int(getattr(self.config, "max_num_seqs", 128) or 128)
        return max(32, min(max_num_seqs, 128))

    async def get_constrained_beam_runtime_metrics(self) -> dict[str, int]:
        semaphore_waiters = getattr(self._constrained_beam_engine_request_semaphore, "_waiters", None)
        waiter_count = len(semaphore_waiters) if semaphore_waiters is not None else 0
        available_slots = int(getattr(self._constrained_beam_engine_request_semaphore, "_value", 0))
        inflight = max(0, self._constrained_beam_max_inflight_requests - available_slots)
        pending_build_tasks = sum(0 if task.done() else 1 for task in self._constrained_beam_build_tasks.values())
        return {
            "max_inflight_engine_requests": int(self._constrained_beam_max_inflight_requests),
            "inflight_engine_requests": inflight,
            "engine_request_waiters": int(waiter_count),
            "pending_build_tasks": int(pending_build_tasks),
            "cache_entries": int(len(self._constrained_beam_cache)),
        }

    def _get_worker_extension_cls(self) -> str:
        return "verl_gr.workers.rollout.zmq_utils.VerlGRVLLMColocateWorkerExtension"

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        import gc as _gc

        build_tasks = list(self._constrained_beam_build_tasks.values())
        cancelled_count = 0
        for task in build_tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        if build_tasks:
            await asyncio.gather(*build_tasks, return_exceptions=True)
        self._constrained_beam_build_tasks.clear()
        cleared_cache_entries = len(self._constrained_beam_cache)
        self._constrained_beam_cache.clear()
        result = await super().abort_all_requests(reset_prefix_cache=reset_prefix_cache)
        # Free CUDA memory accumulated by frequent weight updates
        _gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        result["constrained_beam_cancelled_build_tasks"] = cancelled_count
        result["constrained_beam_cleared_cache_entries"] = cleared_cache_entries
        return result

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        sampling_params = dict(sampling_params)
        if not sampling_params.pop("enable_constrained_beam_rollout", False):
            return await super().generate(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                priority=priority,
            )
        return await self._generate_constrained_beam(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=image_data,
            video_data=video_data,
            priority=priority,
        )

    async def _generate_constrained_beam(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        priority: int,
    ) -> TokenOutput:
        beam_config = resolve_beam_search_config(sampling_params, config=self.config, request_id=request_id)
        beam_index = beam_config.index % max(beam_config.width, 1)
        if beam_config.decode_mode == "stochastic_constrained":
            sample = await self._run_constrained_stochastic_sample(
                prompt_ids=prompt_ids,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                priority=priority,
                beam_config=beam_config,
            )
            extra_fields = {"global_steps": self.global_steps}
            extra_fields["generated_items"] = list(sample["token_ids"])
            extra_fields["_beam_index"] = beam_index
            extra_fields["_beam_group_id"] = str(beam_config.group_id)
            # TokenOutput.stop_reason is a string field. Keep raw reason in extra_fields for debugging.
            extra_fields["_raw_stop_reason"] = sample.get("stop_reason", "completed")
            return TokenOutput(
                token_ids=sample["token_ids"],
                log_probs=sample["log_probs"],
                routed_experts=None,
                stop_reason="completed",
                num_preempted=None,
                extra_fields=extra_fields,
            )
        cache_key = str(beam_config.group_id)
        disable_cache_for_this_request = beam_config.disable_cache_in_train and not self._is_validation_group_id(cache_key)
        if disable_cache_for_this_request:
            cache_entry = await self._build_constrained_beam_cache_entry(
                prompt_ids=prompt_ids,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                priority=priority,
                beam_config=beam_config,
            )
        else:
            cache_entry = await self._get_or_build_constrained_beam_cache_entry(
                cache_key=cache_key,
                prompt_ids=prompt_ids,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                priority=priority,
                beam_config=beam_config,
            )
            self._constrained_beam_cache.move_to_end(cache_key)
        selected_idx = min(beam_index, len(cache_entry["responses"]) - 1)
        selected = cache_entry["responses"][selected_idx]
        extra_fields = dict(cache_entry["extra_fields"])
        extra_fields["generated_items"] = cache_entry["generated_items"][selected_idx]
        extra_fields["_beam_index"] = selected_idx
        extra_fields["_beam_group_id"] = cache_key

        if not disable_cache_for_this_request:
            remaining = int(cache_entry.get("remaining", 0)) - 1
            if remaining <= 0:
                self._constrained_beam_cache.pop(cache_key, None)
            else:
                cache_entry["remaining"] = remaining

        return TokenOutput(
            token_ids=selected["token_ids"],
            log_probs=selected["log_probs"],
            routed_experts=None,
            stop_reason="completed",
            num_preempted=None,
            extra_fields=extra_fields,
        )

    @staticmethod
    def _is_validation_group_id(group_id: str) -> bool:
        parts = str(group_id).split(":")
        return len(parts) >= 2 and parts[1] == "1"

    async def _build_constrained_beam_cache_entry(
        self,
        *,
        prompt_ids: list[int],
        request_id: str,
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        priority: int,
        beam_config: BeamSearchConfig,
    ) -> dict[str, Any]:
        prompt_ids = normalize_token_ids(prompt_ids)
        prompt_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        lora_request = await self._build_lora_request()
        candidates = await self._run_constrained_beam_search(
            prompt_token_ids=prompt_ids,
            multi_modal_data=multi_modal_data,
            request_id=f"{request_id}:constrained_beam",
            lora_request=lora_request,
            priority=priority,
            beam_config=beam_config,
        )
        responses = []
        generated_items = []
        for candidate in candidates:
            token_ids = list(candidate.generated_token_ids)
            responses.append({"token_ids": token_ids, "log_probs": candidate.log_probs})
            generated_items.append(token_ids)
        return {
            "responses": responses,
            "generated_items": generated_items,
            "extra_fields": {"global_steps": self.global_steps},
            "remaining": max(1, beam_config.width),
        }

    async def _run_constrained_stochastic_sample(
        self,
        *,
        prompt_ids: list[int],
        request_id: str,
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        priority: int,
        beam_config: BeamSearchConfig,
    ) -> dict[str, Any]:
        prompt_token_ids = normalize_token_ids(prompt_ids)
        prompt_token_ids = qwen2_5_vl_dedup_image_tokens(prompt_token_ids, self.model_config.processor)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        lora_request = await self._build_lora_request()
        eos_token_id = self.model_config.tokenizer.eos_token_id
        allowed_tokens_fn = build_constraint_from_config(beam_config.constraint, tokenizer=self.model_config.tokenizer)
        generated_token_ids: list[int] = []
        generated_log_probs: list[float] = []
        stop_reason: str = "length"
        for step in range(max(1, beam_config.max_tokens)):
            allowed_token_ids = None
            if allowed_tokens_fn is not None:
                allowed_token_ids = list(allowed_tokens_fn(prompt_token_ids, generated_token_ids))
                if not allowed_token_ids and eos_token_id is not None:
                    allowed_token_ids = [int(eos_token_id)]
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids + generated_token_ids, multi_modal_data=multi_modal_data)
            params = SamplingParams(
                max_tokens=1,
                logprobs=1,
                temperature=beam_config.temperature,
                top_p=beam_config.top_p,
                top_k=beam_config.top_k,
                repetition_penalty=1.0,
                allowed_token_ids=allowed_token_ids,
            )
            output = await self._run_generate_request(
                prompt=prompt,
                sampling_params=params,
                request_id=f"{request_id}:sample_step_{step}",
                lora_request=lora_request,
                priority=priority,
            )
            sampled_token_id = None
            sampled_logprob = 0.0
            if output.outputs and output.outputs[0].token_ids:
                sampled_token_id = int(output.outputs[0].token_ids[0])
                if output.outputs[0].logprobs:
                    sampled_logprob = float(output.outputs[0].logprobs[0].get(sampled_token_id).logprob) if sampled_token_id in output.outputs[0].logprobs[0] else 0.0
            if sampled_token_id is None or (allowed_token_ids is not None and sampled_token_id not in set(allowed_token_ids)):
                if allowed_token_ids:
                    sampled_token_id = int(eos_token_id) if eos_token_id in allowed_token_ids else int(min(allowed_token_ids))
                    sampled_logprob = 0.0
                elif eos_token_id is not None:
                    sampled_token_id = int(eos_token_id)
                    sampled_logprob = 0.0
                else:
                    break
            generated_token_ids.append(sampled_token_id)
            generated_log_probs.append(sampled_logprob)
            if sampled_token_id == eos_token_id and not beam_config.ignore_eos:
                stop_reason = "eos"
                break
        return {"token_ids": generated_token_ids, "log_probs": generated_log_probs, "stop_reason": stop_reason}

    async def _get_or_build_constrained_beam_cache_entry(self, *, cache_key: str, **kwargs) -> dict[str, Any]:
        cached = self._constrained_beam_cache.get(cache_key)
        if cached is not None:
            return cached
        build_task = self._constrained_beam_build_tasks.get(cache_key)
        if build_task is None:
            build_task = asyncio.create_task(self._build_constrained_beam_cache_entry(**kwargs))
            self._constrained_beam_build_tasks[cache_key] = build_task
            build_task.add_done_callback(
                lambda finished_task, key=cache_key: self._constrained_beam_build_tasks.pop(key, None)
                if self._constrained_beam_build_tasks.get(key) is finished_task
                else None
            )
        cache_entry = await build_task
        existing = self._constrained_beam_cache.get(cache_key)
        if existing is not None:
            return existing
        self._constrained_beam_cache[cache_key] = cache_entry
        while len(self._constrained_beam_cache) > self._MAX_CONSTRAINED_BEAM_CACHE_SIZE:
            self._constrained_beam_cache.popitem(last=False)
        return cache_entry

    async def _run_constrained_beam_search(
        self,
        *,
        prompt_token_ids: list[int],
        multi_modal_data: dict[str, Any],
        request_id: str,
        lora_request: Optional[LoRARequest],
        priority: int,
        beam_config: BeamSearchConfig,
    ):
        eos_token_id = self.model_config.tokenizer.eos_token_id
        allowed_tokens_fn = build_constraint_from_config(beam_config.constraint, tokenizer=self.model_config.tokenizer)

        async def generate_next_tokens(
            current_prompt_token_ids_list: list[list[int]],
            request_suffixes: list[str],
            allowed_token_ids_list: list[list[int]] | None,
        ):
            tasks = []
            if allowed_token_ids_list is None:
                allowed_token_ids_list = [None] * len(current_prompt_token_ids_list)  # type: ignore[list-item]
            for current_prompt_token_ids, request_suffix, allowed_token_ids in zip(
                current_prompt_token_ids_list,
                request_suffixes,
                allowed_token_ids_list,
                strict=True,
            ):
                tasks.append(
                    asyncio.create_task(
                        generate_one_token(
                            current_prompt_token_ids=current_prompt_token_ids,
                            request_suffix=request_suffix,
                            allowed_token_ids=allowed_token_ids,
                        )
                    )
                )
            return await asyncio.gather(*tasks)

        async def generate_one_token(
            current_prompt_token_ids: list[int],
            request_suffix: str,
            allowed_token_ids: list[int] | None,
        ):
            prompt = TokensPrompt(prompt_token_ids=current_prompt_token_ids, multi_modal_data=multi_modal_data)
            params = SamplingParams(
                max_tokens=1,
                # Beam backend only consumes top-(2*beam_width) candidates.
                # Requesting more logprobs increases payload/CPU with no effect.
                logprobs=max(2 * beam_config.width, 1),
                temperature=beam_config.temperature,
                top_p=beam_config.top_p,
                top_k=beam_config.top_k,
                repetition_penalty=1.0,
                allowed_token_ids=allowed_token_ids,
            )
            return await self._run_generate_request(
                prompt=prompt,
                sampling_params=params,
                request_id=f"{request_id}:{request_suffix}",
                lora_request=lora_request,
                priority=priority,
            )

        return await run_async_beam_search(
            prompt_token_ids=prompt_token_ids,
            beam_width=beam_config.width,
            max_tokens=max(0, min(beam_config.max_tokens, self.config.max_model_len - len(prompt_token_ids))),
            eos_token_id=eos_token_id,
            ignore_eos=beam_config.ignore_eos,
            length_penalty=beam_config.length_penalty,
            temperature=beam_config.temperature,
            generate_next_tokens=generate_next_tokens,
            allowed_tokens_fn=allowed_tokens_fn,
            decode_mode=beam_config.decode_mode,
        )

    async def _run_generate_request(
        self,
        *,
        prompt: TokensPrompt,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest],
        priority: int,
    ) -> RequestOutput:
        await self._constrained_beam_engine_request_semaphore.acquire()
        try:
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                priority=priority,
            )
            final_res = await _drain_final_request_output(generator)
            assert final_res is not None
            return final_res
        finally:
            self._constrained_beam_engine_request_semaphore.release()

    async def _build_lora_request(self) -> Optional[LoRARequest]:
        if self.lora_as_adapter:
            loaded_loras = await self.engine.list_loras()
            if VLLM_LORA_INT_ID in loaded_loras:
                return LoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=VLLM_LORA_INT_ID,
                    lora_path=VLLM_LORA_PATH,
                )
        return None


class ConstrainedBeamvLLMReplica(vLLMReplica):
    """vLLM replica that launches the async constrained-beam server."""

    def __init__(
        self,
        replica_rank: int,
        config,
        model_config,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model,
            is_teacher_model,
            name_suffix,
        )
        self.server_class = ray.remote(ConstrainedBeamvLLMHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "constrained_beam_"
