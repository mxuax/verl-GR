"""Async vLLM rollout server with OpenOneRec-style two-stage generation."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Optional

import ray

from verl_gr.workers.rollout.beam_backend import run_async_beam_search
from verl_gr.workers.rollout.beam_config import (
    BeamSearchConfig,
    get_rollout_custom_value,
    resolve_beam_search_config,
    resolve_two_stage_decode_config,
)
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    extract_prompt_logprobs,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    LoRARequest,
    RequestOutput,
    SamplingParams,
    TokensPrompt,
    vLLMHttpServer,
    vLLMReplica,
)

logger = logging.getLogger(__name__)


def _extract_output_log_probs(output, token_ids: list[int]) -> Optional[list[float]]:
    if output.logprobs is None:
        return None
    return [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(output.logprobs)]


async def _drain_final_request_output(generator: Any) -> Optional[RequestOutput]:
    final_res: Optional[RequestOutput] = None
    while True:
        try:
            final_res = await generator.__anext__()
        except StopAsyncIteration:
            return final_res


class TwoStagevLLMHttpServer(vLLMHttpServer):
    """Serve one stage-1 sample and reuse its stage-2 beams across async calls."""

    _MAX_TWO_STAGE_CACHE_SIZE = 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._two_stage_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._two_stage_build_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        # vLLM V1 can become unstable when too many short-lived async requests hit the
        # engine core at once. Two-stage beam search multiplies request fan-out, so we
        # cap in-flight engine requests per server to keep IPC pressure bounded.
        max_inflight_requests = int(
            get_rollout_custom_value(
                self.config,
                "two_stage_max_inflight_requests",
                get_rollout_custom_value(self.config, "beam_subrequest_parallelism", 8),
            )
        )
        self._two_stage_engine_request_semaphore = asyncio.Semaphore(max(1, max_inflight_requests))

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort vLLM requests and clear two-stage state kept outside vLLM."""
        build_tasks = list(self._two_stage_build_tasks.values())
        cancelled_count = 0
        for task in build_tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1

        if build_tasks:
            await asyncio.gather(*build_tasks, return_exceptions=True)

        self._two_stage_build_tasks.clear()
        cleared_cache_entries = len(self._two_stage_cache)
        self._two_stage_cache.clear()

        result = await super().abort_all_requests(reset_prefix_cache=reset_prefix_cache)
        result["two_stage_cancelled_build_tasks"] = cancelled_count
        result["two_stage_cleared_cache_entries"] = cleared_cache_entries
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
        if not sampling_params.pop("enable_two_stage_rollout", False):
            return await super().generate(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                priority=priority,
            )
        return await self._generate_two_stage(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=image_data,
            video_data=video_data,
            priority=priority,
        )

    async def _generate_two_stage(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        priority: int,
    ) -> TokenOutput:
        beam_config = resolve_beam_search_config(
            sampling_params,
            config=self.config,
            request_id=request_id,
        )
        beam_index = beam_config.index % max(beam_config.width, 1)
        cache_key = str(beam_config.group_id)
        cache_entry = await self._get_or_build_two_stage_cache_entry(
            cache_key=cache_key,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=image_data,
            video_data=video_data,
            priority=priority,
            beam_config=beam_config,
        )
        self._two_stage_cache.move_to_end(cache_key)
        selected_idx = min(beam_index, len(cache_entry["responses"]) - 1)
        selected = cache_entry["responses"][selected_idx]

        extra_fields = dict(cache_entry["extra_fields"])
        extra_fields["generated_items"] = cache_entry["generated_items"][selected_idx]
        extra_fields["_beam_index"] = selected_idx
        extra_fields["_beam_group_id"] = cache_key

        remaining = int(cache_entry.get("remaining", 0)) - 1
        if remaining <= 0:
            self._two_stage_cache.pop(cache_key, None)
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

    async def _build_two_stage_cache_entry(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
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

        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)
        lora_request = await self._build_lora_request()
        want_logprobs = bool(sampling_params.pop("logprobs", False))
        temperature = sampling_params.pop("temperature", 1.0)
        top_p = sampling_params.pop("top_p", 1.0)
        top_k = sampling_params.pop("top_k", -1)
        decode_config = resolve_two_stage_decode_config(
            sampling_params,
            config=self.config,
            response_length=sampling_params.get("max_tokens", self.config.response_length),
        )

        stage1_params = SamplingParams(
            max_tokens=max(0, min(decode_config.reasoning.max_tokens, self.config.max_model_len - len(prompt_ids))),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=decode_config.reasoning.stop,
            include_stop_str_in_output=decode_config.reasoning.include_stop_str_in_output,
            logprobs=0 if want_logprobs else None,
            repetition_penalty=self.config.get("repetition_penalty", 1.0),
        )
        stage1_res = await self._run_generate_request(
            prompt=prompt,
            sampling_params=stage1_params,
            request_id=f"{request_id}:stage1",
            lora_request=lora_request,
            priority=priority,
        )

        stage1_output = stage1_res.outputs[0]
        stage1_token_ids = list(stage1_output.token_ids)
        prefix_ids = self.model_config.tokenizer.encode(
            decode_config.item_generation.prefix_text,
            add_special_tokens=False,
        )
        stage2_prompt_ids = prompt_ids + stage1_token_ids + prefix_ids
        stage2_candidates = await self._run_stage2_beam_search(
            prompt_token_ids=stage2_prompt_ids,
            multi_modal_data=multi_modal_data,
            request_id=f"{request_id}:stage2",
            lora_request=lora_request,
            priority=priority,
            beam_config=beam_config,
        )

        extra_fields = {"global_steps": self.global_steps}
        extract_prompt_logprobs(
            output=stage1_res,
            num_prompt_logprobs=stage1_params.prompt_logprobs,
            result_dict=extra_fields,
        )

        responses: list[dict[str, Any]] = []
        generated_items: list[list[int]] = []
        for candidate in stage2_candidates:
            item_token_ids = list(candidate.generated_token_ids)
            full_response_ids = stage1_token_ids + prefix_ids + item_token_ids
            stage1_log_probs = _extract_output_log_probs(stage1_output, stage1_token_ids)
            stage2_log_probs = candidate.log_probs if want_logprobs else None
            if stage1_log_probs is not None and stage2_log_probs is not None:
                combined_log_probs = stage1_log_probs + [0.0] * len(prefix_ids) + stage2_log_probs
            else:
                combined_log_probs = None
            responses.append({"token_ids": full_response_ids, "log_probs": combined_log_probs})
            generated_items.append(item_token_ids)

        return {
            "responses": responses,
            "generated_items": generated_items,
            "extra_fields": extra_fields,
            "remaining": max(1, beam_config.width),
        }

    async def _get_or_build_two_stage_cache_entry(
        self,
        *,
        cache_key: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        priority: int,
        beam_config: BeamSearchConfig,
    ) -> dict[str, Any]:
        cached = self._two_stage_cache.get(cache_key)
        if cached is not None:
            return cached

        build_task = self._two_stage_build_tasks.get(cache_key)
        if build_task is None:
            build_task = asyncio.create_task(
                self._build_two_stage_cache_entry(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    image_data=image_data,
                    video_data=video_data,
                    priority=priority,
                    beam_config=beam_config,
                )
            )
            self._two_stage_build_tasks[cache_key] = build_task
            build_task.add_done_callback(
                lambda finished_task, key=cache_key: self._two_stage_build_tasks.pop(key, None)
                if self._two_stage_build_tasks.get(key) is finished_task
                else None
            )

        cache_entry = await build_task
        existing = self._two_stage_cache.get(cache_key)
        if existing is not None:
            return existing

        self._two_stage_cache[cache_key] = cache_entry
        self._trim_two_stage_cache()
        return cache_entry

    async def _run_stage2_beam_search(
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

        async def generate_one_token(current_prompt_token_ids: list[int], request_suffix: str):
            prompt = TokensPrompt(
                prompt_token_ids=current_prompt_token_ids,
                multi_modal_data=multi_modal_data,
            )
            params = SamplingParams(
                max_tokens=1,
                logprobs=max(2 * beam_config.width, 1),
                temperature=beam_config.temperature,
                top_p=beam_config.top_p,
                top_k=beam_config.top_k,
                repetition_penalty=1.0,
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
            generate_one_token=generate_one_token,
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
        await self._two_stage_engine_request_semaphore.acquire()
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
            self._two_stage_engine_request_semaphore.release()

    async def _build_lora_request(self) -> Optional[LoRARequest]:
        if self.lora_as_adapter:
            loaded_loras = None
            loaded_loras = await self.engine.list_loras()
            if VLLM_LORA_INT_ID in loaded_loras:
                return LoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=VLLM_LORA_INT_ID,
                    lora_path=VLLM_LORA_PATH,
                )
        return None

    def _trim_two_stage_cache(self) -> None:
        while len(self._two_stage_cache) > self._MAX_TWO_STAGE_CACHE_SIZE:
            self._two_stage_cache.popitem(last=False)


class TwoStagevLLMReplica(vLLMReplica):
    """vLLM replica that launches the async two-stage server."""

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
        self.server_class = ray.remote(TwoStagevLLMHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "two_stage_"
