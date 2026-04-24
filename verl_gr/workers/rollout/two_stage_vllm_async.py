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


def extract_prompt_logprobs(output: RequestOutput, num_prompt_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None or output.prompt_logprobs is None:
        return

    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls


def _read_rollout_custom_value(config, key: str, default):
    custom = getattr(config, "custom", None)
    if isinstance(custom, dict):
        return custom.get(key, default)
    if custom is None:
        return default
    try:
        return custom.get(key, default)
    except AttributeError:
        return default


def _extract_output_log_probs(output, token_ids: list[int]) -> Optional[list[float]]:
    if output.logprobs is None:
        return None
    return [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(output.logprobs)]


class TwoStagevLLMHttpServer(vLLMHttpServer):
    """Serve one stage-1 sample and reuse its stage-2 beams across async calls."""

    _MAX_TWO_STAGE_CACHE_SIZE = 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._two_stage_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._two_stage_inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}
        # vLLM V1 can become unstable when too many short-lived async requests hit the
        # engine core at once. Two-stage beam search multiplies request fan-out, so we
        # cap in-flight subrequests per server to keep IPC pressure bounded.
        beam_parallelism = int(get_rollout_custom_value(self.config, "beam_subrequest_parallelism", 8))
        self._two_stage_request_semaphore = asyncio.Semaphore(max(1, beam_parallelism))

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
        beam_width = int(
            sampling_params.pop(
                "stage2_beam_size",
                _read_rollout_custom_value(self.config, "stage2_beam_size", 32),
            )
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
        selected_idx = min(beam_idx, len(cache_entry["responses"]) - 1)
        selected = cache_entry["responses"][selected_idx]

        extra_fields = dict(cache_entry["extra_fields"])
        extra_fields["generated_items"] = cache_entry["generated_items"][selected_idx]
        extra_fields["_beam_idx"] = selected_idx
        extra_fields["_two_stage_group_id"] = cache_key

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
        beam_width: int,
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

        stage1_max_tokens = int(
            sampling_params.pop(
                "stage1_max_tokens",
                _read_rollout_custom_value(
                    self.config,
                    "stage1_max_tokens",
                    sampling_params.get("max_tokens", self.config.response_length),
                ),
            )
        )
        stage2_num_tokens = int(
            sampling_params.pop(
                "stage2_num_tokens",
                sampling_params.pop(
                    "stage2_max_tokens",
                    _read_rollout_custom_value(self.config, "stage2_num_tokens", 16),
                ),
            )
        )

        stage1_params = SamplingParams(
            max_tokens=max(0, min(stage1_max_tokens, self.config.max_model_len - len(prompt_ids))),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=["</think>"],
            include_stop_str_in_output=True,
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
        prefix_ids = self.model_config.tokenizer.encode("\n<|sid_begin|>", add_special_tokens=False)
        stage2_prompt_ids = prompt_ids + stage1_token_ids + prefix_ids
        stage2_prompt = TokensPrompt(prompt_token_ids=stage2_prompt_ids, multi_modal_data=multi_modal_data)

        stage2_params = SamplingParams(
            max_tokens=max(0, min(stage2_num_tokens, self.config.max_model_len - len(stage2_prompt_ids))),
            n=max(1, beam_width),
            temperature=stage2_temperature,
            top_p=1.0,
            top_k=-1,
            logprobs=0 if want_logprobs else None,
            repetition_penalty=1.0,
        )
        stage2_res = await self._run_generate_request(
            prompt=stage2_prompt,
            sampling_params=stage2_params,
            request_id=f"{request_id}:stage2",
            lora_request=lora_request,
            priority=priority,
        )

        extra_fields = {"global_steps": self.global_steps}
        extract_prompt_logprobs(
            output=stage1_res,
            num_prompt_logprobs=stage1_params.prompt_logprobs,
            result_dict=extra_fields,
        )

        responses: list[dict[str, Any]] = []
        generated_items: list[list[int]] = []
        stage2_outputs = stage2_res.outputs or []
        if not stage2_outputs:
            stage2_outputs = [stage2_res.outputs[0]]
        for seq_idx in range(max(1, beam_width)):
            seq_output = stage2_outputs[seq_idx] if seq_idx < len(stage2_outputs) else stage2_outputs[0]
            item_token_ids = list(seq_output.token_ids)
            full_response_ids = stage1_token_ids + prefix_ids + item_token_ids
            stage1_log_probs = _extract_output_log_probs(stage1_output, stage1_token_ids)
            stage2_log_probs = _extract_output_log_probs(seq_output, item_token_ids)
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
            "remaining": max(1, beam_width),
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

        inflight = self._two_stage_inflight.get(cache_key)
        if inflight is None:
            inflight = asyncio.create_task(
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
            self._two_stage_inflight[cache_key] = inflight
            inflight.add_done_callback(
                lambda finished_task, key=cache_key: self._two_stage_inflight.pop(key, None)
                if self._two_stage_inflight.get(key) is finished_task
                else None
            )

        cache_entry = await inflight
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
        async with self._two_stage_request_semaphore:
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                priority=priority,
            )
            final_res: Optional[RequestOutput] = None
            async for output in generator:
                final_res = output
            assert final_res is not None
            return final_res

    async def _build_lora_request(self) -> Optional[LoRARequest]:
        if not self.lora_as_adapter:
            return None
        lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
        if not lora_loaded:
            return None
        return LoRARequest(
            lora_name=VLLM_LORA_NAME,
            lora_int_id=VLLM_LORA_INT_ID,
            lora_path=VLLM_LORA_PATH,
        )

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
    ):
        # verl_071 vLLMReplica.__init__ doesn't accept is_teacher_model; keep this
        # argument for caller compatibility but do not forward it upstream.
        _ = is_teacher_model
        super().__init__(
            replica_rank=replica_rank,
            config=config,
            model_config=model_config,
            gpus_per_node=gpus_per_node,
            is_reward_model=is_reward_model,
        )
        self.server_class = ray.remote(TwoStagevLLMHttpServer)
