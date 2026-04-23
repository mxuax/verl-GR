"""Async vLLM rollout server with OpenOneRec-style two-stage generation."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional

import ray

from verl_gr.workers.rollout.beam_backend import run_async_beam_search
from verl_gr.workers.rollout.beam_config import (
    BeamSearchConfig,
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


class TwoStagevLLMHttpServer(vLLMHttpServer):
    """Serve one stage-1 sample and reuse its stage-2 beams across async calls."""

    _MAX_TWO_STAGE_CACHE_SIZE = 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._two_stage_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

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

        if cache_key not in self._two_stage_cache:
            self._two_stage_cache[cache_key] = await self._build_two_stage_cache_entry(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                priority=priority,
                beam_config=beam_config,
            )
            self._trim_two_stage_cache()

        cache_entry = self._two_stage_cache[cache_key]
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
        stage2_temperature = float(
            sampling_params.pop(
                "stage2_temperature",
                _read_rollout_custom_value(self.config, "stage2_temperature", 0.0),
            )
        )
        if beam_width > 1 and stage2_temperature <= 0.0:
            # vLLM treats temperature==0 as greedy sampling, which requires n==1.
            # Auto-bump temperature when requesting multiple stage-2 candidates.
            stage2_temperature = 1.0

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
