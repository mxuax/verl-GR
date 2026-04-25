"""OpenOneRec-style two-stage vLLM rollout."""

from __future__ import annotations

from importlib import import_module
import logging

import ray
import torch

from verl import DataProto
from verl_gr.third_party.vllm import BeamSearchParams, LoRARequest
from verl_gr.workers.rollout.beam_config import (
    BeamSearchConfig,
    resolve_beam_search_config,
    resolve_two_stage_decode_config,
)
from verl_gr.workers.rollout.primitives import (
    build_lora_requests,
    build_sampling_params,
    expand_beam_candidates,
    pack_rollout_batch,
    prepare_prompt_token_inputs,
)

logger = logging.getLogger(__name__)

ServerAdapter = getattr(import_module("verl.workers.rollout.vllm_rollout.vllm_rollout"), "ServerAdapter")


class TwoStagevLLMRollout(ServerAdapter):
    """Generate CoT first, then beam-search item outputs."""

    def __init__(self, *args, **kwargs):
        if {"config", "model_config", "device_mesh"}.issubset(kwargs):
            super().__init__(
                config=kwargs["config"],
                model_config=kwargs["model_config"],
                device_mesh=kwargs["device_mesh"],
                replica_rank=kwargs.get("replica_rank", -1),
            )
            logger.warning(
                "TwoStagevLLMRollout is running in async adapter mode on verl>=0.7.1. "
                "Two-stage generation logic must be implemented in async agent-loop flow."
            )
            return

        raise RuntimeError(
            "TwoStagevLLMRollout async adapter requires kwargs: config, model_config, device_mesh."
        )

    @torch.no_grad()
    def _two_stage_generation(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        prepared_inputs = prepare_prompt_token_inputs(
            prompts,
            pad_token_id=self.pad_token_id,
            preprocess_inputs=None,
        )
        vllm_inputs = prepared_inputs.vllm_inputs
        non_tensor_batch = prepared_inputs.non_tensor_batch

        generation_kwargs = dict(kwargs)
        decode_config = resolve_two_stage_decode_config(
            generation_kwargs,
            config=self.config,
            response_length=kwargs.get("max_tokens", 1024),
        )
        beam_config: BeamSearchConfig = resolve_beam_search_config(
            generation_kwargs,
            config=self.config,
            request_id="sync-two-stage",
            default_max_tokens=decode_config.item_generation.max_tokens or 16,
        )
        cot_sampling_params = build_sampling_params(
            max_tokens=decode_config.reasoning.max_tokens,
            n=1,
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            stop=decode_config.reasoning.stop,
            include_stop_str_in_output=decode_config.reasoning.include_stop_str_in_output,
        )

        lora_requests = build_lora_requests(
            self.inference_engine,
            lora_kwargs=self.lora_kwargs,
            lora_request_cls=LoRARequest,
            batch_size=batch_size,
        )

        cot_outputs = self.inference_engine.generate(
            prompts=vllm_inputs,
            sampling_params=cot_sampling_params,
            lora_request=lora_requests,
            use_tqdm=False,
        )

        stage2_inputs = []
        tokenizer = self.inference_engine.get_tokenizer()
        prefix_ids = tokenizer.encode(decode_config.item_generation.prefix_text, add_special_tokens=False)
        vocab_size = len(tokenizer)

        for i, output in enumerate(cot_outputs):
            cot_token_ids = list(output.outputs[0].token_ids)
            cot_token_ids_filtered = [tid for tid in cot_token_ids if tid < vocab_size]
            original_prompt_ids = vllm_inputs[i]["prompt_token_ids"]
            new_prompt_ids = original_prompt_ids + cot_token_ids_filtered + prefix_ids

            stage2_input = {"prompt_token_ids": new_prompt_ids}
            if "multi_modal_data" in vllm_inputs[i]:
                stage2_input["multi_modal_data"] = vllm_inputs[i]["multi_modal_data"]
            stage2_inputs.append(stage2_input)

        if BeamSearchParams is None:
            raise ImportError("BeamSearchParams not available; cannot run stage-2 beam search.")

        beam_params = BeamSearchParams(
            beam_width=beam_config.width,
            max_tokens=beam_config.max_tokens,
            ignore_eos=beam_config.ignore_eos,
            temperature=beam_config.temperature,
            length_penalty=beam_config.length_penalty,
        )
        item_outputs = self.inference_engine.beam_search(prompts=stage2_inputs, params=beam_params)

        expansion = expand_beam_candidates(
            item_outputs=item_outputs,
            stage_inputs=stage2_inputs,
            idx=idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            non_tensor_batch=non_tensor_batch,
            beam_width=beam_config.width,
            beam_return_mode=beam_config.return_mode,
            beam_indices=non_tensor_batch.get("beam_index", non_tensor_batch.get("beam_idx")),
        )

        return pack_rollout_batch(
            idx=expansion.idx,
            responses=expansion.responses,
            attention_mask=expansion.attention_mask,
            position_ids=expansion.position_ids,
            pad_token_id=self.pad_token_id,
            eos_token_id=eos_token_id,
            response_length=self.config.response_length,
            calculate_log_probs=self.config.calculate_log_probs,
            non_tensor_batch=expansion.non_tensor_batch,
        )

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if not hasattr(self, "inference_engine"):
            raise NotImplementedError(
                "TwoStagevLLMRollout.generate_sequences() requires inference_engine and is "
                "typically bypassed in verl>=0.7.1 async server mode."
            )
        for key in [
            "max_tokens",
            "temperature",
            "n",
            "top_p",
            "top_k",
            "beam_width",
            "beam_search_params",
            "beam_return_mode",
            "decode_config",
            "stage1_max_tokens",
            "stage2_beam_size",
            "stage2_max_tokens",
            "stage2_num_tokens",
            "return_all_beams",
        ]:
            if key in prompts.meta_info:
                kwargs[key] = prompts.meta_info[key]
        return self._two_stage_generation(prompts, **kwargs)

    async def resume(self, tags: list[str]):
        """Lifecycle hook required by BaseRollout in verl>=0.7.x."""
        await super().resume(tags=tags)

    async def update_weights(self, weights, global_steps: int = None, **kwargs):
        """Abort two-stage requests before syncing weights through ServerAdapter."""
        await self._execute_server_method("abort_all_requests", reset_prefix_cache=True)
        try:
            await super().update_weights(weights=weights, global_steps=global_steps, **kwargs)
        finally:
            await self._execute_server_method("resume_generation")

    async def release(self):
        """Lifecycle hook required by BaseRollout in verl>=0.7.x."""
        await super().release()

    async def _execute_server_method(self, method: str, **kwargs):
        """Call a method on the Ray server actor, not on vLLM engine workers."""
        if self.rollout_rank != 0:
            return None

        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            self.server_handle = ray.get_actor(f"{prefix}server_{self.replica_rank}_{self.node_rank}")

        return await getattr(self.server_handle, method).remote(**kwargs)
