"""OpenOneRec-style two-stage vLLM rollout."""

from __future__ import annotations

from importlib import import_module
import logging

import torch

from verl import DataProto
from verl_gr.third_party.vllm import BeamSearchParams, LoRARequest
from verl_gr.workers.rollout.primitives import (
    build_lora_requests,
    build_sampling_params,
    expand_beam_candidates,
    pack_rollout_batch,
    prepare_prompt_token_inputs,
)

logger = logging.getLogger(__name__)

ServerAdapter = getattr(import_module("verl.workers.rollout.vllm_rollout.vllm_rollout"), "ServerAdapter")


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

        stage1_max_tokens = kwargs.get(
            "stage1_max_tokens",
            _read_rollout_custom_value(self.config, "stage1_max_tokens", kwargs.get("max_tokens", 1024)),
        )
        cot_sampling_params = build_sampling_params(
            max_tokens=stage1_max_tokens,
            n=1,
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            stop=["</think>"],
            include_stop_str_in_output=True,
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
        prefix_ids = tokenizer.encode("\n<|sid_begin|>", add_special_tokens=False)
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

        beam_width = kwargs.get(
            "stage2_beam_size",
            _read_rollout_custom_value(self.config, "stage2_beam_size", 32),
        )
        max_tokens_item = kwargs.get(
            "stage2_max_tokens",
            kwargs.get(
                "stage2_num_tokens",
                _read_rollout_custom_value(self.config, "stage2_num_tokens", 16),
            ),
        )
        if BeamSearchParams is None:
            raise ImportError("BeamSearchParams not available; cannot run stage-2 beam search.")

        beam_params = BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens_item)
        item_outputs = self.inference_engine.beam_search(prompts=stage2_inputs, params=beam_params)

        expansion = expand_beam_candidates(
            item_outputs=item_outputs,
            stage_inputs=stage2_inputs,
            idx=idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            non_tensor_batch=non_tensor_batch,
            beam_width=beam_width,
            return_all_beams=kwargs.get("return_all_beams", True),
            beam_idxs=non_tensor_batch.get("beam_idx"),
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
        """Delegate weight sync to ServerAdapter async transport."""
        await super().update_weights(weights=weights, global_steps=global_steps, **kwargs)

    async def release(self):
        """Lifecycle hook required by BaseRollout in verl>=0.7.x."""
        await super().release()
