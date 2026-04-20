"""OpenOneRec-style two-stage vLLM rollout."""

from __future__ import annotations

from importlib import import_module

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

_LEGACY_SPMD_AVAILABLE = False
vLLMRollout = getattr(import_module("verl.workers.rollout.base"), "BaseRollout")
_pre_process_inputs = None

class TwoStagevLLMRollout(vLLMRollout):
    """Generate CoT first, then beam-search item outputs."""

    def __init__(self, *args, **kwargs):
        if not _LEGACY_SPMD_AVAILABLE:
            raise RuntimeError(
                "TwoStagevLLMRollout requires legacy vLLM SPMD symbols "
                "(`verl.workers.rollout.vllm_rollout.vllm_rollout_spmd`), which are "
                "removed in verl>=0.7.1. Use async vLLM rollout mode (name=vllm) "
                "or pin to a legacy verl version."
            )
        super().__init__(*args, **kwargs)

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
            preprocess_inputs=_pre_process_inputs,
        )
        vllm_inputs = prepared_inputs.vllm_inputs
        non_tensor_batch = prepared_inputs.non_tensor_batch

        stage1_max_tokens = kwargs.get(
            "stage1_max_tokens",
            getattr(self.config, "stage1_max_tokens", kwargs.get("max_tokens", 1024)),
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

        beam_width = kwargs.get("stage2_beam_size", getattr(self.config, "stage2_beam_size", 32))
        max_tokens_item = kwargs.get(
            "stage2_max_tokens",
            kwargs.get("stage2_num_tokens", getattr(self.config, "stage2_num_tokens", 16)),
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

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
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
