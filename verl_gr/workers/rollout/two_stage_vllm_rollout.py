"""OpenOneRec-style two-stage vLLM rollout."""

from __future__ import annotations

from importlib import util as importlib_util
from importlib import import_module
import logging
from pathlib import Path
import sys
import types

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


def _load_local_legacy_spmd_module():
    """Load legacy SPMD rollout module from local OneRec fork."""
    legacy_path = (
        Path(__file__).resolve().parents[4]
        / "oneonerec_fredfork/verl_rl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    )
    if not legacy_path.exists():
        return None

    spec = importlib_util.spec_from_file_location("oneonerec_legacy_vllm_rollout_spmd", legacy_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib_util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        # Compatibility shim for newer vLLM versions where legacy
        # `vllm.model_executor.sampling_metadata` no longer exists.
        if exc.name == "vllm.model_executor.sampling_metadata":
            shim = types.ModuleType("vllm.model_executor.sampling_metadata")

            class SamplingMetadata:  # noqa: D401
                """Compatibility placeholder for legacy imports."""

            shim.SamplingMetadata = SamplingMetadata
            sys.modules["vllm.model_executor.sampling_metadata"] = shim
            spec.loader.exec_module(module)
        else:
            raise
    return module

try:
    rollout_spmd_module = import_module("verl.workers.rollout.vllm_rollout.vllm_rollout_spmd")
    vLLMRollout = getattr(rollout_spmd_module, "vLLMRollout")
    _pre_process_inputs = getattr(rollout_spmd_module, "_pre_process_inputs")
    _LEGACY_SPMD_AVAILABLE = True
except ModuleNotFoundError:
    # verl>=0.7.1 removed upstream legacy SPMD rollout symbols.
    rollout_spmd_module = _load_local_legacy_spmd_module()
    if rollout_spmd_module is not None:
        vLLMRollout = getattr(rollout_spmd_module, "vLLMRollout")
        _pre_process_inputs = getattr(rollout_spmd_module, "_pre_process_inputs")
        _LEGACY_SPMD_AVAILABLE = True
        logger.warning("Loaded legacy vLLM SPMD rollout symbols from local OneRec fork.")
    else:
        # Keep async compatibility when no local legacy module is available.
        vLLMRollout = ServerAdapter
        _pre_process_inputs = None
        _LEGACY_SPMD_AVAILABLE = False

class TwoStagevLLMRollout(vLLMRollout):
    """Generate CoT first, then beam-search item outputs."""

    def __init__(self, *args, **kwargs):
        is_async_adapter_call = {"config", "model_config", "device_mesh"}.issubset(kwargs) and not {
            "model_path",
            "tokenizer",
            "model_hf_config",
        }.issubset(kwargs)

        if _LEGACY_SPMD_AVAILABLE:
            if is_async_adapter_call:
                # Some verl>=0.7 async worker paths still instantiate this class
                # via ServerAdapter-style kwargs. Skip legacy constructor in that case.
                ServerAdapter.__init__(
                    self,
                    config=kwargs["config"],
                    model_config=kwargs["model_config"],
                    device_mesh=kwargs["device_mesh"],
                    replica_rank=kwargs.get("replica_rank", -1),
                )
                logger.warning(
                    "TwoStagevLLMRollout received async-style ctor kwargs while legacy "
                    "SPMD symbols are available; using ServerAdapter compatibility path."
                )
                return
            try:
                super().__init__(*args, **kwargs)
            except TypeError as exc:
                # Legacy OneRec vLLMRollout calls `BaseRollout.__init__()` with no args,
                # but verl>=0.7 BaseRollout now requires (config, model_config, device_mesh).
                # Patch BaseRollout.__init__ temporarily to keep legacy constructor working.
                if "BaseRollout.__init__" not in str(exc):
                    raise

                base_rollout_cls = type(self).__mro__[2]
                old_init = base_rollout_cls.__init__

                def _compat_base_init(_self, *init_args, **init_kwargs):
                    if len(init_args) >= 3:
                        _self.config = init_args[0]
                        _self.model_config = init_args[1]
                        _self.device_mesh = init_args[2]
                    elif {"config", "model_config", "device_mesh"}.issubset(init_kwargs):
                        _self.config = init_kwargs["config"]
                        _self.model_config = init_kwargs["model_config"]
                        _self.device_mesh = init_kwargs["device_mesh"]
                    # no-op for legacy no-arg super() calls
                    return None

                base_rollout_cls.__init__ = _compat_base_init
                try:
                    super().__init__(*args, **kwargs)
                finally:
                    base_rollout_cls.__init__ = old_init
            return

        # Async mode constructor for verl>=0.7.1 (ServerAdapter).
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
            preprocess_inputs=_pre_process_inputs,
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

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if not _LEGACY_SPMD_AVAILABLE:
            raise NotImplementedError(
                "TwoStagevLLMRollout.generate_sequences() uses legacy SPMD inference_engine and "
                "is not available in verl>=0.7.1 async server mode."
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
        if not _LEGACY_SPMD_AVAILABLE:
            return
        if getattr(self.config, "free_cache_engine", False):
            # Legacy SPMD path keeps a local inference engine.
            wake_tags = set(tags or [])
            if "weights" in wake_tags or "kv_cache" in wake_tags:
                self.inference_engine.wake_up()

    async def update_weights(self, weights, global_steps: int = None, **kwargs):
        """Best-effort compatibility hook for legacy sync two-stage rollout."""
        _ = (weights, global_steps, kwargs)
        # Legacy sync path updates weights through sharding manager interactions;
        # keep this as a no-op to satisfy new rollout interface.
        return

    async def release(self):
        """Lifecycle hook required by BaseRollout in verl>=0.7.x."""
        if not _LEGACY_SPMD_AVAILABLE:
            return
        if getattr(self.config, "free_cache_engine", False):
            self.inference_engine.sleep(level=1)
