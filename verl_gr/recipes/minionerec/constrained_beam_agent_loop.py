"""MiniOneRec agent loop for single-stage constrained beam rollout."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import ray
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    RolloutTraceConfig,
    get_trajectory_info,
    register,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput
from verl_gr.workers.rollout.beam_config import (
    BEAM_GROUP_ID_KEY,
    BEAM_INDEX_KEY,
    BEAM_RETURN_MODE_KEY,
    BEAM_SEARCH_PARAMS_KEY,
    BEAM_WIDTH_KEY,
    get_rollout_custom_nested_value,
    get_rollout_custom_value,
)


@register("minionerec_constrained_beam_agent")
class MiniOneRecConstrainedBeamAgentLoop(SingleTurnAgentLoop):
    """Tokenize MiniOneRec's plain prompt and request one constrained beam."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        sampling_params = dict(sampling_params)
        raw_prompt_text = kwargs.get("raw_prompt_text")
        if raw_prompt_text is None:
            raise ValueError("MiniOneRecConstrainedBeamAgentLoop requires raw_prompt_text.")

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(str(raw_prompt_text), add_special_tokens=False),
        )
        prompt_ids = normalize_token_ids(prompt_ids)

        beam_width = max(1, int(sampling_params.get(BEAM_WIDTH_KEY, 1)))
        rollout_n = int(kwargs.get("trajectory_rollout_n", 0))
        beam_index = rollout_n % beam_width
        sample_index = kwargs.get("trajectory_sample_index", -1)
        step = kwargs.get("trajectory_step", -1)
        validate = int(bool(kwargs.get("trajectory_validate", False)))
        beam_params = sampling_params.get(BEAM_SEARCH_PARAMS_KEY) or {}
        decode_mode = str(beam_params.get("decode_mode", "deterministic_beam")).strip().lower()

        sampling_params[BEAM_INDEX_KEY] = beam_index
        if decode_mode == "stochastic_constrained" and not validate:
            sampling_params[BEAM_GROUP_ID_KEY] = f"{step}:{validate}:{sample_index}:{rollout_n}"
        else:
            sampling_params[BEAM_GROUP_ID_KEY] = f"{step}:{validate}:{sample_index}"
        request_id = sampling_params[BEAM_GROUP_ID_KEY]

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
                video_data=None,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_mask = [1] * len(output.token_ids)
        extra_fields = dict(output.extra_fields or {})
        extra_info = kwargs.get("extra_info")
        extra_fields["extra_info"] = dict(extra_info) if extra_info is not None else {}
        if "generated_items" in extra_fields:
            extra_fields["extra_info"]["generated_items"] = extra_fields["generated_items"]
        if "_beam_index" in extra_fields:
            extra_fields["extra_info"]["_beam_index"] = extra_fields["_beam_index"]
        extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=None,
            multi_modal_data={},
            num_turns=1,
            metrics=metrics,
            extra_fields=extra_fields,
        )


class MiniOneRecConstrainedBeamAgentLoopWorker(AgentLoopWorker):
    """Inject single-stage constrained beam params."""

    def __init__(self, *args, **kwargs):
        import verl_gr.recipes.minionerec.constrained_beam_agent_loop  # noqa: F401

        super().__init__(*args, **kwargs)

    async def generate_sequences(self, batch):
        if self.rollout_config.name != "constrained_beam":
            return await super().generate_sequences(batch)

        config = self.rollout_config
        beam_search_params = dict(batch.meta_info.get(BEAM_SEARCH_PARAMS_KEY, {}))
        beam_width = int(
            batch.meta_info.get(
                BEAM_WIDTH_KEY,
                get_rollout_custom_value(config, BEAM_WIDTH_KEY, get_rollout_custom_value(config, "beam_size", 20)),
            )
        )
        item_max_tokens = int(
            beam_search_params.get(
                "max_tokens",
                get_rollout_custom_nested_value(config, (BEAM_SEARCH_PARAMS_KEY, "max_tokens"), config.response_length),
            )
        )
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
            max_tokens=item_max_tokens,
            enable_constrained_beam_rollout=True,
            **{
                BEAM_WIDTH_KEY: beam_width,
                BEAM_RETURN_MODE_KEY: batch.meta_info.get(BEAM_RETURN_MODE_KEY, "best_only"),
                BEAM_SEARCH_PARAMS_KEY: beam_search_params,
            },
        )
        if batch.meta_info.get("constraint") is not None:
            sampling_params["constraint"] = batch.meta_info["constraint"]
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
        rollout_custom = config.get("custom") or {}
        decode_mode_train = str(rollout_custom.get("decode_mode_train", "stochastic_constrained")).strip().lower()
        decode_mode_val = str(rollout_custom.get("decode_mode_val", "deterministic_beam")).strip().lower()
        decode_mode = decode_mode_val if batch.meta_info.get("validate", False) else decode_mode_train
        if decode_mode not in {"deterministic_beam", "stochastic_constrained"}:
            decode_mode = "deterministic_beam"
        sampling_params[BEAM_SEARCH_PARAMS_KEY]["decode_mode"] = decode_mode
        sampling_params[BEAM_SEARCH_PARAMS_KEY]["disable_cache_in_train"] = bool(
            rollout_custom.get("disable_cache_in_train", True)
        )

        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["minionerec_constrained_beam_agent"] * len(batch), dtype=object)

        index = batch.non_tensor_batch["index"] if "index" in batch.non_tensor_batch else np.arange(len(batch))
        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist())
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )
        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["trajectory_step"] = trajectory_info[i]["step"]
            kwargs["trajectory_sample_index"] = trajectory_info[i]["sample_index"]
            kwargs["trajectory_rollout_n"] = trajectory_info[i]["rollout_n"]
            kwargs["trajectory_validate"] = trajectory_info[i]["validate"]
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=i in traced_indices, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)
        return self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)


class MiniOneRecConstrainedBeamAgentLoopManager(AgentLoopManager):
    """Manager that swaps in the MiniOneRec constrained-beam worker."""

    agent_loop_workers_class = ray.remote(MiniOneRecConstrainedBeamAgentLoopWorker)
