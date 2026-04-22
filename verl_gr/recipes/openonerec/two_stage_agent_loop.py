"""OpenOneRec-specific async agent loop extensions for two-stage rollout."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

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
from verl.workers.rollout.replica import TokenOutput


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


@register("openonerec_two_stage_agent")
class OpenOneRecTwoStageAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop that routes repeated samples into two-stage groups."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        sampling_params = dict(sampling_params)

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        beam_width = max(1, int(sampling_params.get("stage2_beam_size", 1)))
        rollout_n = int(kwargs.get("trajectory_rollout_n", 0))
        stage1_sample_idx = rollout_n // beam_width
        beam_idx = rollout_n % beam_width
        sample_index = kwargs.get("trajectory_sample_index", -1)
        step = kwargs.get("trajectory_step", -1)
        validate = int(bool(kwargs.get("trajectory_validate", False)))

        sampling_params["stage1_sample_idx"] = stage1_sample_idx
        sampling_params["beam_idx"] = beam_idx
        sampling_params["two_stage_group_id"] = f"{step}:{validate}:{sample_index}:{stage1_sample_idx}"
        request_id = sampling_params["two_stage_group_id"]

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        extra_info = kwargs.get("extra_info")
        if extra_info is None:
            extra_info = {}
        else:
            extra_info = dict(extra_info)
        result.extra_fields["extra_info"] = extra_info
        if "generated_items" in result.extra_fields:
            result.extra_fields["extra_info"]["generated_items"] = result.extra_fields["generated_items"]

        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result


class OpenOneRecAgentLoopWorker(AgentLoopWorker):
    """Custom worker that injects two-stage rollout params without patching verl."""

    def __init__(self, *args, **kwargs):
        # Import side effect ensures the custom agent loop is registered on the worker.
        import verl_gr.recipes.openonerec.two_stage_agent_loop  # noqa: F401

        super().__init__(*args, **kwargs)

    async def generate_sequences(self, batch):
        if self.rollout_config.name != "two_stage":
            return await super().generate_sequences(batch)

        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        if batch.meta_info.get("max_tokens") is not None:
            sampling_params["max_tokens"] = batch.meta_info["max_tokens"]

        sampling_params["enable_two_stage_rollout"] = True
        sampling_params["stage1_max_tokens"] = batch.meta_info.get(
            "stage1_max_tokens",
            _read_rollout_custom_value(config, "stage1_max_tokens", config.response_length),
        )
        sampling_params["stage2_beam_size"] = batch.meta_info.get(
            "stage2_beam_size",
            _read_rollout_custom_value(config, "stage2_beam_size", 32),
        )
        sampling_params["stage2_num_tokens"] = batch.meta_info.get(
            "stage2_num_tokens",
            _read_rollout_custom_value(config, "stage2_num_tokens", 16),
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["openonerec_two_stage_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
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
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["trajectory_step"] = trajectory_info[i]["step"]
            kwargs["trajectory_sample_index"] = trajectory_info[i]["sample_index"]
            kwargs["trajectory_rollout_n"] = trajectory_info[i]["rollout_n"]
            kwargs["trajectory_validate"] = trajectory_info[i]["validate"]
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )

        outputs = await asyncio.gather(*tasks)
        output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)
        return output


class OpenOneRecAgentLoopManager(AgentLoopManager):
    """Manager that swaps in the OpenOneRec worker implementation."""

    agent_loop_workers_class = ray.remote(OpenOneRecAgentLoopWorker)
