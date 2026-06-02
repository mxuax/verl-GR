"""MiniOneRec agent loop for single-stage constrained beam rollout."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any

import numpy as np
import ray
import torch
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    RolloutTraceConfig,
    get_trajectory_info,
    register,
)
from verl.utils.ray_utils import auto_await
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


@contextmanager
def _nvtx_range(name: str):
    enabled = torch.cuda.is_available() and hasattr(torch.cuda, "nvtx")
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


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
        # Accept new HF branch modes (trainer intercepts before agent-loop generate).
        # Fall back to deterministic_beam only for truly unrecognised values.
        _beam_modes = {"deterministic_beam", "stochastic_constrained"}
        _hf_modes = {"hf_constrained_beam_sample", "hf_constrained_beam_eval"}
        if decode_mode in _hf_modes:
            decode_mode = "deterministic_beam"  # safe fallback for legacy path
        elif decode_mode not in _beam_modes:
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
    """Manager that swaps in the MiniOneRec constrained-beam worker.

    Overrides ``generate_sequences`` to route HF decode modes
    (``hf_constrained_beam_sample`` / ``hf_constrained_beam_eval``)
    directly to the actor FSDP model, bypassing the per-token Python
    beam backend.  All other decode modes fall through to the legacy
    agent-loop + vLLM path.
    """

    agent_loop_workers_class = ray.remote(MiniOneRecConstrainedBeamAgentLoopWorker)

    # ------------------------------------------------------------------
    # vLLM-free init: MiniOneRec routes all generation through HF
    # model.generate(), so vLLM engines are never needed.
    # ------------------------------------------------------------------

    async def _initialize_llm_servers(self):
        """Skip vLLM engine creation — MiniOneRec uses HF generate exclusively.

        .. warning::
           This only works when ``checkpoint_engine.backend`` is ``"naive"``
           (the default).  Non-naive backends require ``rollout_replicas`` to
           build a process group for weight sync and will fail here.
        """
        self.server_handles = []
        self.server_addresses = []
        self.rollout_replicas = []

    async def _init_global_load_balancer(self) -> None:
        """Skip load balancer — never used by HF-only MiniOneRec path."""
        self.global_load_balancer = None

    async def _init_agent_loop_workers(self):
        """Skip agent-loop workers — never dispatched by HF-only MiniOneRec path."""
        self.agent_loop_workers = []

    # ------------------------------------------------------------------

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        if not self._should_route_to_hf(prompts):
            if not self.agent_loop_workers:
                raise RuntimeError(
                    "MiniOneRec agent-loop workers are empty (vLLM disabled), "
                    "but the current decode mode is not routed to HF. "
                    "Set decode_mode_train/decode_mode_val to hf_constrained_beam_* "
                    "in rollout.custom to use the HF-only path."
                )
            return await super().generate_sequences(prompts)
        return await self._hf_generate_sequences(prompts)

    # ------------------------------------------------------------------
    # HF routing helpers
    # ------------------------------------------------------------------

    def _resolve_hf_decode_mode(self, *, is_validate: bool) -> str:
        custom = getattr(self.rollout_config, "custom", {}) or {}
        if hasattr(custom, "items"):
            custom = dict(custom.items())
        if isinstance(custom, dict):
            key = "decode_mode_val" if is_validate else "decode_mode_train"
            default = "hf_constrained_beam_eval" if is_validate else "hf_constrained_beam_sample"
            return str(custom.get(key, default)).strip().lower()
        return ""

    def _should_route_to_hf(self, prompts: DataProto) -> bool:
        """Check whether the current request should use the HF branch.

        Both training and validation are routed through HF when the
        corresponding decode modes are set — MiniOneRec never needs vLLM.
        """
        is_validate = bool(prompts.meta_info.get("validate", False))
        decode_mode = self._resolve_hf_decode_mode(is_validate=is_validate)
        if is_validate:
            return decode_mode == "hf_constrained_beam_eval"
        return decode_mode == "hf_constrained_beam_sample"

    @staticmethod
    def _extract_prompt_groups(prompts: DataProto, *, rows_per_group: int) -> tuple[list[str], list[int]]:
        raw_prompt_text = list(prompts.non_tensor_batch.get("raw_prompt_text", []))
        if not raw_prompt_text:
            raw_prompt_text = list(prompts.non_tensor_batch.get("raw_prompt", []))
        if not raw_prompt_text:
            return [], []
        if rows_per_group <= 0:
            raise ValueError(f"rows_per_group must be > 0, got {rows_per_group}")

        group_ids = None
        for key in ("uid", "index"):
            values = prompts.non_tensor_batch.get(key)
            if values is not None and len(values) == len(raw_prompt_text):
                group_ids = [str(v) for v in values]
                break
        if group_ids is None:
            group_ids = [str(v) for v in raw_prompt_text]

        unique_texts: list[str] = []
        group_sizes: list[int] = []
        run_start = 0
        while run_start < len(group_ids):
            run_end = run_start + 1
            while run_end < len(group_ids) and group_ids[run_end] == group_ids[run_start]:
                run_end += 1
            run_len = run_end - run_start
            if run_len % rows_per_group != 0:
                raise RuntimeError(
                    f"Prompt group '{group_ids[run_start]}' has {run_len} rows, "
                    f"which is not divisible by expected beam group size {rows_per_group}."
                )
            num_groups = run_len // rows_per_group
            prompt_text = str(raw_prompt_text[run_start])
            unique_texts.extend([prompt_text] * num_groups)
            group_sizes.extend([rows_per_group] * num_groups)
            run_start = run_end
        return unique_texts, group_sizes

    async def _hf_generate_sequences(self, prompts: DataProto) -> DataProto:
        """Route generation to ``actor_rollout_wg.hf_constrained_beam_generate``."""
        from verl_gr.trainers.rl_trainer import _get_constraint_info_file

        is_validate = bool(prompts.meta_info.get("validate", False))
        decode_mode = self._resolve_hf_decode_mode(is_validate=is_validate)
        custom = getattr(self.rollout_config, "custom", {}) or {}
        if hasattr(custom, "items"):
            custom = dict(custom.items())
        beam_width = int(custom.get("beam_width", 16)) if isinstance(custom, dict) else 16
        val_beam_width = int(custom.get("val_beam_width", beam_width)) if isinstance(custom, dict) else beam_width
        n_beams = val_beam_width if is_validate else beam_width
        info_file = _get_constraint_info_file(self.rollout_config)

        # Training: 128 (original max_completion_length).  Validation: 256 (evaluate.sh).
        train_max_new_tokens = 128
        val_max_new_tokens = int(getattr(self.rollout_config, "response_length", 256) or 256)
        max_new_tokens = val_max_new_tokens if is_validate else train_max_new_tokens

        unique_texts, group_sizes = self._extract_prompt_groups(prompts, rows_per_group=n_beams)
        n_unique = len(unique_texts)
        if n_unique == 0:
            return prompts

        # Collect pre-tokenized prompt IDs (already truncated in __getitem__)
        raw_prompt_ids_list = [
            list(int(x) for x in ids) if hasattr(ids, "__iter__") else []
            for ids in prompts.non_tensor_batch.get("raw_prompt_ids", [])
        ]
        # Get the first prompt_ids per unique group (same grouping as _extract_prompt_groups)
        unique_prompt_ids: list[list[int]] = []
        if raw_prompt_ids_list:
            group_ids = None
            for key in ("uid", "index"):
                values = prompts.non_tensor_batch.get(key)
                if values is not None and len(values) == len(raw_prompt_ids_list):
                    group_ids = [str(v) for v in values]
                    break
            if group_ids is not None:
                run_start = 0
                while run_start < len(group_ids):
                    unique_prompt_ids.append(raw_prompt_ids_list[run_start])
                    run_end = run_start + 1
                    while run_end < len(group_ids) and group_ids[run_end] == group_ids[run_start]:
                        run_end += 1
                    run_start = run_end

        meta_info = {
            "beam_width": beam_width,
            "val_beam_width": val_beam_width,
            "do_sample": decode_mode == "hf_constrained_beam_sample",
            "info_file": info_file,
            "temperature": float(getattr(self.rollout_config, "temperature", 1.0)),
            "max_new_tokens": max_new_tokens,
            "validate": is_validate,
        }

        # Pass pre-tokenized prompt IDs when available (avoids re-tokenization)
        if unique_prompt_ids and len(unique_prompt_ids) == n_unique:
            meta_info["prompt_token_ids"] = unique_prompt_ids

        # Call actor_rollout_wg via Ray (non-blocking I/O friendly — Ray RPC
        # releases the GIL so this does not stall the event loop).
        with _nvtx_range("gen.generate"):
            result = self.worker_group.hf_constrained_beam_generate(unique_texts, meta_info)

        # Reassemble per-rank prompt shards back into original prompt-group order.
        ordered_response_groups: list[list[list[int]] | None] = [None] * n_unique
        if isinstance(result, list):
            for rank_out in result:
                if isinstance(rank_out, dict):
                    prompt_indices = list(rank_out.get("prompt_indices", []))
                    response_groups = list(rank_out.get("response_ids", []))
                    for prompt_idx, responses_for_prompt in zip(prompt_indices, response_groups, strict=True):
                        ordered_response_groups[int(prompt_idx)] = [list(r) for r in responses_for_prompt]
        elif isinstance(result, dict):
            prompt_indices = list(result.get("prompt_indices", []))
            response_groups = list(result.get("response_ids", []))
            for prompt_idx, responses_for_prompt in zip(prompt_indices, response_groups, strict=True):
                ordered_response_groups[int(prompt_idx)] = [list(r) for r in responses_for_prompt]

        if any(group is None for group in ordered_response_groups):
            missing = [str(i) for i, group in enumerate(ordered_response_groups) if group is None]
            raise RuntimeError(f"HF constrained beam returned incomplete prompt shards: missing {', '.join(missing)}")

        all_resp_ids = [resp for group in ordered_response_groups for resp in (group or [])]
        n_total = len(all_resp_ids)
        if n_total == 0:
            return prompts
        expected_total = sum(group_sizes)
        if expected_total != len(prompts):
            raise RuntimeError(
                f"HF prompt-group reconstruction expected {len(prompts)} rows, "
                f"but grouped input collapsed to {expected_total}."
            )
        if n_total != expected_total:
            raise RuntimeError(
                f"HF constrained beam returned {n_total} sequences, "
                f"but input batch expects {expected_total} rows."
            )

        # Assemble DataProto with full tensor fields
        max_resp = max(len(r) for r in all_resp_ids) if all_resp_ids else 1

        # ensure dense (B, L) tensors — remove_padding may produce NestedTensors
        def _to_dense(t: torch.Tensor) -> torch.Tensor:
            if t.is_nested:
                return t.to_padded_tensor(padding=0.0)
            return t

        device = prompts.batch["input_ids"].device
        pad_id = int(prompts.meta_info.get("pad_token_id", 0) or 0)
        responses = torch.full((n_total, max_resp), pad_id, dtype=torch.long, device=device)
        for i, r in enumerate(all_resp_ids):
            if r:
                responses[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)

        prompt_ids_exp = _to_dense(prompts.batch["input_ids"])
        attn_exp = _to_dense(prompts.batch["attention_mask"])
        pos_exp = _to_dense(prompts.batch["position_ids"])

        # sanity: input tensor rows should match n_unique (or repeat-factor adjusted)
        if prompt_ids_exp.shape[0] != n_total:
            if prompt_ids_exp.shape[0] == n_unique:
                rep = n_total // n_unique
                prompt_ids_exp = prompt_ids_exp.repeat_interleave(rep, dim=0)
                attn_exp = attn_exp.repeat_interleave(rep, dim=0)
                pos_exp = pos_exp.repeat_interleave(rep, dim=0)
            else:
                raise RuntimeError(
                    f"Prompt tensor batch size {prompt_ids_exp.shape[0]} "
                    f"does not match response count {n_total}."
                )

        resp_mask = torch.ones(n_total, max_resp, dtype=attn_exp.dtype, device=device)
        eos_id = int(prompts.meta_info.get("eos_token_id", 0) or 0)
        for i, r in enumerate(all_resp_ids):
            for j, tid in enumerate(r):
                if eos_id and tid == eos_id:
                    resp_mask[i, j+1:] = 0
                    break
            if len(r) < max_resp:
                resp_mask[i, len(r):] = 0

        input_ids = torch.cat([prompt_ids_exp, responses], dim=1)
        attention_mask = torch.cat([attn_exp, resp_mask], dim=1)
        last_pos = pos_exp[:, -1:]
        delta_pos = torch.arange(1, max_resp + 1, device=responses.device).unsqueeze(0)
        position_ids = torch.cat([pos_exp, last_pos + delta_pos], dim=1)

        out_meta = dict(prompts.meta_info)
        out_meta.setdefault("timing", {})
        out = DataProto.from_dict(
            tensors={
                "prompts": prompt_ids_exp,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            meta_info=out_meta,
        )
        for key, arr in prompts.non_tensor_batch.items():
            out.non_tensor_batch[key] = arr
        # verl training loop (ray_trainer.py:1404) expects multi_modal_inputs in every batch.
        # MiniOneRec is a text-only task, so supply an empty placeholder.
        if "multi_modal_inputs" not in out.non_tensor_batch:
            out.non_tensor_batch["multi_modal_inputs"] = np.array([{} for _ in range(n_total)], dtype=object)
        return out
