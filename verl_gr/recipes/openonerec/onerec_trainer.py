"""OpenOneRec-specific trainer helpers."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from fnmatch import fnmatch
from importlib import import_module

DataProto = getattr(import_module("verl"), "DataProto")
metric_utils_mod = import_module("verl.trainer.ppo.metric_utils")
protocol_mod = import_module("verl.protocol")
reward_mod = import_module("verl.trainer.ppo.reward")
np = import_module("numpy")
torch = import_module("torch")

process_validation_metrics = getattr(metric_utils_mod, "process_validation_metrics")
pad_dataproto_to_divisor = getattr(protocol_mod, "pad_dataproto_to_divisor")
unpad_dataproto = getattr(protocol_mod, "unpad_dataproto")
extract_reward = getattr(reward_mod, "extract_reward")

from verl_gr.workers.rollout.beam_config import (
    BEAM_RETURN_MODE_KEY,
    BEAM_SEARCH_PARAMS_KEY,
    BEAM_WIDTH_KEY,
    DECODE_CONFIG_KEY,
    build_two_stage_sampling_params,
    get_rollout_custom_nested_value,
)
from verl_gr.trainers.task_adapter import TrainerTaskAdapter


class ValidationGenerationsLogger:
    """Local validation generations logger for OpenOneRec.

    This avoids relying on external project forks. For tensorboard, we emit a
    compact text preview to stdout since table logging is backend-specific.
    """

    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

    @staticmethod
    def _normalize_backends(logger_backends):
        if logger_backends is None:
            return []
        if isinstance(logger_backends, str):
            return [logger_backends]
        return list(logger_backends)

    def log(self, logger_backends, samples, global_step: int) -> None:
        backends = self._normalize_backends(logger_backends)
        if not samples:
            return

        # Tensorboard does not have a standard table API in this trainer stack.
        # Keep behavior deterministic and visible via logs.
        if "tensorboard" in backends:
            preview = samples[: min(3, len(samples))]
            print(
                f"[val_generations] step={global_step} project={self.project_name} "
                f"exp={self.experiment_name} logged={len(samples)} preview={len(preview)}"
            )
            for idx, (inp, out, score) in enumerate(preview):
                inp_text = str(inp)[:160].replace("\n", "\\n")
                out_text = str(out)[:160].replace("\n", "\\n")
                print(f"[val_generations][{idx}] score={score} input='{inp_text}' output='{out_text}'")


def openonerec_dump_generations(
    trainer,
    inputs,
    outputs,
    scores,
    reward_extra_infos_dict,
    dump_path,
    ground_truths=None,
):
    """Dump rollout/validation samples as JSONL."""
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{trainer.global_steps}.jsonl")

    n = len(inputs)
    base_data = {
        "input": inputs,
        "output": outputs,
        "score": scores,
        "step": [trainer.global_steps] * n,
    }

    if ground_truths and len(ground_truths) == n:
        base_data["ground_truth"] = ground_truths

    for key, values in reward_extra_infos_dict.items():
        if len(values) == n:
            base_data[key] = values

    lines = []
    for i in range(n):
        entry = {k: v[i] for k, v in base_data.items()}
        lines.append(json.dumps(entry, ensure_ascii=False))

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Dumped generations to {filename}")


def openonerec_maybe_log_val_generations(trainer, inputs, outputs, scores):
    """Log a table of validation samples to the configured logger."""
    generations_to_log = trainer.config.trainer.get("log_val_generations", 0)
    if generations_to_log == 0:
        return

    if not hasattr(trainer, "validation_generations_logger") or trainer.validation_generations_logger is None:
        trainer.validation_generations_logger = ValidationGenerationsLogger(
            project_name=trainer.config.trainer.project_name,
            experiment_name=trainer.config.trainer.experiment_name,
        )

    samples = list(zip(inputs, outputs, scores, strict=True))
    samples.sort(key=lambda x: x[0])
    rng = np.random.RandomState(42)
    rng.shuffle(samples)
    samples = samples[:generations_to_log]
    trainer.validation_generations_logger.log(trainer.config.trainer.logger, samples, trainer.global_steps)


class OpenOneRecTrainerAdapter(TrainerTaskAdapter):
    """OpenOneRec trainer adapter preserving the existing validation helpers."""

    def prepare_gen_batch(self, trainer, batch):
        return trainer._prepare_recommendation_gen_batch(batch)

    def validate(self, trainer):
        return openonerec_validate(trainer)

    def dump_generations(
        self,
        trainer,
        inputs,
        outputs,
        scores,
        reward_extra_infos_dict,
        dump_path,
        ground_truths=None,
    ):
        return openonerec_dump_generations(
            trainer,
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
            ground_truths=ground_truths,
        )

    def maybe_log_val_generations(self, trainer, inputs, outputs, scores):
        return openonerec_maybe_log_val_generations(trainer, inputs=inputs, outputs=outputs, scores=scores)


def _extract_eval_sids(text) -> set[str]:
    if not isinstance(text, str):
        return set()

    sids = []
    for part in text.split("<|sid_begin|>"):
        if "<|sid_end|>" not in part:
            continue
        sid = part.split("<|sid_end|>", 1)[0].strip()
        if sid:
            sids.append(sid)
    return set(sids)


def _extract_eval_generation_sid(text) -> str:
    if not isinstance(text, str):
        return ""

    generation = text.strip()
    if "</think>" in generation:
        generation = generation.split("</think>")[-1].strip()
    if "<|sid_begin|>" in generation:
        # Two-stage validation responses may include CoT text before the item
        # prefix. Skip that pre-prefix segment to match the standalone evaluator.
        for part in generation.split("<|sid_begin|>")[1:]:
            if "<|sid_end|>" in part:
                sid = part.split("<|sid_end|>", 1)[0].strip()
                if sid:
                    return sid
            sid = part.strip()
            if sid:
                return sid
    return generation


def _add_pass_at_k_reward_info(
    reward_extra_infos_dict: dict[str, list],
    data_sources,
    sample_inputs: list[str],
    sample_outputs: list[str],
    sample_ground_truths: list[str],
    k: int = 32,
) -> dict[str, float]:
    if not sample_outputs or not sample_ground_truths:
        return {}

    grouped_indices = defaultdict(list)
    for idx, (data_source, prompt) in enumerate(zip(data_sources, sample_inputs, strict=True)):
        grouped_indices[(data_source, prompt)].append(idx)

    pass_at_k_values = [0.0] * len(sample_outputs)
    source_values = defaultdict(list)
    source_counts = defaultdict(lambda: [0, 0])
    for (data_source, _prompt), indices in grouped_indices.items():
        candidate_indices = indices[:k]
        gt_ids = _extract_eval_sids(sample_ground_truths[indices[0]])
        predicted = [_extract_eval_generation_sid(sample_outputs[idx]) for idx in candidate_indices]
        if source_counts[data_source][1] < 3:
            print(f"[pass_at_{k}/debug] {data_source} gt_sample={list(gt_ids)[:3]} pred_top5={predicted[:5]}")

        group_value = float(any(sid in gt_ids for sid in predicted[:k] if sid))
        source_values[data_source].append(group_value)
        source_counts[data_source][0] += int(group_value)
        source_counts[data_source][1] += 1
        for idx in indices:
            pass_at_k_values[idx] = group_value

    reward_extra_infos_dict[f"pass_at_{k}"] = pass_at_k_values
    for data_source, (hits, total) in source_counts.items():
        print(f"[pass_at_{k}/evaluator_style] {data_source}: {hits}/{total} = {hits / max(total, 1):.6f}")
    return {
        f"val-aux/{data_source}/pass_at_{k}": float(np.mean(values))
        for data_source, values in source_values.items()
        if values
    }


def openonerec_validate(trainer):
    """OpenOneRec validation override for trainer instances."""

    data_source_lst = []
    reward_extra_infos_dict: dict[str, list] = defaultdict(list)

    # Debug: print dataset sizes before validation
    print(
        f"[_validate] Starting validation. train_dataset size: {len(trainer.train_dataset)}, "
        f"val_dataset size: {len(trainer.val_dataset)}"
    )
    print(f"[_validate] actor_rollout_wg world_size: {trainer.actor_rollout_wg.world_size}")

    sample_inputs = []
    sample_outputs = []
    sample_scores = []
    sample_turns = []
    sample_ground_truths = []
    total_val_batches = len(trainer.val_dataloader)
    cumulative_raw_prompts = 0
    cumulative_expanded_requests = 0
    batch_idx = 0

    for test_data in trainer.val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)
        print(f"[Validation Debug] Batch {batch_idx}: test_batch size = {len(test_batch)}")
        raw_batch_size = len(test_batch)
        batch_idx += 1
        val_kwargs = trainer.config.actor_rollout_ref.rollout.val_kwargs
        rollout_config = trainer.config.actor_rollout_ref.rollout
        use_beam_search_val = val_kwargs.get("use_beam_search", False)
        is_two_stage_rollout_val = rollout_config.get("name") == "two_stage"
        rollout_custom = rollout_config.get("custom") or {}
        beam_width = int(
            rollout_custom.get(
                BEAM_WIDTH_KEY,
                rollout_custom.get("stage2_beam_size", 32),
            )
        )

        if is_two_stage_rollout_val and trainer.async_rollout_mode:
            repeat_times = int(val_kwargs.n) * beam_width
            print(
                "[Validation Debug] Async two-stage request expansion: "
                f"repeat_times={repeat_times} (val_n={val_kwargs.n}, beam_width={beam_width})"
            )
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
        elif not use_beam_search_val:
            test_batch = test_batch.repeat(repeat_times=val_kwargs.n, interleave=True)

        expanded_request_count = len(test_batch)
        cumulative_raw_prompts += raw_batch_size
        cumulative_expanded_requests += expanded_request_count
        print(
            "[Validation Global] "
            f"batch={batch_idx}/{total_val_batches}, raw_prompts={raw_batch_size}, "
            f"expanded_requests={expanded_request_count}, "
            f"cumulative_raw_prompts={cumulative_raw_prompts}, "
            f"cumulative_expanded_requests={cumulative_expanded_requests}"
        )

        if (
            trainer.use_rm
            and "reward_model" in test_batch[0].non_tensor_batch
            and test_batch[0].non_tensor_batch["reward_model"].get("style") == "model"
        ):
            return {}

        input_ids = test_batch.batch["input_ids"]
        input_texts = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        if "reward_model" in test_batch.non_tensor_batch:
            ground_truths = [item["ground_truth"] for item in test_batch.non_tensor_batch["reward_model"]]
        else:
            ground_truths = []

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        for key in (
            "multi_modal_data",
            "raw_prompt",
            "raw_prompt_text",
            "tools_kwargs",
            "interaction_kwargs",
            "agent_name",
            "extra_info",
        ):
            if key in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append(key)
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        # Keep reward-routing metadata in generation batch so async reward loop
        # can resolve source-specific scoring during validation.
        for key in ("source", "data_source", "reward_model", "uid"):
            if key in test_batch.non_tensor_batch and key not in test_gen_batch.non_tensor_batch:
                test_gen_batch.non_tensor_batch[key] = test_batch.non_tensor_batch[key]
        trainer._ensure_reward_routing_keys(test_gen_batch)

        meta_info = {
            "eos_token_id": trainer.tokenizer.eos_token_id,
            "pad_token_id": trainer.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": val_kwargs.do_sample,
            "validate": True,
            "global_steps": trainer.global_steps,
        }
        if is_two_stage_rollout_val:
            reasoning_max_tokens = rollout_custom.get(
                "stage1_max_tokens",
                get_rollout_custom_nested_value(
                    rollout_config,
                    (DECODE_CONFIG_KEY, "reasoning", "max_tokens"),
                    trainer.config.data.get("max_response_length", 1024),
                ),
            )
            item_max_tokens = rollout_custom.get(
                "stage2_num_tokens",
                get_rollout_custom_nested_value(
                    rollout_config,
                    (BEAM_SEARCH_PARAMS_KEY, "max_tokens"),
                    3,
                ),
            )
            meta_info["enable_two_stage_rollout"] = True
            beam_search_params = rollout_custom.get(BEAM_SEARCH_PARAMS_KEY) or {}
            if beam_search_params.get("constraint") is not None:
                meta_info["constraint"] = beam_search_params.get("constraint")
            meta_info.update(
                build_two_stage_sampling_params(
                    reasoning_max_tokens=int(reasoning_max_tokens),
                    item_max_tokens=int(item_max_tokens),
                    beam_width=int(beam_width),
                    return_all_beams=True,
                )
            )
            meta_info["max_tokens"] = trainer.config.data.get("max_response_length", 1024)
            meta_info["temperature"] = val_kwargs.get("temperature", rollout_config.get("temperature", 1.0))
            meta_info["top_p"] = val_kwargs.get("top_p", rollout_config.get("top_p", 1.0))
            meta_info["top_k"] = val_kwargs.get("top_k", rollout_config.get("top_k", -1))
            meta_info["n"] = val_kwargs.get("n", 1)
            print(f"[OneRecTrainer] Validation Two-Stage Enabled: {meta_info}")
        elif use_beam_search_val:
            meta_info["use_beam_search"] = True
            meta_info["best_of"] = val_kwargs.get("best_of", 4)
            meta_info["max_tokens"] = trainer.config.data.get("max_response_length", 16)
            meta_info["temperature"] = 0
            meta_info["n"] = val_kwargs.get("n", 1)
            meta_info[BEAM_RETURN_MODE_KEY] = "all_beams"
            print(f"[OneRecTrainer] Validation Beam Search Enabled (optimized, no repeat): {meta_info}")

        test_gen_batch.meta_info = meta_info
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
        size_divisor = (
            trainer.actor_rollout_wg.world_size
            if not trainer.async_rollout_mode
            else trainer.config.actor_rollout_ref.rollout.agent.num_workers
        )
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
        if not trainer.async_rollout_mode:
            test_output_gen_batch_padded = trainer.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
        else:
            test_output_gen_batch_padded = trainer.async_rollout_manager.generate_sequences(test_gen_batch_padded)

        if is_two_stage_rollout_val:
            beam_return_mode = test_gen_batch.meta_info.get(BEAM_RETURN_MODE_KEY, "best_only")
            n_beams = 1 if trainer.async_rollout_mode else (
                test_gen_batch.meta_info.get(BEAM_WIDTH_KEY, 1) if beam_return_mode == "all_beams" else 1
            )
            print(
                "[Validation Debug] Two-stage unpad: "
                f"original pad_size={pad_size}, beam_return_mode={beam_return_mode}, "
                f"n_beams={n_beams}, actual_pad_size={pad_size * n_beams}"
            )
            actual_pad_size = pad_size * n_beams
        elif use_beam_search_val:
            n_beams = (
                val_kwargs.get("n", 1)
            )
            print(
                "[Validation Debug] Beam search unpad: "
                f"original pad_size={pad_size}, n_beams={n_beams}, actual_pad_size={pad_size * n_beams}"
            )
            actual_pad_size = pad_size * n_beams
        else:
            actual_pad_size = pad_size
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=actual_pad_size)
        print(f"[Trainer Debug] test_output_gen_batch keys: {test_output_gen_batch.non_tensor_batch.keys()}")
        print("validation generation end")

        output_len = len(test_output_gen_batch)
        input_len = len(test_batch)
        if output_len > input_len and (use_beam_search_val or is_two_stage_rollout_val):
            expand_factor = output_len // input_len
            print(
                f"[Validation Debug] Batch {batch_idx-1}: Beam/TwoStage expansion - "
                f"input={input_len}, output={output_len}, factor={expand_factor}"
            )
            test_batch = test_batch.repeat(repeat_times=expand_factor, interleave=True)
            input_texts = [t for t in input_texts for _ in range(expand_factor)]
            if ground_truths:
                ground_truths = [t for t in ground_truths for _ in range(expand_factor)]
            print(
                f"[Validation Debug] Batch {batch_idx-1}: After expansion - "
                f"len(input_texts)={len(input_texts)}, len(test_batch)={len(test_batch)}"
            )

        before_extend = len(sample_inputs)
        sample_inputs.extend(input_texts)
        print(
            f"[Validation Debug] Batch {batch_idx-1}: Extended sample_inputs from "
            f"{before_extend} to {len(sample_inputs)} (+{len(input_texts)})"
        )
        if ground_truths:
            sample_ground_truths.extend(ground_truths)

        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        sample_outputs.extend(output_texts)
        response_lengths = [(ids != trainer.tokenizer.pad_token_id).sum().item() for ids in output_ids]
        reward_extra_infos_dict["response_length"].extend(response_lengths)

        test_batch = test_batch.union(test_output_gen_batch)
        test_batch.meta_info["validate"] = True
        print(f"[Trainer Debug] test_batch keys after union: {test_batch.non_tensor_batch.keys()}")

        if "generated_items" in test_batch.non_tensor_batch:
            print("[Trainer Debug] Moving generated_items into extra_info...")
            generated_items_arr = test_batch.non_tensor_batch["generated_items"]
            batch_size = len(generated_items_arr)
            if "extra_info" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["extra_info"] = np.array([{} for _ in range(batch_size)], dtype=object)
            extra_info_arr = test_batch.non_tensor_batch["extra_info"]
            for i in range(batch_size):
                if extra_info_arr[i] is None:
                    extra_info_arr[i] = {}
                extra_info_arr[i]["generated_items"] = generated_items_arr[i]

        reward_tensor, reward_extra_info = extract_reward(test_batch)
        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_scores.extend(scores)
        reward_extra_infos_dict["reward"].extend(scores)
        print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
        for key, values in reward_extra_info.items():
            if isinstance(values, np.ndarray):
                reward_extra_infos_dict[key].extend(values.tolist())
            elif isinstance(values, list):
                reward_extra_infos_dict[key].extend(values)
            else:
                reward_extra_infos_dict[key].append(values)
            print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

        if "__num_turns__" in test_batch.non_tensor_batch:
            sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

        reward_fn_key = trainer.config.data.get("reward_fn_key", "data_source")
        data_sources_batch = test_batch.non_tensor_batch.get(reward_fn_key)
        if data_sources_batch is None:
            data_sources_batch = test_batch.non_tensor_batch.get("source")
        if data_sources_batch is None:
            data_sources_batch = test_batch.non_tensor_batch.get("data_source")
        if data_sources_batch is None:
            data_sources_batch = ["unknown"] * reward_tensor.shape[0]
        data_source_lst.append(data_sources_batch)

    openonerec_maybe_log_val_generations(trainer, inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
    # dump generations
    val_data_dir = trainer.config.trainer.get("validation_data_dir", None)
    if val_data_dir:
        openonerec_dump_generations(
            trainer,
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=val_data_dir,
            ground_truths=sample_ground_truths,
        )

    from collections import Counter

    prompt_counts = Counter(sample_inputs)
    duplicate_prompts = {p: c for p, c in prompt_counts.items() if c > 1}
    if duplicate_prompts:
        print(f"[Validation Debug] Found {len(duplicate_prompts)} prompts having duplicates for beam search!")
        for p, c in list(duplicate_prompts.items())[:3]:
            print(f"  Prompt (truncated): '{p[:100]}...' appears {c} times")
    else:
        print(f"[Validation Debug] No duplicate prompts found. Total unique prompts: {len(prompt_counts)}")
    print(f"[Validation Debug] Total samples: {len(sample_inputs)}, Total scores: {len(sample_scores)}")

    for key_info, values in reward_extra_infos_dict.items():
        assert len(values) == 0 or len(values) == len(sample_scores), (
            f"{key_info}: len(values)={len(values)}, len(sample_scores)={len(sample_scores)}"
        )

    data_sources = np.concatenate(data_source_lst, axis=0)
    pass_at_k_metrics = _add_pass_at_k_reward_info(
        reward_extra_infos_dict=reward_extra_infos_dict,
        data_sources=data_sources,
        sample_inputs=sample_inputs,
        sample_outputs=sample_outputs,
        sample_ground_truths=sample_ground_truths,
        k=32,
    )
    if "pass_at_32" in reward_extra_infos_dict:
        for key, value in pass_at_k_metrics.items():
            data_source = key.split("/")[1] if "/" in key else "unknown"
            print(f"[pass_at_32] {data_source}: {value}")
        print(f"len reward_extra_infos_dict['pass_at_32']: {len(reward_extra_infos_dict['pass_at_32'])}")

    validation_infos_for_aggregation = {
        key: values for key, values in reward_extra_infos_dict.items() if key != "pass_at_32"
    }
    data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, validation_infos_for_aggregation)
    metric_dict = {}
    pass_at_aliases = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys())
            for metric_name, metric_val in metric2val.items():
                is_core = (
                    var_name == core_var
                    and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best", "pass"])
                    and f"@{n_max}" in metric_name
                )
                metric_sec = "val-core" if is_core else "val-aux"
                metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = metric_val
                if var_name in {"pass_at_1", "score"} and metric_name == f"best@{n_max}/mean" and n_max > 1:
                    alias = f"val-aux/{data_source}/pass_at_{n_max}/mean"
                    if var_name == "pass_at_1" or alias not in pass_at_aliases:
                        pass_at_aliases[alias] = metric_val

    metric_dict.update(pass_at_aliases)
    metric_dict.update(pass_at_k_metrics)
    metric_dict.update({f"{key}/mean": value for key, value in pass_at_k_metrics.items()})

    if len(sample_turns) > 0:
        sample_turns = np.concatenate(sample_turns)
        metric_dict["val-aux/num_turns/min"] = sample_turns.min()
        metric_dict["val-aux/num_turns/max"] = sample_turns.max()
        metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

    if "response_length" in reward_extra_infos_dict and len(reward_extra_infos_dict["response_length"]) > 0:
        response_lengths_tensor = torch.tensor(reward_extra_infos_dict["response_length"])
        metric_dict["val/response_length/mean"] = response_lengths_tensor.float().mean().item()
        metric_dict["val/response_length/max"] = response_lengths_tensor.max().item()
        metric_dict["val/response_length/min"] = response_lengths_tensor.min().item()
    return metric_dict


def _config_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _checkpoint_score_from_metrics(trainer, local_global_step_folder, metrics):
    if not metrics:
        return None

    metric_pattern = str(trainer.config.trainer.get("best_ckpt_metric", "val-aux/*/pass_at_32/mean"))
    exact_matches = []
    wildcard_matches = []
    for key, value in metrics.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if key == metric_pattern:
            exact_matches.append((key, numeric_value))
        elif fnmatch(key, metric_pattern):
            wildcard_matches.append((key, numeric_value))

    matches = exact_matches or wildcard_matches
    if not matches:
        return None

    score = sum(value for _, value in matches) / len(matches)
    return {
        "score": float(score),
        "metric": metric_pattern,
        "matched_metrics": {key: value for key, value in matches},
        "path": os.path.abspath(local_global_step_folder),
        "global_step": int(trainer.global_steps),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def openonerec_evaluate_and_prune_checkpoint(trainer, local_global_step_folder, metrics=None):
    if not _config_bool(trainer.config.trainer.get("best_ckpt_prune_enable", True), default=True):
        print("[checkpoint_eval] best_ckpt_prune_enable is false; skipping best-checkpoint pruning.")
        return

    eval_score = _checkpoint_score_from_metrics(trainer, local_global_step_folder, metrics)
    if eval_score is None:
        metric_pattern = trainer.config.trainer.get("best_ckpt_metric", "val-aux/*/pass_at_32/mean")
        print(
            f"[checkpoint_eval] No logged checkpoint metric matching {metric_pattern}; "
            f"skipping best-checkpoint pruning for {local_global_step_folder}."
        )
        return

    keep_count = int(trainer.config.trainer.get("best_ckpts_to_keep", 3))
    if keep_count <= 0:
        print("[checkpoint_eval] best_ckpts_to_keep <= 0; skipping pruning.")
        return

    ckpt_root = os.path.abspath(str(trainer.config.trainer.default_local_dir))
    score_path = os.path.join(ckpt_root, "best_pass32_checkpoints.json")
    os.makedirs(ckpt_root, exist_ok=True)

    records = []
    if os.path.isfile(score_path):
        try:
            with open(score_path, "r", encoding="utf-8") as handle:
                records = json.load(handle).get("checkpoints", [])
        except Exception as exc:
            print(f"[checkpoint_eval] Failed to read existing score file {score_path}: {exc}")
            records = []

    current_path = os.path.abspath(local_global_step_folder)
    records = [record for record in records if os.path.isdir(record.get("path", "")) and record.get("path") != current_path]
    records.append(eval_score)
    records.sort(
        key=lambda record: (float(record.get("score", float("-inf"))), int(record.get("global_step", -1))),
        reverse=True,
    )

    keep_records = records[:keep_count]
    remove_records = records[keep_count:]
    keep_paths = {record["path"] for record in keep_records}

    for record in remove_records:
        path = record.get("path")
        if path and os.path.isdir(path) and path not in keep_paths:
            print(
                f"[checkpoint_eval] Removing checkpoint outside top {keep_count}: "
                f"{path} ({record.get('metric')}={record.get('score')})"
            )
            shutil.rmtree(path, ignore_errors=True)

    latest_tracker = os.path.join(ckpt_root, "latest_checkpointed_iteration.txt")
    if keep_records:
        latest_kept_step = max(int(record["global_step"]) for record in keep_records)
        with open(latest_tracker, "w", encoding="utf-8") as handle:
            handle.write(str(latest_kept_step))

    with open(score_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metric": eval_score["metric"],
                "keep_count": keep_count,
                "checkpoints": keep_records,
            },
            handle,
            indent=2,
        )

    print(
        "[checkpoint_eval] Best checkpoints: "
        + ", ".join(f"global_step_{record['global_step']}={float(record['score']):.6f}" for record in keep_records)
    )

