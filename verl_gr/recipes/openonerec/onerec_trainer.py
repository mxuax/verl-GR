"""OpenOneRec-specific trainer helpers."""

from __future__ import annotations

import json
import os
from collections import defaultdict
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
    batch_idx = 0

    for test_data in trainer.val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)
        print(f"[Validation Debug] Batch {batch_idx}: test_batch size = {len(test_batch)}")
        batch_idx += 1
        val_kwargs = trainer.config.actor_rollout_ref.rollout.val_kwargs
        rollout_config = trainer.config.actor_rollout_ref.rollout
        use_beam_search_val = val_kwargs.get("use_beam_search", False)
        is_two_stage_rollout_val = rollout_config.get("name") == "two_stage"

        if not use_beam_search_val:
            test_batch = test_batch.repeat(repeat_times=val_kwargs.n, interleave=True)

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
        for key in ("multi_modal_data", "raw_prompt", "tools_kwargs", "interaction_kwargs", "agent_name"):
            if key in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append(key)
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        # Keep reward-routing metadata in generation batch so async reward loop
        # can resolve source-specific scoring during validation.
        for key in ("source", "data_source", "reward_model", "extra_info", "uid"):
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
        rollout_custom = rollout_config.get("custom") or {}

        if is_two_stage_rollout_val:
            meta_info["enable_two_stage_rollout"] = True
            meta_info["stage1_max_tokens"] = rollout_custom.get(
                "stage1_max_tokens",
                trainer.config.data.get("max_response_length", 1024),
            )
            meta_info["stage2_beam_size"] = rollout_custom.get("stage2_beam_size", 32)
            meta_info["stage2_num_tokens"] = rollout_custom.get("stage2_num_tokens", 3)
            meta_info["max_tokens"] = trainer.config.data.get("max_response_length", 1024)
            meta_info["use_beam_search"] = False
            meta_info["n"] = val_kwargs.get("n", 1)
            meta_info["return_all_beams"] = True
            print(f"[OneRecTrainer] Validation Two-Stage Enabled: {meta_info}")
        elif use_beam_search_val:
            meta_info["use_beam_search"] = True
            meta_info["best_of"] = val_kwargs.get("best_of", 4)
            meta_info["max_tokens"] = trainer.config.data.get("max_response_length", 16)
            meta_info["temperature"] = 0
            meta_info["n"] = val_kwargs.get("n", 1)
            meta_info["return_all_beams"] = True
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

        if use_beam_search_val or is_two_stage_rollout_val:
            n_beams = (
                rollout_custom.get("stage2_beam_size", 2)
                if is_two_stage_rollout_val
                else val_kwargs.get("n", 1)
            )
            if is_two_stage_rollout_val:
                print(
                    "[Validation Debug] Two-stage unpad: "
                    f"original pad_size={pad_size}, stage2_beam_size={n_beams}, actual_pad_size={pad_size * n_beams}"
                )
            else:
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
        print(f"[Validation Debug] Found {len(duplicate_prompts)} duplicate prompts!")
        for p, c in list(duplicate_prompts.items())[:3]:
            print(f"  Prompt (truncated): '{p[:100]}...' appears {c} times")
    else:
        print(f"[Validation Debug] No duplicate prompts found. Total unique prompts: {len(prompt_counts)}")
    print(f"[Validation Debug] Total samples: {len(sample_inputs)}, Total scores: {len(sample_scores)}")

    data_sources = np.concatenate(data_source_lst, axis=0)
    data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
    metric_dict = {}
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

