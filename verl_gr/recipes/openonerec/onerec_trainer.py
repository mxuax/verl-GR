"""OpenOneRec-specific trainer helpers."""

from __future__ import annotations

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


def openonerec_validate(trainer):
    """OpenOneRec validation override for trainer instances."""

    data_source_lst = []
    reward_extra_infos_dict: dict[str, list] = defaultdict(list)
    sample_inputs = []
    sample_outputs = []
    sample_scores = []
    sample_turns = []
    sample_ground_truths = []

    for test_data in trainer.val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)
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
        elif use_beam_search_val:
            meta_info["use_beam_search"] = True
            meta_info["best_of"] = val_kwargs.get("best_of", 4)
            meta_info["max_tokens"] = trainer.config.data.get("max_response_length", 16)
            meta_info["temperature"] = 0
            meta_info["n"] = val_kwargs.get("n", 1)
            meta_info["return_all_beams"] = True

        test_gen_batch.meta_info = meta_info
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
            actual_pad_size = pad_size * n_beams
        else:
            actual_pad_size = pad_size
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=actual_pad_size)

        output_len = len(test_output_gen_batch)
        input_len = len(test_batch)
        if output_len > input_len and (use_beam_search_val or is_two_stage_rollout_val):
            expand_factor = output_len // input_len
            test_batch = test_batch.repeat(repeat_times=expand_factor, interleave=True)
            input_texts = [t for t in input_texts for _ in range(expand_factor)]
            if ground_truths:
                ground_truths = [t for t in ground_truths for _ in range(expand_factor)]

        sample_inputs.extend(input_texts)
        if ground_truths:
            sample_ground_truths.extend(ground_truths)

        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        sample_outputs.extend(output_texts)
        response_lengths = [(ids != trainer.tokenizer.pad_token_id).sum().item() for ids in output_ids]
        reward_extra_infos_dict["response_length"].extend(response_lengths)

        test_batch = test_batch.union(test_output_gen_batch)
        test_batch.meta_info["validate"] = True

        if "generated_items" in test_batch.non_tensor_batch:
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
        for key, values in reward_extra_info.items():
            if isinstance(values, np.ndarray):
                reward_extra_infos_dict[key].extend(values.tolist())
            elif isinstance(values, list):
                reward_extra_infos_dict[key].extend(values)
            else:
                reward_extra_infos_dict[key].append(values)

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

    trainer._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
    val_data_dir = trainer.config.trainer.get("validation_data_dir", None)
    if val_data_dir:
        trainer._dump_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=val_data_dir,
            ground_truths=sample_ground_truths,
        )

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

