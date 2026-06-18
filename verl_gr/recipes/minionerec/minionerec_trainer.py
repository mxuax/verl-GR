"""MiniOneRec trainer adapter."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from fnmatch import fnmatch
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.reward import extract_reward

from verl_gr.recipes.minionerec.minionerec_reward import (
    RewardPenaltyConfig,
    compute_group_training_rewards,
    normalize_sid,
)
from verl_gr.trainers.task_adapter import TrainerTaskAdapter
from verl_gr.workers.rollout.beam_config import (
    BEAM_RETURN_MODE_KEY,
    BEAM_SEARCH_PARAMS_KEY,
    BEAM_WIDTH_KEY,
)


class MiniOneRecTrainerAdapter(TrainerTaskAdapter):
    """MiniOneRec-specific trainer adapter."""

    def prepare_gen_batch(self, trainer, batch):
        return trainer._prepare_recommendation_gen_batch(batch)

    def postprocess_rewards(
        self,
        trainer,
        batch: DataProto,
        reward_batch: DataProto,
    ) -> tuple[DataProto, dict[str, Any]]:
        reward_tensor = reward_batch.batch["rm_scores"]
        if "responses" not in batch.batch or "reward_model" not in batch.non_tensor_batch:
            return reward_batch, {}

        response_token_ids = self._valid_response_token_ids(batch, trainer.tokenizer.pad_token_id)
        completions = [normalize_sid(trainer.tokenizer.decode(ids, skip_special_tokens=True)) for ids in response_token_ids]
        targets = [normalize_sid(item.get("ground_truth", "")) for item in batch.non_tensor_batch["reward_model"]]
        group_keys = self._group_keys(batch)
        penalty_cfg = self._reward_penalty_config(trainer)
        reward_parts = compute_group_training_rewards(
            completions, targets, group_keys, penalty_cfg=penalty_cfg
        )
        rule_rewards = np.array(reward_parts["rule_rewards"], dtype=np.float32)
        ranking_rewards = np.array(reward_parts["ranking_rewards"], dtype=np.float32)
        shape_penalties = np.array(reward_parts["shape_penalties"], dtype=np.float32)
        total_rewards = np.array(reward_parts["total_rewards"], dtype=np.float32)
        group_has_hit = np.array(reward_parts["group_has_hit"], dtype=np.float32)

        reward_batch.batch["rm_scores"] = self._write_sequence_rewards(
            batch=batch,
            reward_tensor=reward_tensor,
            sequence_rewards=torch.tensor(total_rewards, dtype=reward_tensor.dtype, device=reward_tensor.device),
            pad_token_id=trainer.tokenizer.pad_token_id,
        )
        return reward_batch, {
            "minionerec_rule_reward": rule_rewards.astype(object),
            "minionerec_ranking_reward": ranking_rewards.astype(object),
            "minionerec_shape_penalty": shape_penalties.astype(object),
            "minionerec_total_reward": total_rewards.astype(object),
            "minionerec_group_has_hit": group_has_hit.astype(object),
            "minionerec_invalid_sid": np.array(reward_parts["invalid_sid"], dtype=object),
            "minionerec_empty_completion": np.array(reward_parts["empty_completion"], dtype=object),
        }

    @staticmethod
    def _reward_penalty_config(trainer) -> RewardPenaltyConfig:
        """Read optional Hydra overrides under ``task.reward_penalties``."""

        defaults = RewardPenaltyConfig()
        try:
            task_cfg = trainer.config.get("task") or {}
            penalties = task_cfg.get("reward_penalties") or {}
            return RewardPenaltyConfig(
                empty_completion=float(penalties.get("empty_completion", defaults.empty_completion)),
                invalid_sid=float(penalties.get("invalid_sid", defaults.invalid_sid)),
            )
        except (AttributeError, TypeError, ValueError):
            return defaults

    def validate(self, trainer):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_uids = []
        sample_outputs = []
        sample_scores = []
        sample_ground_truths = []

        for test_data in trainer.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            val_kwargs = trainer.config.actor_rollout_ref.rollout.val_kwargs
            rollout_cfg = trainer.config.actor_rollout_ref.rollout
            rollout_custom = rollout_cfg.get("custom") or {}
            beam_width = int(rollout_custom.get(BEAM_WIDTH_KEY, val_kwargs.get("n", 1)))
            val_beam_width = int(rollout_custom.get("val_beam_width", beam_width))
            base_generations_per_prompt = int(
                rollout_custom.get("num_generations_per_prompt", max(1, int(rollout_cfg.get("n", 1)) // max(beam_width, 1)))
            )
            repeat_times = max(1, base_generations_per_prompt) * max(1, val_beam_width)
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)

            input_ids = test_batch.batch["input_ids"]
            if "raw_prompt" in test_batch.non_tensor_batch:
                sample_inputs.extend([str(v) for v in test_batch.non_tensor_batch["raw_prompt"]])
            else:
                sample_inputs.extend([trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids])
            sample_uids.extend(self._group_keys(test_batch))
            if "reward_model" in test_batch.non_tensor_batch:
                sample_ground_truths.extend(
                    [normalize_sid(item.get("ground_truth", "")) for item in test_batch.non_tensor_batch["reward_model"]]
                )

            test_gen_batch = trainer._prepare_recommendation_gen_batch(test_batch)
            meta_info = {
                **test_gen_batch.meta_info,
                "eos_token_id": trainer.tokenizer.eos_token_id,
                "pad_token_id": trainer.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": val_kwargs.do_sample,
                "validate": True,
                "global_steps": trainer.global_steps,
                BEAM_RETURN_MODE_KEY: "all_beams",
            }
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
            if trainer.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                trainer.checkpoint_manager.sleep_replicas()
                batch_reward = trainer._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                trainer.checkpoint_manager.update_weights(trainer.global_steps)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [normalize_sid(trainer.tokenizer.decode(ids, skip_special_tokens=True)) for ids in output_ids]
            sample_outputs.extend(output_texts)
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

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

            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    trainer.config.data.get("reward_fn_key", "data_source"),
                    test_batch.non_tensor_batch.get("source", test_batch.non_tensor_batch.get("data_source", ["minionerec"] * len(test_batch))),
                )
            )

        self.maybe_log_val_generations(trainer, sample_inputs, sample_outputs, sample_scores)
        val_data_dir = trainer.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self.dump_generations(
                trainer,
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                ground_truths=sample_ground_truths,
            )
        if data_source_lst:
            data_sources = np.concatenate(data_source_lst, axis=0)
        else:
            data_sources = np.array(["minionerec"] * len(sample_outputs), dtype=object)
        metric_dict = self._compute_ranking_metrics(
            data_sources=data_sources,
            sample_uids=sample_uids,
            sample_outputs=sample_outputs,
            sample_ground_truths=sample_ground_truths,
            ks=(1, 3, 5, 10, 20, 32),
        )
        metric_dict.update(
            self._compute_scalar_means(data_sources=data_sources, reward_extra_infos_dict=reward_extra_infos_dict)
        )
        return metric_dict

    @staticmethod
    def _compute_ranking_metrics(
        *,
        data_sources,
        sample_uids: list[str],
        sample_outputs: list[str],
        sample_ground_truths: list[str],
        ks: tuple[int, ...],
    ) -> dict[str, float]:
        if not sample_outputs or not sample_ground_truths or not sample_uids:
            return {}
        grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, (data_source, uid) in enumerate(zip(data_sources, sample_uids, strict=True)):
            grouped_indices[(str(data_source), str(uid))].append(idx)
        source_hits: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        source_ndcgs: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for (data_source, _uid), indices in grouped_indices.items():
            gt_sid = normalize_sid(sample_ground_truths[indices[0]])
            if gt_sid == "":
                for k in ks:
                    source_hits[data_source][k].append(0.0)
                    source_ndcgs[data_source][k].append(0.0)
                continue
            ranked_outputs = [normalize_sid(sample_outputs[idx]) for idx in indices]
            first_hit_rank = next((rank for rank, sid in enumerate(ranked_outputs) if sid == gt_sid), None)
            for k in ks:
                hit = 1.0 if first_hit_rank is not None and first_hit_rank < k else 0.0
                ndcg = 1.0 / np.log2(first_hit_rank + 2) if hit == 1.0 else 0.0
                source_hits[data_source][k].append(hit)
                source_ndcgs[data_source][k].append(float(ndcg))
        metrics = {}
        for data_source in source_hits:
            for k in ks:
                hr_val = float(np.mean(source_hits[data_source][k])) if source_hits[data_source][k] else 0.0
                ndcg_val = float(np.mean(source_ndcgs[data_source][k])) if source_ndcgs[data_source][k] else 0.0
                hr_key = f"val-aux/{data_source}/hr@{k}"
                ndcg_key = f"val-aux/{data_source}/ndcg@{k}"
                pass_key = f"val-aux/{data_source}/pass_at_{k}"
                metrics[hr_key] = hr_val
                metrics[f"{hr_key}/mean"] = hr_val
                metrics[ndcg_key] = ndcg_val
                metrics[f"{ndcg_key}/mean"] = ndcg_val
                metrics[pass_key] = hr_val
                metrics[f"{pass_key}/mean"] = hr_val
        return metrics

    @staticmethod
    def _compute_scalar_means(*, data_sources, reward_extra_infos_dict: dict[str, list]) -> dict[str, float]:
        metrics = {}
        if len(data_sources) == 0:
            return metrics
        for key, values in reward_extra_infos_dict.items():
            if not values or len(values) != len(data_sources):
                continue
            grouped: dict[str, list[float]] = defaultdict(list)
            for source, value in zip(data_sources, values, strict=True):
                if isinstance(value, (list, dict, str)):
                    continue
                try:
                    grouped[str(source)].append(float(value))
                except (TypeError, ValueError):
                    continue
            for source, grouped_values in grouped.items():
                if not grouped_values:
                    continue
                metric_key = f"val-aux/{source}/{key}/mean"
                metrics[metric_key] = float(np.mean(grouped_values))
        return metrics

    @staticmethod
    def _group_keys(batch: DataProto) -> list[Any]:
        if "uid" in batch.non_tensor_batch:
            return [str(item) for item in batch.non_tensor_batch["uid"]]
        if "index" in batch.non_tensor_batch:
            return [str(item) for item in batch.non_tensor_batch["index"]]
        return [idx for idx in range(len(batch))]

    @staticmethod
    def _response_attention_mask(batch, pad_token_id: int | None):
        responses = batch.batch["responses"]
        attention_mask = batch.batch.get("attention_mask")
        if attention_mask is not None and attention_mask.shape[-1] >= responses.shape[-1]:
            return attention_mask[:, -responses.shape[-1]:].to(dtype=torch.bool)
        if pad_token_id is None:
            return torch.ones_like(responses, dtype=torch.bool)
        return responses != pad_token_id

    @classmethod
    def _valid_response_token_ids(cls, batch, pad_token_id: int | None) -> list[list[int]]:
        responses = batch.batch["responses"]
        response_mask = cls._response_attention_mask(batch, pad_token_id)
        return [ids[mask].detach().cpu().tolist() for ids, mask in zip(responses, response_mask, strict=True)]

    @staticmethod
    def _write_sequence_rewards(batch, reward_tensor, sequence_rewards, pad_token_id: int | None):
        rewritten = torch.zeros_like(reward_tensor)
        response_mask = MiniOneRecTrainerAdapter._response_attention_mask(batch, pad_token_id)
        valid_lengths = response_mask.sum(dim=1).clamp(min=1)
        rewritten[torch.arange(rewritten.size(0), device=rewritten.device), valid_lengths - 1] = sequence_rewards
        return rewritten

    @staticmethod
    def _looks_like_sid(text: str) -> bool:
        return text.startswith("<a_") and "<b_" in text and "<c_" in text

    def evaluate_and_prune_checkpoint(self, trainer, local_global_step_folder: str, metrics=None) -> None:
        _evaluate_and_prune_checkpoint(trainer, local_global_step_folder, metrics=metrics)


MiniOneRecTrainerHooks = MiniOneRecTrainerAdapter


def _checkpoint_score_from_metrics(trainer, local_global_step_folder, metrics):
    if not metrics:
        return None
    metric_pattern = str(trainer.config.trainer.get("best_ckpt_metric", "val-aux/*/hr@20/mean"))
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


def _evaluate_and_prune_checkpoint(trainer, local_global_step_folder, metrics=None):
    if not bool(trainer.config.trainer.get("best_ckpt_prune_enable", True)):
        return
    eval_score = _checkpoint_score_from_metrics(trainer, local_global_step_folder, metrics)
    if eval_score is None:
        return
    keep_count = int(trainer.config.trainer.get("best_ckpts_to_keep", 3))
    if keep_count <= 0:
        return
    ckpt_root = os.path.abspath(str(trainer.config.trainer.default_local_dir))
    score_path = os.path.join(ckpt_root, "best_minionerec_checkpoints.json")
    os.makedirs(ckpt_root, exist_ok=True)
    records = []
    if os.path.isfile(score_path):
        try:
            with open(score_path, "r", encoding="utf-8") as handle:
                records = json.load(handle).get("checkpoints", [])
        except Exception:
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
