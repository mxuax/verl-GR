"""Rank-GRPO trainer adapter and validation helpers."""

from __future__ import annotations

import math
import uuid
from collections import defaultdict

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.reward import extract_reward

from verl_gr.trainers.task_adapter import TrainerTaskAdapter

__all__ = ["RankGRPOTrainerAdapter", "rankgrpo_validate"]


class RankGRPOTrainerAdapter(TrainerTaskAdapter):
    def prepare_gen_batch(self, trainer, batch: DataProto) -> DataProto:
        return trainer._prepare_recommendation_gen_batch(batch)

    def validate(self, trainer):
        return rankgrpo_validate(trainer)

    def maybe_log_val_generations(self, trainer, inputs, outputs, scores):
        ground_truths = getattr(trainer, "_rankgrpo_preview_ground_truths", None)
        _print_rankgrpo_val_generation_preview(trainer, inputs, outputs, scores, ground_truths=ground_truths)
        return super().maybe_log_val_generations(trainer, inputs, outputs, scores)


def _print_rankgrpo_val_generation_preview(trainer, inputs, outputs, scores, ground_truths=None) -> None:
    generations_to_log = trainer.config.trainer.get("log_val_generations", 0)
    if generations_to_log == 0:
        return

    if ground_truths is not None and len(ground_truths) == len(scores):
        samples = list(zip(inputs, outputs, scores, ground_truths, strict=True))
    else:
        samples = [(inp, out, score, None) for inp, out, score in zip(inputs, outputs, scores, strict=True)]
    samples.sort(key=lambda item: item[0])
    rng = np.random.RandomState(42)
    rng.shuffle(samples)
    preview = samples[: min(generations_to_log, len(samples))]
    print(
        f"[val_generations] step={trainer.global_steps} project={trainer.config.trainer.project_name} "
        f"exp={trainer.config.trainer.experiment_name} logged={min(generations_to_log, len(samples))} "
        f"preview={len(preview)}"
    )
    for idx, (inp, out, score, gt) in enumerate(preview):
        inp_text = str(inp).replace("\n", "\\n")
        out_text = str(out).replace("\n", "\\n")
        gt_text = str(gt).replace("\n", "\\n")
        print(f"[val_generations][{idx}] score={score} ground_truth='{gt_text}' input='{inp_text}' output='{out_text}'")


def rankgrpo_validate(trainer):
    from verl_gr.trainers.rl_trainer import apply_kl_penalty, compute_advantage, compute_response_mask

    data_source_lst = []
    reward_extra_infos_dict: dict[str, list] = defaultdict(list)
    eval_loss_values: list[tuple[float, int]] = []

    sample_inputs = []
    sample_outputs = []
    sample_gts = []
    sample_scores = []
    sample_turns = []
    sample_uids = []

    val_kwargs = trainer.config.actor_rollout_ref.rollout.val_kwargs
    for test_data in trainer.val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)
        if "uid" not in test_batch.non_tensor_batch:
            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            )

        test_batch = test_batch.repeat(repeat_times=val_kwargs.n, interleave=True)
        ground_truths = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch]
        sample_gts.extend(ground_truths)

        test_gen_batch = trainer._get_gen_batch(test_batch)
        test_gen_batch.meta_info = {
            "eos_token_id": trainer.tokenizer.eos_token_id,
            "pad_token_id": trainer.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": val_kwargs.do_sample,
            "validate": True,
            "global_steps": trainer.global_steps,
        }

        size_divisor = trainer.config.actor_rollout_ref.rollout.agent.num_workers
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
        test_output_gen_batch_padded = trainer.async_rollout_manager.generate_sequences(test_gen_batch_padded)
        trainer.checkpoint_manager.sleep_replicas()
        if trainer.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
            batch_reward = trainer._compute_reward_colocate(test_output_gen_batch_padded)
            test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)

        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        sample_outputs.extend(output_texts)

        test_batch = test_batch.union(test_output_gen_batch)
        test_batch.meta_info["validate"] = True
        test_batch.meta_info["temperature"] = trainer.config.actor_rollout_ref.rollout.temperature

        input_ids = test_batch.batch["prompts"]
        input_texts = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        sample_inputs.extend(input_texts)
        sample_uids.extend(test_batch.non_tensor_batch["uid"])

        reward_tensor, reward_extra_info = extract_reward(test_batch)
        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_scores.extend(scores)

        reward_extra_infos_dict["reward"].extend(scores)
        for key, values in reward_extra_info.items():
            reward_extra_infos_dict.setdefault(key, [])
            if isinstance(values, np.ndarray):
                reward_extra_infos_dict[key].extend(values.tolist())
            else:
                reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

        if "__num_turns__" in test_batch.non_tensor_batch:
            sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

        data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        if "response_mask" not in test_batch.batch.keys():
            test_batch.batch["response_mask"] = compute_response_mask(test_batch)
        test_batch.meta_info["global_token_num"] = torch.sum(test_batch.batch["attention_mask"], dim=-1).tolist()

        old_log_prob, _ = trainer._compute_old_log_prob(test_batch)
        old_log_prob.batch.pop("entropys", None)
        test_batch = test_batch.union(old_log_prob)
        if trainer.use_reference_policy:
            ref_log_prob = trainer._compute_ref_log_prob(test_batch)
            test_batch = test_batch.union(ref_log_prob)

        test_batch.batch["token_level_scores"] = reward_tensor
        if reward_extra_info:
            test_batch.non_tensor_batch.update({key: np.array(value) for key, value in reward_extra_info.items()})
        if trainer.config.algorithm.use_kl_in_reward:
            test_batch, _ = apply_kl_penalty(
                test_batch,
                kl_ctrl=trainer.kl_ctrl_in_reward,
                kl_penalty=trainer.config.algorithm.kl_penalty,
            )
        else:
            test_batch.batch["token_level_rewards"] = test_batch.batch["token_level_scores"]

        test_batch = compute_advantage(
            test_batch,
            adv_estimator=trainer.config.algorithm.adv_estimator,
            gamma=trainer.config.algorithm.gamma,
            lam=trainer.config.algorithm.lam,
            num_repeat=trainer.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=trainer.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=trainer.config.algorithm,
            tokenizer=trainer.tokenizer,
        )
        # Capture rank_reward_sum / rank_reward_mean from the advantage
        # computation as a fallback when extract_reward() didn't provide
        # them (e.g. reward.reward_model.enable=False).  If extract_reward
        # already populated these keys we must NOT double-accumulate.
        rank_sums = test_batch.non_tensor_batch.get("rank_reward_sum")
        rank_means = test_batch.non_tensor_batch.get("rank_reward_mean")
        if rank_sums is not None and "rank_reward_sum" not in reward_extra_infos_dict:
            reward_extra_infos_dict["rank_reward_sum"] = []
            reward_extra_infos_dict["rank_reward_mean"] = []
            reward_extra_infos_dict["rank_reward_sum"].extend(
                rank_sums.tolist() if isinstance(rank_sums, np.ndarray) else list(rank_sums)
            )
            if rank_means is not None:
                reward_extra_infos_dict["rank_reward_mean"].extend(
                    rank_means.tolist() if isinstance(rank_means, np.ndarray) else list(rank_means)
                )

        eval_actor_metrics = trainer._compute_eval_actor_metrics(test_batch)
        eval_loss = eval_actor_metrics.get("loss")
        if eval_loss is not None and math.isfinite(trainer._as_float(eval_loss, default=float("nan"))):
            eval_loss_values.append((float(eval_loss), int(len(rank_sums) if rank_sums is not None else reward_tensor.shape[0])))
        trainer.checkpoint_manager.update_weights(trainer.global_steps)

    trainer._rankgrpo_preview_ground_truths = sample_gts
    try:
        trainer._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
    finally:
        trainer._rankgrpo_preview_ground_truths = None

    val_data_dir = trainer.config.trainer.get("validation_data_dir", None)
    if val_data_dir:
        trainer._dump_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=val_data_dir,
            ground_truths=sample_gts,
        )

    for key_info, lst in reward_extra_infos_dict.items():
        assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

    data_sources = np.concatenate(data_source_lst, axis=0)
    metric_dict = trainer._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
    _add_rankgrpo_eval_aliases(metric_dict)
    eval_loss = trainer._mean_metric(eval_loss_values)
    if eval_loss is not None:
        metric_dict["eval/loss"] = eval_loss
    return metric_dict


def _add_rankgrpo_eval_aliases(metric_dict: dict[str, float]) -> None:
    """Expose Rank-GRPO validation metrics under the reference TensorBoard names."""

    reward = _select_rankgrpo_mean_metric(metric_dict, "rank_rewards")
    if reward is not None:
        metric_dict["eval/reward"] = reward

    reward_total = _select_rankgrpo_mean_metric(metric_dict, "rank_reward_sum")
    if reward_total is not None:
        metric_dict["eval/reward_total"] = reward_total


def _select_rankgrpo_mean_metric(metric_dict: dict[str, float], metric_name: str) -> float | None:
    prefix = f"val-aux/rankgrpo/{metric_name}/mean@"
    candidates: list[tuple[int, float]] = []
    for key, value in metric_dict.items():
        if not key.startswith(prefix):
            continue
        try:
            n_responses = int(key.removeprefix(prefix))
        except ValueError:
            continue
        candidates.append((n_responses, value))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
