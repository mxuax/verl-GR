"""RL trainer extensions for verl-GR with bridged ray-trainer API."""

import math
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as RayPPOTrainerBase
from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager
from verl.utils.torch_functional import masked_mean

from verl_gr.recipes.openonerec.onerec_trainer import (
    openonerec_dump_generations,
    openonerec_maybe_log_val_generations,
    openonerec_validate,
)
from verl_gr.workers.rollout.beam_config import (
    BEAM_SEARCH_PARAMS_KEY,
    BEAM_WIDTH_KEY,
    DECODE_CONFIG_KEY,
    build_two_stage_sampling_params,
    get_rollout_custom_nested_value,
)

AdvantageEstimator = getattr(core_algos, "AdvantageEstimator")
_RANKGRPO_TOKENIZER = None


def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty: str = "kl"):
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    return data, {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _rankgrpo_enabled(config: Any) -> bool:
    rank_cfg = _cfg_get(config, "rank_grpo", None)
    return bool(_cfg_get(rank_cfg, "enable", False))


def _decode_response_texts(responses: torch.Tensor, response_mask: torch.Tensor, tokenizer) -> list[str]:
    texts: list[str] = []
    for ids, mask in zip(responses, response_mask, strict=True):
        valid_ids = ids[mask.bool()].detach().cpu().tolist()
        texts.append(tokenizer.decode(valid_ids, skip_special_tokens=True))
    return texts


def _segment_rank_tokens(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    *,
    rank_separator: str,
    rec_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign natural rank ids to response tokens using newline-like separators."""

    device = responses.device
    batch_size, response_length = responses.size()
    seg_ids = torch.full((batch_size, response_length), -1, dtype=torch.long, device=device)
    try:
        separator_ids = tokenizer.encode(rank_separator, add_special_tokens=False)
    except Exception:
        separator_ids = []
    single_separator_id = int(separator_ids[0]) if len(separator_ids) == 1 else None

    for row_idx in range(batch_size):
        valid = int(response_mask[row_idx].sum().item())
        item_id = 0
        for token_idx in range(valid):
            seg_ids[row_idx, token_idx] = item_id
            token_id = int(responses[row_idx, token_idx].item())
            separator_count = 0
            if single_separator_id is not None and token_id == single_separator_id:
                separator_count = 1
            else:
                try:
                    piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False)
                except TypeError:
                    piece = tokenizer.decode([token_id])
                except Exception:
                    piece = ""
                separator_count = str(piece).count(rank_separator)
            if separator_count > 0:
                item_id += separator_count

    rank_token_mask = response_mask.bool() & (seg_ids >= 0) & (seg_ids < rec_num)
    return seg_ids, rank_token_mask


def _compute_rank_grpo_advantage(
    data: DataProto,
    *,
    config,
    tokenizer,
    norm_adv_by_std_in_grpo: bool,
) -> DataProto:
    if tokenizer is None:
        raise ValueError("Rank-GRPO advantage computation requires the trainer tokenizer.")

    rank_cfg = _cfg_get(config, "rank_grpo", {}) or {}
    rec_num = int(_cfg_get(rank_cfg, "rec_num", 20))
    rank_separator = _cfg_get(rank_cfg, "rank_separator", "\n")
    year_tolerance = int(_cfg_get(rank_cfg, "year_tolerance", 2))
    exclude_seen = bool(_cfg_get(rank_cfg, "exclude_seen", True))
    normalize_by_std = bool(_cfg_get(rank_cfg, "normalize_by_std", norm_adv_by_std_in_grpo))

    from verl_gr.recipes.rankgrpo.rankgrpo_reward import rank_rewards_from_text

    responses = data.batch["responses"]
    response_mask = data.batch["response_mask"]
    response_texts = _decode_response_texts(responses, response_mask, tokenizer)
    reward_models = data.non_tensor_batch.get("reward_model")
    if reward_models is None:
        raise KeyError("Rank-GRPO requires `reward_model` in data.non_tensor_batch.")

    reward_rows = [
        rank_rewards_from_text(
            text,
            reward_model,
            rec_num=rec_num,
            year_tolerance=year_tolerance,
            exclude_seen=exclude_seen,
        )
        for text, reward_model in zip(response_texts, reward_models, strict=True)
    ]
    rank_rewards = torch.tensor(reward_rows, dtype=torch.float32, device=responses.device)

    uids = data.non_tensor_batch.get("uid")
    if uids is None:
        uids = list(range(rank_rewards.size(0)))
    uid_to_indices: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[uid].append(idx)

    rank_advantages = torch.zeros_like(rank_rewards)
    for indices in uid_to_indices.values():
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=responses.device)
        group_rewards = rank_rewards.index_select(0, idx_tensor)
        centered = group_rewards - group_rewards.mean(dim=0, keepdim=True)
        if normalize_by_std:
            std = group_rewards.std(dim=0, unbiased=False, keepdim=True)
            centered = centered / (std + 1e-4)
        rank_advantages.index_copy_(0, idx_tensor, centered)

    seg_ids, rank_token_mask = _segment_rank_tokens(
        responses,
        response_mask,
        tokenizer,
        rank_separator=rank_separator,
        rec_num=rec_num,
    )
    clamped_seg_ids = seg_ids.clamp(min=0, max=rec_num - 1)
    token_advantages = rank_advantages.gather(1, clamped_seg_ids)
    token_advantages = token_advantages * rank_token_mask.float()

    data.batch["advantages"] = token_advantages
    data.batch["returns"] = token_advantages
    data.batch["rank_token_mask"] = rank_token_mask
    data.batch["rank_seg_ids"] = seg_ids
    return data


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,  # noqa: ARG001
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    tokenizer=None,  # noqa: ARG001
) -> DataProto:
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        if _rankgrpo_enabled(config):
            if tokenizer is None:
                tokenizer = _RANKGRPO_TOKENIZER
            data = _compute_rank_grpo_advantage(
                data,
                config=config,
                tokenizer=tokenizer,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RLTrainer(RayPPOTrainerBase):
    """RayPPOTrainer override with different workload helpers."""

    def __init__(self, *args, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None and len(args) >= 2:
            tokenizer = args[1]
        super().__init__(*args, **kwargs)
        global _RANKGRPO_TOKENIZER
        _RANKGRPO_TOKENIZER = tokenizer
        if _rankgrpo_enabled(self.config.algorithm):
            import verl.trainer.ppo.ray_trainer as ray_trainer_mod

            ray_trainer_mod.compute_advantage = compute_advantage

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _expected_actor_lr(self) -> float | None:
        """Best-effort actor LR for logging when the worker omits it."""

        optim_config = _cfg_get(self.config.actor_rollout_ref.actor, "optim", None)
        if optim_config is None:
            return None

        base_lr = self._as_float(_cfg_get(optim_config, "lr", None), default=-1.0)
        if base_lr < 0:
            return None

        total_steps = self._as_int(
            _cfg_get(optim_config, "total_training_steps", self.total_training_steps),
            default=self.total_training_steps,
        )
        if total_steps <= 0:
            total_steps = self.total_training_steps

        warmup_steps = self._as_int(_cfg_get(optim_config, "lr_warmup_steps", -1), default=-1)
        if warmup_steps <= 0:
            warmup_ratio = self._as_float(_cfg_get(optim_config, "lr_warmup_steps_ratio", 0.0), default=0.0)
            warmup_steps = int(warmup_ratio * total_steps)

        step = max(self._as_int(getattr(self, "global_steps", 0), default=0), 0)
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * float(step) / float(max(1, warmup_steps))

        scheduler_type = _cfg_get(optim_config, "lr_scheduler_type", _cfg_get(optim_config, "warmup_style", "constant"))
        if scheduler_type != "cosine":
            return base_lr

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
        min_lr_ratio = self._as_float(_cfg_get(optim_config, "min_lr_ratio", 0.0), default=0.0)
        num_cycles = self._as_float(_cfg_get(optim_config, "num_cycles", 0.5), default=0.5)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_scale)

    def _add_actor_lr_metrics(self, metrics: dict[str, Any]) -> None:
        optim_config = _cfg_get(self.config.actor_rollout_ref.actor, "optim", None)
        if optim_config is not None and "actor/base_lr" not in metrics:
            base_lr = self._as_float(_cfg_get(optim_config, "lr", None), default=-1.0)
            if base_lr >= 0:
                metrics["actor/base_lr"] = base_lr

        if "actor/lr" in metrics:
            return
        if "lr" in metrics:
            metrics["actor/lr"] = metrics["lr"]
            return

        expected_lr = self._expected_actor_lr()
        if expected_lr is not None:
            metrics["actor/lr"] = expected_lr

    def _update_actor(self, batch: DataProto) -> DataProto:
        actor_output = super()._update_actor(batch)
        self._add_actor_lr_metrics(actor_output.meta_info["metrics"])
        return actor_output

    @staticmethod
    def _ensure_reward_routing_keys(proto: DataProto) -> None:
        """Ensure both source aliases exist for reward-loop compatibility."""
        non_tensor = proto.non_tensor_batch
        if "data_source" not in non_tensor and "source" in non_tensor:
            non_tensor["data_source"] = non_tensor["source"]
        if "source" not in non_tensor and "data_source" in non_tensor:
            non_tensor["source"] = non_tensor["data_source"]

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Prepare generation batch without conflicting prompt tensors.

        In verl>=0.7.1 async rollout mode, generation output may include input_ids.
        If original training batch still carries prompt-side input_ids/attention_mask/
        position_ids, DataProto.union() asserts on key collisions. For OneRec dataset,
        we remove those prompt tensors before generation and keep reward-routing keys.
        """
        reward_keys = set(
            {
                "source",
                "data_source",
                "reward_model",
                "uid",
                "raw_prompt",
                "multi_modal_data",
                "tools_kwargs",
                "interaction_kwargs",
            }
        ) & batch.non_tensor_batch.keys()
        batch_keys_to_pop = [
            key for key in ("input_ids", "attention_mask", "position_ids") if key in batch.batch.keys()
        ]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        self._ensure_reward_routing_keys(gen_batch)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        if rollout_cfg.get("name") == "two_stage":
            rollout_custom = rollout_cfg.get("custom") or {}
            reasoning_max_tokens = rollout_custom.get(
                "stage1_max_tokens",
                get_rollout_custom_nested_value(
                    rollout_cfg,
                    (DECODE_CONFIG_KEY, "reasoning", "max_tokens"),
                    self.config.data.get("max_response_length", rollout_cfg.response_length),
                ),
            )
            beam_width = rollout_custom.get(
                BEAM_WIDTH_KEY,
                rollout_custom.get("stage2_beam_size", 32),
            )
            item_max_tokens = rollout_custom.get(
                "stage2_num_tokens",
                get_rollout_custom_nested_value(
                    rollout_cfg,
                    (BEAM_SEARCH_PARAMS_KEY, "max_tokens"),
                    3,
                ),
            )
            gen_batch.meta_info.update(
                {
                    "enable_two_stage_rollout": True,
                    "max_tokens": self.config.data.get("max_response_length", rollout_cfg.response_length),
                }
            )
            gen_batch.meta_info.update(
                build_two_stage_sampling_params(
                    reasoning_max_tokens=int(reasoning_max_tokens),
                    item_max_tokens=int(item_max_tokens),
                    beam_width=int(beam_width),
                )
            )
        return gen_batch

    def _validate(self):
        return openonerec_validate(self)

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, ground_truths=None):
        return openonerec_dump_generations(
            self,
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
            ground_truths=ground_truths,
        )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        return openonerec_maybe_log_val_generations(self, inputs=inputs, outputs=outputs, scores=scores)

