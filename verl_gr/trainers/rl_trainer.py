"""RL trainer extensions for verl-GR with bridged ray-trainer API."""

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo import reward as reward_mod
from verl.trainer.ppo import ray_trainer as ray_trainer_mod
from verl.trainer.ppo import metric_utils as metric_utils_mod
from verl.trainer.ppo.metric_utils import compute_data_metrics as base_compute_data_metrics
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


def _to_numeric_1d_array(values):
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        return None
    if np.issubdtype(arr.dtype, np.number):
        return arr
    if arr.dtype == np.dtype("O"):
        try:
            return np.asarray(arr, dtype=np.float64)
        except (TypeError, ValueError):
            return None
    return None


def _collect_reward_extra_metrics(batch: DataProto) -> dict[str, float]:
    non_tensor = batch.non_tensor_batch
    if not non_tensor:
        return {}

    batch_size = len(batch)
    data_sources = non_tensor.get("source")
    if data_sources is None:
        data_sources = non_tensor.get("data_source")
    if data_sources is not None:
        data_sources = np.asarray(data_sources)
        if data_sources.ndim != 1 or len(data_sources) != batch_size:
            data_sources = None

    skip_keys = {
        "source",
        "data_source",
        "uid",
        "reward_model",
        "extra_info",
        "request_id",
        "__num_turns__",
    }
    metrics: dict[str, float] = {}
    for key, values in non_tensor.items():
        if key in skip_keys:
            continue
        numeric_values = _to_numeric_1d_array(values)
        if numeric_values is None or len(numeric_values) != batch_size:
            continue
        metrics[f"reward/all/{key}/mean"] = float(np.mean(numeric_values))
        metrics[f"reward/all/{key}/max"] = float(np.max(numeric_values))
        metrics[f"reward/all/{key}/min"] = float(np.min(numeric_values))

        if data_sources is None:
            continue
        for source in np.unique(data_sources):
            source_mask = data_sources == source
            if not np.any(source_mask):
                continue
            source_values = numeric_values[source_mask]
            source_name = str(source)
            metrics[f"reward/{source_name}/{key}/mean"] = float(np.mean(source_values))
            metrics[f"reward/{source_name}/{key}/max"] = float(np.max(source_values))
            metrics[f"reward/{source_name}/{key}/min"] = float(np.min(source_values))
            metrics[f"reward/{source_name}/{key}/count"] = int(np.sum(source_mask))
    return metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, float]:
    metrics = base_compute_data_metrics(batch=batch, use_critic=use_critic)
    metrics.update(_collect_reward_extra_metrics(batch))
    return metrics


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


ray_trainer_mod.apply_kl_penalty = apply_kl_penalty
ray_trainer_mod.compute_advantage = compute_advantage
metric_utils_mod.compute_data_metrics = compute_data_metrics
ray_trainer_mod.compute_data_metrics = compute_data_metrics


class RLTrainer(RayPPOTrainerBase):
    """RayPPOTrainer override with different workload helpers."""

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
        reward_keys = set({"source", "data_source", "reward_model", "uid"}) & batch.non_tensor_batch.keys()
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

    def compute_validation_reward(self, batch: DataProto) -> dict:
        """Return validation reward in a unified dict shape.

        Normalized output format:
            {
                "reward_tensor": torch.Tensor,
                "reward_extra_info": dict[str, list | np.ndarray]
            }
        """
        if hasattr(self, "val_reward_fn") and self.val_reward_fn is not None:
            result = self.val_reward_fn(batch, return_dict=True)
            return {
                "reward_tensor": result["reward_tensor"],
                "reward_extra_info": result.get("reward_extra_info", {}),
            }

        # Compatibility for verl_080_dev-style trainer stacks.
        reward_tensor, reward_extra_info = reward_mod.extract_reward(batch)
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info or {},
        }

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

