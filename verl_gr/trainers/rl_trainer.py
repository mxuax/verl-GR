"""RL trainer extensions for verl-GR with bridged ray-trainer API."""

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

AdvantageEstimator = getattr(core_algos, "AdvantageEstimator")


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
        reward_keys = set({"source", "data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
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

