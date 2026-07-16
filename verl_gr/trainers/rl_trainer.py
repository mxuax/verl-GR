"""RL trainer extensions for verl-GR with bridged ray-trainer API."""

import numpy as np
import json
import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import torch
import time
from omegaconf import open_dict

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as RayPPOTrainerBase, rename_dict
from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager
from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import masked_mean
from verl.workers.utils.padding import left_right_2_no_padding

from verl_gr.recipes.task_factory import load_object
from verl_gr.recipes.openonerec.onerec_profile_metrics import compute_openonerec_data_metrics
from verl_gr.recipes.openonerec.onerec_trainer import (
    openonerec_evaluate_and_prune_checkpoint,
    openonerec_dump_generations,
    openonerec_maybe_log_val_generations,
    openonerec_validate,
)
from verl_gr.recipes.rankgrpo.rankgrpo_algorithm import (
    compute_rank_grpo_advantage,
    compute_rank_grpo_training_reward_metrics,
    rankgrpo_enabled,
)
from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import (
    alignment_report_enabled,
    calculate_rankgrpo_logprob_gate_metrics,
    maybe_export_rankgrpo_logprobs,
    merge_sidecar_probe_timings,
    record_rankgrpo_alignment_metrics,
    write_rankgrpo_alignment_report,
)
from verl_gr.recipes.rankgrpo.rankgrpo_trainer import RankGRPOTrainerAdapter
from verl_gr.trainers.task_adapter import TrainerTaskAdapter
from verl_gr.workers.rollout.beam_config import (
    BEAM_RETURN_MODE_KEY,
    BEAM_SEARCH_PARAMS_KEY,
    BEAM_WIDTH_KEY,
    DECODE_CONFIG_KEY,
    build_two_stage_sampling_params,
    get_rollout_custom_nested_value,
)

AdvantageEstimator = getattr(core_algos, "AdvantageEstimator")
_RANKGRPO_TOKENIZER = None


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


class _OpenOneRecTrainerAdapter(TrainerTaskAdapter):
    def prepare_gen_batch(self, trainer, batch: DataProto) -> DataProto:
        return trainer._prepare_recommendation_gen_batch(batch)

    def validate(self, trainer):
        return openonerec_validate(trainer)

    def dump_generations(self, trainer, inputs, outputs, scores, reward_extra_infos_dict, dump_path, ground_truths=None):
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
        if rankgrpo_enabled(config):
            if tokenizer is None:
                tokenizer = _RANKGRPO_TOKENIZER
            data = compute_rank_grpo_advantage(
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
                epsilon=1e-4,
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


def _openonerec_enabled(config) -> bool:
    task_name = str(_cfg_get(_cfg_get(config, "task", None), "name", "")).lower()
    return task_name == "openonerec"


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    from verl.trainer.ppo.metric_utils import compute_data_metrics as _base_compute_data_metrics

    metrics = _base_compute_data_metrics(batch=batch, use_critic=use_critic)
    metrics.update(compute_rank_grpo_training_reward_metrics(batch))
    probe_t0 = time.perf_counter()
    metrics.update(calculate_rankgrpo_logprob_gate_metrics(batch))
    if alignment_report_enabled():
        metrics["timing_rankgrpo/probe_logprob_gate"] = time.perf_counter() - probe_t0
    step = batch.meta_info.get("global_steps") if isinstance(getattr(batch, "meta_info", None), dict) else None
    if step is not None:
        try:
            maybe_export_rankgrpo_logprobs(batch, step=int(step))
        except (TypeError, ValueError):
            pass
    return metrics


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_constraint_info_file(rollout_config) -> str:
    """Extract info_file path from rollout custom config (OmegaConf-safe)."""
    custom = getattr(rollout_config, "custom", {}) or {}
    if hasattr(custom, "items"):
        custom = dict(custom.items())
    if isinstance(custom, dict):
        beam_params = custom.get("beam_search_params", {}) or {}
        if hasattr(beam_params, "items"):
            beam_params = dict(beam_params.items())
        if isinstance(beam_params, dict):
            constraint = beam_params.get("constraint", {}) or {}
            if hasattr(constraint, "items"):
                constraint = dict(constraint.items())
            if isinstance(constraint, dict):
                info = constraint.get("info_file", "")
                return str(info) if info else ""
    return ""


def _prune_unkept_checkpoint_dirs(ckpt_root: str, keep_paths: set[str]) -> list[str]:
    """Delete `global_step_*` dirs under `ckpt_root` that are not in `keep_paths`."""

    removed: list[str] = []
    if not os.path.isdir(ckpt_root):
        return removed

    normalized_keep = {os.path.abspath(path) for path in keep_paths}
    for name in os.listdir(ckpt_root):
        if not name.startswith("global_step_"):
            continue
        path = os.path.join(ckpt_root, name)
        abs_path = os.path.abspath(path)
        if abs_path in normalized_keep:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(path)
    return removed


class RLTrainer(RayPPOTrainerBase):
    """RayPPOTrainer override with different workload helpers."""

    def __init__(self, *args, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None and len(args) >= 2:
            tokenizer = args[1]
        super().__init__(*args, **kwargs)
        self._apply_actor_scheduler_total_steps_override()
        global _RANKGRPO_TOKENIZER
        _RANKGRPO_TOKENIZER = tokenizer
        if rankgrpo_enabled(self.config.algorithm):
            import verl.trainer.ppo.ray_trainer as ray_trainer_mod

            ray_trainer_mod.compute_advantage = compute_advantage
            ray_trainer_mod.compute_data_metrics = compute_data_metrics
        elif _openonerec_enabled(self.config):
            import verl.trainer.ppo.ray_trainer as ray_trainer_mod

            ray_trainer_mod.compute_data_metrics = compute_openonerec_data_metrics

    def _apply_actor_scheduler_total_steps_override(self) -> None:
        """Allow short runs to keep the original LR schedule length.

        Upstream verl uses ``trainer.total_training_steps`` for both early stop
        and ``actor.optim.total_training_steps``. Alignment probes often need to
        stop at step 165 while preserving the full original/H69 scheduler horizon.
        """

        actor_cfg = _cfg_get(_cfg_get(self.config, "actor_rollout_ref", None), "actor", None)
        optim_cfg = _cfg_get(actor_cfg, "optim", None)
        if optim_cfg is None:
            return
        override_steps = _cfg_get(
            optim_cfg,
            "scheduler_total_training_steps",
            _cfg_get(optim_cfg, "lr_scheduler_total_training_steps", None),
        )
        override_steps = self._as_int(override_steps, default=-1)
        if override_steps <= 0:
            return
        with open_dict(optim_cfg):
            optim_cfg.total_training_steps = override_steps
            for key in ("scheduler_total_training_steps", "lr_scheduler_total_training_steps"):
                if key in optim_cfg:
                    del optim_cfg[key]
        print(
            "[RLTrainer] actor scheduler total_training_steps override: "
            f"{override_steps} (trainer stop remains {self.total_training_steps})",
            flush=True,
        )

    def init_workers(self):
        super().init_workers()
        # MiniOneRec uses a rule-based reward function without an RM.
        # self.use_rm must be True so that _compute_reward_colocate runs
        # and the task adapter can postprocess rewards to set rm_scores.
        # For other tasks (RankGRPO, etc.) the default use_rm=False avoids
        # the unnecessary per-step Ray remote call overhead.
        if self._get_task_adapter_is_minionerec():
            self.use_rm = True

    def _rankgrpo_gates_enabled(self) -> bool:
        task_name = str(_cfg_get(_cfg_get(self.config, "task", None), "name", "")).lower()
        return task_name == "rankgrpo"

    def _maybe_rankgrpo_convergence_gate(self, step: int, metrics: Any) -> None:
        if not self._rankgrpo_gates_enabled() or not isinstance(metrics, dict):
            return
        try:
            from verl_gr.recipes.rankgrpo.alignment.convergence_gate import (
                maybe_abort_on_kl_growth_failure,
                maybe_abort_on_length_blowout,
            )

            maybe_abort_on_kl_growth_failure(int(step), metrics)
            maybe_abort_on_length_blowout(int(step), metrics)
        except SystemExit:
            raise
        except Exception:
            pass

    def _write_rankgrpo_convergence_gate_report(self) -> None:
        if not self._rankgrpo_gates_enabled():
            return
        try:
            from verl_gr.recipes.rankgrpo.alignment.convergence_gate import (
                write_convergence_gate_report,
            )

            trainer_cfg = self.config.trainer
            experiment_name = str(_cfg_get(trainer_cfg, "experiment_name", ""))
            default_local_dir = _cfg_get(trainer_cfg, "default_local_dir", None)
            output_dir = Path(default_local_dir).parent if default_local_dir else Path(
                os.environ.get("OUTPUT_DIR", ".")
            )
            write_convergence_gate_report(
                output_dir=output_dir,
                experiment_name=experiment_name,
            )
        except Exception:
            pass

    def _get_task_adapter_is_minionerec(self) -> bool:
        rollout = str(self.config.actor_rollout_ref.rollout.get("name", ""))
        if rollout == "constrained_beam":
            return True
        task_cfg = self.config.get("task", {})
        if task_cfg and str(task_cfg.get("name", "")).lower() == "minionerec":
            return True
        return False

    def fit(self):
        logging_steps = self._as_int(_cfg_get(self.config.trainer, "logging_steps", 1), default=1)
        rankgrpo_report = rankgrpo_enabled(self.config.algorithm) and alignment_report_enabled()

        from verl.utils.tracking import Tracking

        original_log = Tracking.log

        def _wrapped_log(tracking_self, data, step, backend=None):
            step_i = int(step)
            should_log = logging_steps <= 1 or step_i == 0 or step_i % logging_steps == 0
            if rankgrpo_report and isinstance(data, dict):
                accum_t0 = time.perf_counter()
                record_rankgrpo_alignment_metrics(step_i, data)
                merge_sidecar_probe_timings(
                    step_i,
                    {"timing_rankgrpo/probe_align_accum": time.perf_counter() - accum_t0},
                )
            if not should_log:
                return None
            self._maybe_rankgrpo_convergence_gate(step_i, data)
            tb_t0 = time.perf_counter()
            result = original_log(tracking_self, data=data, step=step, backend=backend)
            if rankgrpo_report:
                merge_sidecar_probe_timings(
                    step_i,
                    {"timing_rankgrpo/probe_tb_log": time.perf_counter() - tb_t0},
                )
            return result

        Tracking.log = _wrapped_log
        try:
            super().fit()
        finally:
            Tracking.log = original_log
            self._write_rankgrpo_convergence_gate_report()
            if rankgrpo_report:
                report_root = os.environ.get("VERL_GR_ALIGN_REPORT_DIR")
                if not report_root:
                    output_dir = _cfg_get(self.config.trainer, "default_local_dir", None)
                    if output_dir:
                        report_root = str(Path(output_dir).parent)
                    else:
                        report_root = os.environ.get("OUTPUT_DIR")
                result = write_rankgrpo_alignment_report(
                    output_dir=report_root,
                    experiment_name=str(_cfg_get(self.config.trainer, "experiment_name", "")),
                )
                if result is not None and os.environ.get("VERL_GR_ALIGN_GATE_EXIT", "1").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    _, gate = result
                    if not gate.passed:
                        raise SystemExit(2)

    def _get_task_adapter(self) -> TrainerTaskAdapter:
        if hasattr(self, "_task_adapter"):
            return self._task_adapter

        task_name = str(_cfg_get(_cfg_get(self.config, "task", None), "name", "")).lower()
        rollout_name = str(self.config.actor_rollout_ref.rollout.get("name", ""))
        if task_name == "rankgrpo":
            self._task_adapter = RankGRPOTrainerAdapter()
        elif task_name == "minionerec" or rollout_name == "constrained_beam":
            adapter_cls = load_object("verl_gr.recipes.minionerec.minionerec_trainer.MiniOneRecTrainerAdapter")
            self._task_adapter = adapter_cls()
        elif task_name == "openonerec":
            self._task_adapter = _OpenOneRecTrainerAdapter()
        else:
            self._task_adapter = TrainerTaskAdapter()
        return self._task_adapter

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

    def _compute_old_log_prob(self, batch: DataProto):
        """Override to skip old_log_prob forward pass for REINFORCE loss.

        The minionerec_reinforce loss does not use old_log_probs in its gradient
        (only for the ppo_kl metric), and use_kl_in_reward is false for minionerec.
        This saves one full actor inference per step (~20% training time).

        Only activates when the ACTUAL composed config has
        ``policy_loss.loss_mode == "minionerec_reinforce"`` — never bypass based
        on rollout name alone, because vanilla PPO requires correct old_log_probs
        for the importance ratio exp(logp - old_logp).
        """
        actor_cfg = self.config.actor_rollout_ref.actor
        loss_mode = ""
        try:
            if hasattr(actor_cfg, "policy_loss"):
                pl = actor_cfg.policy_loss
                if hasattr(pl, "loss_mode"):
                    loss_mode = pl.loss_mode
                elif hasattr(pl, "get"):
                    loss_mode = pl.get("loss_mode", "")
            elif hasattr(actor_cfg, "get"):
                pl = actor_cfg.get("policy_loss", {})
                if hasattr(pl, "get"):
                    loss_mode = pl.get("loss_mode", "")
                elif hasattr(pl, "loss_mode"):
                    loss_mode = pl.loss_mode
        except Exception:
            loss_mode = ""

        if loss_mode != "minionerec_reinforce":
            if not getattr(self, "_old_log_prob_losswarned", False):
                print(f"[RLTrainer._compute_old_log_prob] loss_mode={loss_mode!r} "
                      f"-> using parent forward pass", flush=True)
                self._old_log_prob_losswarned = True
            return super()._compute_old_log_prob(batch)

        if not getattr(self, "_old_log_prob_bypass_logged", False):
            print("[RLTrainer._compute_old_log_prob] loss_mode='minionerec_reinforce' — "
                  "old_log_prob bypass active (zero-filled, saving one forward pass per step).",
                  flush=True)
            self._old_log_prob_bypass_logged = True

        # For rollout-bypass mode, keep the proximal anchor tied to the policy
        # that actually generated the samples. This matches upstream verl's
        # bypass_mode (`old_log_probs = rollout_log_probs`) and avoids an
        # actor-side recompute with a numerically different forward path.
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        response_mask = batch.batch["response_mask"]
        if "rollout_log_probs" in batch.batch:
            log_probs = batch.batch["rollout_log_probs"].to(dtype=torch.float32)
            if log_probs.shape != response_mask.shape:
                raise RuntimeError(
                    "rollout_log_probs shape mismatch: "
                    f"{tuple(log_probs.shape)} vs response_mask {tuple(response_mask.shape)}"
                )
            if not getattr(self, "_old_log_prob_rollout_logged", False):
                print(
                    "[RLTrainer._compute_old_log_prob] using rollout_log_probs as old_log_probs "
                    "for minionerec_reinforce.",
                    flush=True,
                )
                self._old_log_prob_rollout_logged = True
        else:
            log_probs = torch.zeros_like(response_mask, dtype=torch.float32)
            if not getattr(self, "_old_log_prob_zero_fallback_logged", False):
                print(
                    "[RLTrainer._compute_old_log_prob] rollout_log_probs missing; "
                    "falling back to zero-filled old_log_probs.",
                    flush=True,
                )
                self._old_log_prob_zero_fallback_logged = True
        entropy = torch.zeros_like(response_mask, dtype=torch.float32)

        old_log_prob_td = tu.get_tensordict({"old_log_probs": log_probs, "entropys": entropy})
        old_log_prob = DataProto.from_tensordict(old_log_prob_td)
        return old_log_prob, 0.0

    def _update_actor(self, batch: DataProto) -> DataProto:
        with _nvtx_range("actor.forward_backward"):
            if self._uses_minionerec_reinforce_loss():
                actor_output = self._update_minionerec_actor(batch)
            else:
                actor_output = super()._update_actor(batch)
        self._add_actor_lr_metrics(actor_output.meta_info["metrics"])
        if not batch.meta_info.get("validate", False):
            self._try_sync_ref_model()
        return actor_output

    def _uses_minionerec_reinforce_loss(self) -> bool:
        actor_cfg = self.config.actor_rollout_ref.actor
        policy_loss = actor_cfg.get("policy_loss", {}) if hasattr(actor_cfg, "get") else {}
        loss_mode = policy_loss.get("loss_mode", "") if hasattr(policy_loss, "get") else getattr(policy_loss, "loss_mode", "")
        return str(loss_mode) == "minionerec_reinforce"

    def _maybe_dump_minionerec_real_batch(self, tag: str, data: Any) -> None:
        """Opt-in dump of the real rollout/update batch for cross-framework replay."""

        dump_dir = os.getenv("MINIONEREC_REALBATCH_DUMP_DIR")
        if not dump_dir:
            return
        max_dumps = int(os.getenv("MINIONEREC_REALBATCH_MAX_DUMPS", "1"))
        count_attr = "_minionerec_realbatch_dump_count"
        count = int(getattr(self, count_attr, 0))
        if count >= max_dumps:
            return

        os.makedirs(dump_dir, exist_ok=True)
        step = int(getattr(self, "global_steps", -1))
        target_steps_raw = os.getenv("MINIONEREC_REALBATCH_DUMP_STEPS", "").strip()
        if target_steps_raw:
            target_steps = {
                int(item.strip())
                for item in target_steps_raw.split(",")
                if item.strip()
            }
            if step not in target_steps:
                return
        payload: dict[str, Any] = {
            "tag": tag,
            "global_step": step,
            "tensor": {},
            "non_tensor": {},
            "meta_info": {},
        }

        tensor_keys = {
            "prompts",
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "response_mask",
            "loss_mask",
            "rollout_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "token_level_scores",
            "token_level_rewards",
            "advantages",
            "returns",
            "rm_scores",
        }

        def add_tensor(key: str, value: Any) -> None:
            if isinstance(value, torch.Tensor):
                payload["tensor"][key] = value.detach().cpu().clone()

        if isinstance(data, DataProto):
            for key in tensor_keys:
                if key in data.batch:
                    add_tensor(key, data.batch[key])
            for key, value in data.non_tensor_batch.items():
                if key in {
                    "uid",
                    "index",
                    "raw_prompt",
                    "raw_prompt_text",
                    "reward_model",
                    "source",
                    "data_source",
                    "minionerec_rule_reward",
                    "minionerec_ranking_reward",
                    "minionerec_shape_penalty",
                    "minionerec_total_reward",
                    "minionerec_invalid_sid",
                    "minionerec_empty_completion",
                }:
                    payload["non_tensor"][key] = np.asarray(value, dtype=object).tolist()
            payload["meta_info"] = dict(data.meta_info)
        else:
            for key in tensor_keys:
                try:
                    add_tensor(key, data.get(key))
                except (AttributeError, KeyError, RuntimeError):
                    pass
            for key in (
                "global_batch_size",
                "mini_batch_size",
                "epochs",
                "seed",
                "calculate_entropy",
                "compute_loss",
                "batch_num_tokens",
                "dp_size",
            ):
                try:
                    value = tu.get(data, key, default=None)
                except (AttributeError, KeyError, RuntimeError):
                    value = None
                if value is not None:
                    payload["meta_info"][key] = value

        base = os.path.join(dump_dir, f"step{step:06d}_{tag}")
        torch.save(payload, f"{base}.pt")
        summary = {
            "tag": tag,
            "global_step": step,
            "tensor": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sum": float(value.float().sum().item()) if value.is_floating_point() else int(value.long().sum().item()),
                }
                for key, value in payload["tensor"].items()
            },
            "non_tensor_keys": sorted(payload["non_tensor"].keys()),
            "meta_info": payload["meta_info"],
        }
        with open(f"{base}.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, default=str)
        if tag == "post_padding":
            setattr(self, count_attr, count + 1)

    def _update_minionerec_actor(self, batch: DataProto) -> DataProto:
        """Update actor once over the fully expanded MiniOneRec beam batch."""

        self._maybe_dump_minionerec_real_batch("pre_padding", batch)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        expanded_batch_size = int(batch_td.shape[0])
        if not getattr(self, "_minionerec_full_batch_update_logged", False):
            cfg_mini = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            cfg_rollout_n = self.config.actor_rollout_ref.rollout.n
            print(
                "[MiniOneRec actor update] using expanded batch size for "
                f"mini/global batch: {expanded_batch_size} "
                f"(configured ppo_mini_batch_size={cfg_mini}, rollout.n={cfg_rollout_n})",
                flush=True,
            )
            self._minionerec_full_batch_update_logged = True

        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            global_batch_size=expanded_batch_size,
            mini_batch_size=expanded_batch_size,
            epochs=self.config.actor_rollout_ref.actor.ppo_epochs,
            seed=self.config.actor_rollout_ref.actor.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            compute_loss=True,
        )
        self._maybe_dump_minionerec_real_batch("post_padding", batch_td)
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        if "actor/mfu" in actor_output:
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        with _nvtx_range("ref.forward"):
            return super()._compute_ref_log_prob(batch)

    def _try_sync_ref_model(self):
        if not self.use_reference_policy or self.ref_in_actor:
            return
        ref_cfg = _cfg_get(self.config.actor_rollout_ref, "ref")
        freq = _cfg_get(ref_cfg, "sync_freq")
        if freq is None:
            rollout = str(self.config.actor_rollout_ref.rollout.get("name", ""))
            freq = 512 if rollout == "constrained_beam" else 0
        freq = int(freq)
        if freq <= 0 or self.global_steps % freq != 0:
            return
        # TRL-style EMA mixup: ref = (1-alpha) * ref + alpha * actor.
        # alpha = 1 → hard copy.
        alpha = float(_cfg_get(ref_cfg, "ref_model_mixup_alpha", 0.6))
        self.ref_policy_wg.sync_ref_weights(mixup_alpha=alpha)

    def _compute_eval_actor_metrics(self, batch: DataProto) -> dict[str, Any]:
        """Compute actor loss metrics in eval mode without stepping the optimizer."""

        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            compute_loss=True,
            global_batch_size=batch_td.shape[0],
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        return dict(tu.get(output, "metrics") or {})

    @staticmethod
    def _mean_metric(values: list[tuple[float, int]]) -> float | None:
        total_weight = sum(weight for _, weight in values)
        if total_weight <= 0:
            return None
        return sum(value * weight for value, weight in values) / total_weight

    def _checkpoint_topk_state_path(self) -> str:
        return os.path.join(self.config.trainer.default_local_dir, "topk_checkpoints.json")

    def _load_topk_checkpoint_state(self) -> list[dict[str, Any]]:
        if hasattr(self, "_topk_checkpoints"):
            return self._topk_checkpoints
        state_path = self._checkpoint_topk_state_path()
        try:
            with open(state_path) as f:
                state = json.load(f)
        except FileNotFoundError:
            state = []
        self._topk_checkpoints = state if isinstance(state, list) else []
        return self._topk_checkpoints

    def _save_topk_checkpoint_state(self, state: list[dict[str, Any]]) -> None:
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        with open(self._checkpoint_topk_state_path(), "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        if state:
            latest_kept_step = max(int(entry["step"]) for entry in state)
            latest_path = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
            with open(latest_path, "w") as f:
                f.write(str(latest_kept_step))
        self._topk_checkpoints = state

    def _select_topk_metric(self, metrics: dict[str, Any]) -> tuple[str | None, float | None]:
        metric_name = _cfg_get(
            self.config.trainer,
            "best_ckpt_metric",
            _cfg_get(self.config.trainer, "topk_ckpt_metric", None),
        )
        if metric_name:
            value = metrics.get(metric_name)
            return metric_name, self._as_float(value, default=float("nan"))

        for candidate in (
            "val-core/rankgrpo/reward/mean@1",
            "val-core/rankgrpo/score/mean@1",
            "val-core/rankgrpo/rank_reward_sum/mean@1",
        ):
            if candidate in metrics:
                return candidate, self._as_float(metrics[candidate], default=float("nan"))

        for key in sorted(metrics):
            if key.startswith("val-core/") and key.endswith("/mean@1"):
                return key, self._as_float(metrics[key], default=float("nan"))
        return None, None

    def _update_topk_checkpoints(self, metrics: dict[str, Any]) -> None:
        prune_enabled = _cfg_get(self.config.trainer, "best_ckpt_prune_enable", True)
        if isinstance(prune_enabled, str):
            prune_enabled = prune_enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
        if not prune_enabled:
            return

        top_k = self._as_int(
            _cfg_get(
                self.config.trainer,
                "best_ckpts_to_keep",
                _cfg_get(self.config.trainer, "topk_ckpt_keep", 0),
            ),
            default=0,
        )
        if top_k <= 0 or self.global_steps <= 0:
            return

        ckpt_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.isdir(ckpt_dir):
            return

        metric_name, metric_value = self._select_topk_metric(metrics)
        if metric_name is None or metric_value is None or not math.isfinite(metric_value):
            print("[topk] No finite validation metric found; skipping checkpoint ranking.")
            return

        mode = str(
            _cfg_get(
                self.config.trainer,
                "best_ckpt_mode",
                _cfg_get(self.config.trainer, "topk_ckpt_mode", "max"),
            )
        ).lower()
        reverse = mode != "min"
        state = [entry for entry in self._load_topk_checkpoint_state() if int(entry.get("step", -1)) != self.global_steps]
        state.append(
            {
                "step": int(self.global_steps),
                "metric": metric_name,
                "value": float(metric_value),
                "path": ckpt_dir,
            }
        )
        state.sort(key=lambda entry: float(entry["value"]), reverse=reverse)
        keep = state[:top_k]
        drop = state[top_k:]

        keep_paths = {entry["path"] for entry in keep}
        for path in _prune_unkept_checkpoint_dirs(self.config.trainer.default_local_dir, keep_paths):
            print(f"[topk] Removed checkpoint outside top-{top_k}: {path}")

        self._save_topk_checkpoint_state(keep)
        print(f"[topk] Kept top-{top_k} checkpoints by {metric_name}: {keep}")

    @staticmethod
    def _ensure_reward_routing_keys(proto: DataProto) -> None:
        """Ensure both source aliases exist for reward-loop compatibility."""
        non_tensor = proto.non_tensor_batch
        if "data_source" not in non_tensor and "source" in non_tensor:
            non_tensor["data_source"] = non_tensor["source"]
        if "source" not in non_tensor and "data_source" in non_tensor:
            non_tensor["source"] = non_tensor["data_source"]

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        return self._get_task_adapter().prepare_gen_batch(self, batch)

    def _prepare_recommendation_gen_batch(self, batch: DataProto) -> DataProto:
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
            beam_search_params = rollout_custom.get(BEAM_SEARCH_PARAMS_KEY) or {}
            if beam_search_params.get("constraint") is not None:
                gen_batch.meta_info["constraint"] = beam_search_params.get("constraint")
            gen_batch.meta_info.update(
                build_two_stage_sampling_params(
                    reasoning_max_tokens=int(reasoning_max_tokens),
                    item_max_tokens=int(item_max_tokens),
                    beam_width=int(beam_width),
                )
            )
        elif rollout_cfg.get("name") == "constrained_beam":
            rollout_custom = rollout_cfg.get("custom") or {}
            beam_search_params = rollout_custom.get(BEAM_SEARCH_PARAMS_KEY) or {}
            beam_width = int(rollout_custom.get(BEAM_WIDTH_KEY, rollout_custom.get("beam_size", 20)))
            item_max_tokens = int(beam_search_params.get("max_tokens", self.config.data.get("max_response_length", 64)))
            gen_batch.meta_info.update(
                {
                    "enable_constrained_beam_rollout": True,
                    "max_tokens": item_max_tokens,
                    BEAM_WIDTH_KEY: beam_width,
                    BEAM_RETURN_MODE_KEY: "best_only",
                    BEAM_SEARCH_PARAMS_KEY: dict(beam_search_params),
                }
            )
            if beam_search_params.get("constraint") is not None:
                gen_batch.meta_info["constraint"] = beam_search_params.get("constraint")
        return gen_batch

    def _validate(self):
        metrics = self._get_task_adapter().validate(self)
        self._last_validation_metrics = metrics
        self._update_topk_checkpoints(metrics)
        return metrics

    def _compute_reward_colocate(self, batch: DataProto):
        with _nvtx_range("reward.compute"):
            reward_batch = super()._compute_reward_colocate(batch)
        if batch.meta_info.get("validate", False):
            return reward_batch
        reward_batch, reward_extra_info = self._get_task_adapter().postprocess_rewards(self, batch, reward_batch)
        if reward_extra_info:
            for key, values in reward_extra_info.items():
                reward_batch.non_tensor_batch[key] = np.array(values, dtype=object)
            reward_extra_keys = list(reward_batch.meta_info.get("reward_extra_keys", []))
            for key in reward_extra_info:
                if key not in reward_extra_keys:
                    reward_extra_keys.append(key)
            reward_batch.meta_info["reward_extra_keys"] = reward_extra_keys

            # Compute per-step scalar reward metrics for wandb logging.
            # The parent framework only logs generic critic/score/mean and
            # critic/rewards/mean; these add MiniOneRec-specific breakdowns.
            reward_metrics = {}
            for key, values in reward_extra_info.items():
                try:
                    arr = np.asarray(values, dtype=np.float64)
                    reward_metrics[f"minionerec/{key}/mean"] = float(arr.mean())
                except (ValueError, TypeError):
                    pass
            existing = reward_batch.meta_info.get("metrics", {})
            existing.update(reward_metrics)
            reward_batch.meta_info["metrics"] = existing
        return reward_batch

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, ground_truths=None):
        return self._get_task_adapter().dump_generations(
            self,
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
            ground_truths=ground_truths,
        )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        return self._get_task_adapter().maybe_log_val_generations(self, inputs=inputs, outputs=outputs, scores=scores)

    def _save_checkpoint(self):
        super()._save_checkpoint()
        task_name = str(_cfg_get(_cfg_get(self.config, "task", None), "name", "")).lower()
        if task_name != "openonerec":
            return
        local_global_step_folder = f"{self.config.trainer.default_local_dir}/global_step_{self.global_steps}"
        self._get_task_adapter().evaluate_and_prune_checkpoint(
            self,
            local_global_step_folder,
            metrics=getattr(self, "_last_validation_metrics", None),
        )

