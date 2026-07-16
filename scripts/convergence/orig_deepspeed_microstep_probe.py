#!/usr/bin/env python3
"""Probe original MiniOneRec DeepSpeed microstep/update boundaries.

Run this file from the MiniOneRec environment, preferably through
``accelerate launch --config_file config/zero2_opt.yaml``.  It monkeypatches the
original ReReTrainer at runtime and writes per-rank JSONL records without
modifying the original repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback


DEFAULT_MINIONEREC_ROOT = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec"
DEFAULT_MODEL = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390"


def rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def is_rank0() -> bool:
    return rank() == 0


def dump_path() -> Path:
    out_dir = Path(os.environ["ORIG_DEBUG_DUMP_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"orig_rank{rank()}.jsonl"


def safe_json(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().float().cpu().item()) if obj.is_floating_point() else int(obj.detach().cpu().item())
        return tensor_summary(obj)
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


def write_event(payload: dict) -> None:
    payload = {"rank": rank(), **payload}
    with dump_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_json(payload), sort_keys=True) + "\n")


def tensor_summary(tensor: torch.Tensor | None) -> dict | None:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    t = tensor.detach()
    out = {"shape": list(t.shape), "dtype": str(t.dtype)}
    if t.numel() == 0:
        return out
    if t.is_floating_point():
        tf = t.float()
        out.update(
            mean=float(tf.mean().item()),
            std=float(tf.std(unbiased=False).item()) if tf.numel() > 1 else 0.0,
            min=float(tf.min().item()),
            max=float(tf.max().item()),
        )
    else:
        out.update(
            min=int(t.min().item()),
            max=int(t.max().item()),
            checksum=int(t.long().sum().item()),
        )
    return out


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def selected_params(model, substrings: list[str]):
    module = unwrap_model(model)
    for name, param in module.named_parameters():
        if any(substr in name for substr in substrings):
            yield name, param


def grad_norm(model) -> float:
    if hasattr(model, "get_global_grad_norm"):
        try:
            value = model.get_global_grad_norm()
            if value is not None:
                return float(value)
        except Exception:
            pass
    device = None
    total = None
    for _, param in unwrap_model(model).named_parameters():
        if param.grad is None:
            continue
        if device is None:
            device = param.grad.device
            total = torch.zeros((), device=device)
        total = total + param.grad.detach().float().norm(2).pow(2)
    if total is None:
        return 0.0
    return float(total.sqrt().item())


def param_snapshot(model, substrings: list[str]) -> dict:
    out = {}
    for name, param in selected_params(model, substrings):
        data = param.detach()
        item = {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "norm": float(data.float().norm().item()),
            "mean": float(data.float().mean().item()),
        }
        if param.grad is not None:
            item["grad_dtype"] = str(param.grad.dtype)
            item["grad_norm"] = float(param.grad.detach().float().norm().item())
        out[name] = item
    return out


def optimizer_lrs(optimizer) -> list[float]:
    """Return actual optimizer param-group LRs when available."""

    if optimizer is None:
        return []
    groups = getattr(optimizer, "param_groups", None)
    if groups is None:
        return []
    lrs = []
    for group in groups:
        try:
            lrs.append(float(group.get("lr", 0.0)))
        except (TypeError, ValueError):
            continue
    return lrs


def lr_scheduler_lrs(lr_scheduler) -> list[float]:
    if lr_scheduler is None:
        return []
    try:
        return [float(v) for v in lr_scheduler.get_last_lr()]
    except Exception:
        return []


def deepspeed_optimizer(model, trainer=None):
    engine = getattr(trainer, "deepspeed", None) if trainer is not None else None
    return getattr(engine, "optimizer", None) or getattr(model, "optimizer", None)


def tensor_sample(tensor: torch.Tensor, sample_size: int = 8192) -> torch.Tensor:
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() == 0:
        return flat.cpu().clone()
    if flat.numel() <= sample_size:
        return flat.cpu().clone()
    # Build indices on CPU to avoid float rounding/device assert on very large
    # DeepSpeed flat fp32 partitions.
    idx = torch.linspace(0, flat.numel() - 1, steps=sample_size, device="cpu")
    idx = idx.round().long().clamp_(0, flat.numel() - 1).to(flat.device)
    return flat.index_select(0, idx).cpu().clone()


def fp32_master_samples(optimizer, sample_size: int = 8192) -> list[dict]:
    groups = getattr(optimizer, "single_partition_of_fp32_groups", None)
    if not isinstance(groups, list):
        return []
    out = []
    for group_idx, tensor in enumerate(groups):
        if not isinstance(tensor, torch.Tensor):
            continue
        sample = tensor_sample(tensor, sample_size=sample_size)
        out.append(
            {
                "group_idx": group_idx,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sample": sample,
                "summary": tensor_summary(sample),
            }
        )
    return out


def sample_deltas(before: list[dict], after: list[dict]) -> list[dict]:
    by_group = {item["group_idx"]: item for item in before}
    out = []
    for item in after:
        group_idx = item["group_idx"]
        prev = by_group.get(group_idx)
        if prev is None:
            continue
        delta = item["sample"] - prev["sample"]
        out.append(
            {
                "group_idx": group_idx,
                "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
                "mean_abs": float(delta.abs().mean().item()) if delta.numel() else 0.0,
                "norm": float(delta.norm().item()) if delta.numel() else 0.0,
                "nonzero": int((delta != 0).sum().item()) if delta.numel() else 0,
                "numel": int(delta.numel()),
                "after_summary": item["summary"],
            }
        )
    return out


def strip_samples(samples: list[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in item.items()
            if key != "sample"
        }
        for item in samples
    ]


def summarize_inputs(inputs: Any) -> dict:
    if isinstance(inputs, list):
        return {
            "container": "list",
            "len": len(inputs),
            "items": [summarize_inputs(item) for item in inputs[:8]],
        }
    if not isinstance(inputs, dict):
        return {"container": type(inputs).__name__}
    return {
        "container": "dict",
        "keys": sorted(str(key) for key in inputs.keys()),
        "completion_mask": tensor_summary(inputs.get("completion_mask")),
        "advantages": tensor_summary(inputs.get("advantages")),
        "ref_per_token_logps": tensor_summary(inputs.get("ref_per_token_logps")),
        "completion_ids": tensor_summary(inputs.get("completion_ids")),
    }


class ProbeCallback(TrainerCallback):
    def __init__(self, substrings: list[str], max_steps: int):
        self.substrings = substrings
        self.max_steps = max_steps
        self.before_params: dict[str, torch.Tensor] = {}
        self.before_master_samples: list[dict] = []
        self.trainer = None

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        self.before_params = {
            name: param.detach().float().clone()
            for name, param in selected_params(model, self.substrings)
        }
        write_event(
            {
                "event": "step_begin",
                "global_step": int(state.global_step),
                "params": param_snapshot(model, self.substrings),
            }
        )

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        opt_state = {}
        for name, param in list(selected_params(model, self.substrings))[:1]:
            state_dict = getattr(optimizer, "state", {}).get(param, {}) if optimizer is not None else {}
            opt_state[name] = {
                key: str(value.dtype) if isinstance(value, torch.Tensor) else type(value).__name__
                for key, value in state_dict.items()
            }
        ds_info = {}
        opt = deepspeed_optimizer(model, trainer=self.trainer)
        if opt is not None:
            ds_info["optimizer_class"] = type(opt).__name__
            ds_info["actual_lrs"] = optimizer_lrs(opt)
            self.before_master_samples = fp32_master_samples(opt)
            ds_info["fp32_master_samples"] = strip_samples(self.before_master_samples)
            for attr in ("single_partition_of_fp32_groups", "bit16_groups"):
                groups = getattr(opt, attr, None)
                if groups is not None:
                    ds_info[attr] = [
                        {"shape": list(t.shape), "dtype": str(t.dtype)}
                        for t in groups[:4]
                        if isinstance(t, torch.Tensor)
                    ]
            for attr in (
                "averaged_gradients",
                "grad_partitions_flat_buffer",
                "single_partition_of_fp32_groups",
            ):
                value = getattr(opt, attr, None)
                if isinstance(value, list):
                    ds_info[f"{attr}_len"] = len(value)
                    ds_info[f"{attr}_sample"] = [
                        tensor_summary(item) for item in value[:2] if isinstance(item, torch.Tensor)
                    ]
                elif isinstance(value, torch.Tensor):
                    ds_info[attr] = tensor_summary(value)
        write_event(
            {
                "event": "pre_optimizer_step",
                "global_step": int(state.global_step),
                "args_learning_rate": float(args.learning_rate),
                "callback_optimizer_lrs": optimizer_lrs(optimizer),
                "scheduler_lrs": lr_scheduler_lrs(lr_scheduler),
                "grad_norm": grad_norm(model),
                "params": param_snapshot(model, self.substrings),
                "optimizer_state": opt_state,
                "deepspeed": ds_info,
            }
        )

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        opt = deepspeed_optimizer(model, trainer=self.trainer)
        after_master_samples = fp32_master_samples(opt) if opt is not None else []
        deltas = {}
        for name, param in selected_params(model, self.substrings):
            if name not in self.before_params:
                continue
            delta = param.detach().float() - self.before_params[name]
            deltas[name] = {
                "max_abs": float(delta.abs().max().item()),
                "mean_abs": float(delta.abs().mean().item()),
                "norm": float(delta.norm().item()),
                "param_dtype_after": str(param.dtype),
                "param_norm_after": float(param.detach().float().norm().item()),
            }
        write_event(
            {
                "event": "optimizer_step",
                "global_step": int(state.global_step),
                "callback_optimizer_lrs": optimizer_lrs(optimizer),
                "scheduler_lrs": lr_scheduler_lrs(lr_scheduler),
                "deepspeed_optimizer_lrs": optimizer_lrs(opt),
                "fp32_master_sample_deltas": sample_deltas(self.before_master_samples, after_master_samples),
                "deltas": deltas,
                "params": param_snapshot(model, self.substrings),
            }
        )

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        deltas = {}
        if model is not None:
            for name, param in selected_params(model, self.substrings):
                if name not in self.before_params:
                    continue
                delta = param.detach().float() - self.before_params[name]
                deltas[name] = {
                    "max_abs": float(delta.abs().max().item()),
                    "mean_abs": float(delta.abs().mean().item()),
                    "norm": float(delta.norm().item()),
                    "param_dtype_after": str(param.dtype),
                    "param_norm_after": float(param.detach().float().norm().item()),
                }
        write_event(
            {
                "event": "step_end",
                "global_step": int(state.global_step),
                "callback_optimizer_lrs": optimizer_lrs(optimizer),
                "scheduler_lrs": lr_scheduler_lrs(lr_scheduler),
                "deltas_since_step_begin": deltas,
                "params": param_snapshot(model, self.substrings) if model is not None else {},
            }
        )
        if int(state.global_step) >= self.max_steps:
            control.should_training_stop = True
            control.should_save = False
            control.should_evaluate = False


def patch_trainer(substrings: list[str], max_steps: int, synthetic_advantages: str):
    import minionerec_trainer

    original_training_step = minionerec_trainer.ReReTrainer.training_step
    original_init = minionerec_trainer.ReReTrainer.__init__
    original_compute_loss = minionerec_trainer.ReReTrainer.compute_loss
    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    except Exception:
        DeepSpeedZeroOptimizer = None

    def patched_training_step(self, model, inputs, num_items_in_batch=None):
        write_event(
            {
                "event": "training_step_begin",
                "trainer_global_step": int(getattr(self.state, "global_step", -1)),
                "current_gradient_accumulation_steps": int(getattr(self, "current_gradient_accumulation_steps", -1)),
                "inputs": summarize_inputs(inputs),
            }
        )
        loss = original_training_step(self, model, inputs, num_items_in_batch=num_items_in_batch)
        write_event(
            {
                "event": "training_step_end",
                "trainer_global_step": int(getattr(self.state, "global_step", -1)),
                "returned_loss": float(loss.detach().float().cpu().item()) if isinstance(loss, torch.Tensor) else None,
                "grad_norm_after_backward": grad_norm(model),
                "params": param_snapshot(model, substrings),
            }
        )
        return loss

    def patched_init(self, *args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", []) or [])
        probe_callback = ProbeCallback(substrings=substrings, max_steps=max_steps)
        callbacks.append(probe_callback)
        kwargs["callbacks"] = callbacks
        result = original_init(self, *args, **kwargs)
        probe_callback.trainer = self
        return result

    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        write_event(
            {
                "event": "compute_loss_begin",
                "trainer_global_step": int(getattr(self.state, "global_step", -1)),
                "inputs": summarize_inputs(inputs),
            }
        )
        if synthetic_advantages in {"ramp", "ones"} and isinstance(inputs, dict) and isinstance(inputs.get("advantages"), torch.Tensor):
            inputs = dict(inputs)
            batch_size = int(inputs["advantages"].shape[0])
            if synthetic_advantages == "ones":
                replacement = torch.ones_like(inputs["advantages"], dtype=torch.float32)
            elif batch_size == 1:
                replacement = torch.ones_like(inputs["advantages"], dtype=torch.float32)
            else:
                replacement = torch.linspace(
                    -1.0,
                    1.0,
                    steps=batch_size,
                    device=inputs["advantages"].device,
                    dtype=torch.float32,
                )
            inputs["advantages"] = replacement
            write_event(
                {
                    "event": "compute_loss_synthetic_advantages",
                    "trainer_global_step": int(getattr(self.state, "global_step", -1)),
                    "advantages": tensor_summary(inputs["advantages"]),
                }
            )
        loss = original_compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        write_event(
            {
                "event": "compute_loss_end",
                "trainer_global_step": int(getattr(self.state, "global_step", -1)),
                "loss": float(loss.detach().float().cpu().item()) if isinstance(loss, torch.Tensor) else None,
            }
        )
        return loss

    minionerec_trainer.ReReTrainer.training_step = patched_training_step
    minionerec_trainer.ReReTrainer.__init__ = patched_init
    minionerec_trainer.ReReTrainer.compute_loss = patched_compute_loss

    if DeepSpeedZeroOptimizer is not None and not getattr(DeepSpeedZeroOptimizer, "_h66_step_patched", False):
        original_zero_step = DeepSpeedZeroOptimizer.step

        def patched_zero_step(self, *args, **kwargs):
            before = fp32_master_samples(self)
            before_lrs = optimizer_lrs(self)
            result = original_zero_step(self, *args, **kwargs)
            after = fp32_master_samples(self)
            write_event(
                {
                    "event": "deepspeed_zero_step_internal",
                    "optimizer_class": type(self).__name__,
                    "before_lrs": before_lrs,
                    "after_lrs": optimizer_lrs(self),
                    "fp32_master_sample_deltas": sample_deltas(before, after),
                    "fp32_master_samples_after": strip_samples(after),
                }
            )
            return result

        DeepSpeedZeroOptimizer.step = patched_zero_step
        DeepSpeedZeroOptimizer._h66_step_patched = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minionerec-root", default=DEFAULT_MINIONEREC_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--synthetic-advantages", choices=("none", "ramp", "ones"), default="none")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Override rl.py GRPOConfig warmup_ratio (default 0.03). Use 0 for nonzero first-step LR probes.",
    )
    parser.add_argument(
        "--param-substr",
        action="append",
        default=["layers.0.self_attn.q_proj.weight", "layers.27.mlp.down_proj.weight"],
    )
    args = parser.parse_args()

    os.environ["ORIG_DEBUG_DUMP_DIR"] = args.dump_dir
    root = Path(args.minionerec_root).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)

    patch_trainer(
        substrings=args.param_substr,
        max_steps=args.max_steps,
        synthetic_advantages=args.synthetic_advantages,
    )

    import rl
    from trl import GRPOConfig

    # `rl.train()` always preloads an `llm_model` with `device_map="auto"` before
    # constructing ReReTrainer, but ranking rewards never use it. In a short
    # multi-rank probe that extra copy can OOM before the actual DeepSpeed model
    # is initialized, so bypass only this unused pre-load. ReReTrainer still
    # loads and trains the real model through MiniOneRec's original path.
    class _ProbeUnusedLLM:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")

    original_rl_from_pretrained = rl.AutoModelForCausalLM.from_pretrained

    def _probe_from_pretrained(*args, **kwargs):
        if kwargs.get("device_map") == "auto":
            return _ProbeUnusedLLM()
        return original_rl_from_pretrained(*args, **kwargs)

    rl.AutoModelForCausalLM.from_pretrained = _probe_from_pretrained

    if args.warmup_ratio is not None:
        _orig_grpo_init = GRPOConfig.__init__

        def _grpo_init_warmup(self, *a, **kw):
            kw["warmup_ratio"] = float(args.warmup_ratio)
            return _orig_grpo_init(self, *a, **kw)

        GRPOConfig.__init__ = _grpo_init_warmup
        # rl.py binds GRPOConfig at import time; patch the name it uses too.
        rl.GRPOConfig = GRPOConfig

    train = rl.train

    category = "Industrial_and_Scientific"
    train(
        model_path=args.model_path,
        seed=42,
        train_file=str(root / "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"),
        eval_file=str(root / "data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"),
        info_file=str(root / "data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"),
        category=category,
        output_dir=str(Path(args.dump_dir) / "orig_output"),
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        temperature=1.0,
        eval_step=999999,
        num_generations=args.num_generations,
        num_train_epochs=1,
        learning_rate=args.learning_rate,
        beta=args.beta,
        beam_search=True,
        test_during_training=False,
        dynamic_sampling=False,
        mask_all_zero=False,
        sync_ref_model=True,
        test_beam=20,
        reward_type="ranking",
        add_gt=False,
        dapo=False,
        sid_index_path=str(root / "data/Amazon/index/Industrial_and_Scientific.index.json"),
        item_meta_path=str(root / "data/Amazon/index/Industrial_and_Scientific.item.json"),
    )


if __name__ == "__main__":
    main()
