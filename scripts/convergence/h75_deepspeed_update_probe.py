#!/usr/bin/env python3
"""Run a fixed MiniOneRec update fixture through DeepSpeed ZeRO-2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import deepspeed
import torch
from transformers import AutoModelForCausalLM


PARAM_SUBSTRINGS = ("layers.0.self_attn.q_proj.weight", "layers.27.mlp.down_proj.weight")


def rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def tensor_summary(t: torch.Tensor | None) -> dict[str, Any] | None:
    if t is None:
        return None
    d = t.detach()
    out = {"shape": list(d.shape), "dtype": str(d.dtype)}
    if d.numel() == 0:
        return out
    if d.is_floating_point():
        f = d.float()
        out.update(
            norm=float(f.norm().item()),
            mean=float(f.mean().item()),
            max_abs=float(f.abs().max().item()),
        )
    else:
        out.update(checksum=int(d.long().sum().item()), min=int(d.min().item()), max=int(d.max().item()))
    return out


def selected_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    module = model.module if hasattr(model, "module") else model
    return [(name, p) for name, p in module.named_parameters() if any(s in name for s in PARAM_SUBSTRINGS)]


def grad_norm(model) -> float:
    total = None
    for _, p in selected_params(model):
        if p.grad is None:
            continue
        v = p.grad.detach().float().norm().pow(2)
        total = v if total is None else total + v
    return float(total.sqrt().item()) if total is not None else 0.0


def global_grad_norm(model) -> float:
    total = None
    module = model.module if hasattr(model, "module") else model
    for p in module.parameters():
        if p.grad is None:
            continue
        v = p.grad.detach().float().norm().pow(2)
        total = v if total is None else total + v
    return float(total.sqrt().item()) if total is not None else 0.0


def deepspeed_grad_norm(engine, optimizer) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, obj in (("engine", engine), ("optimizer", optimizer)):
        for method_name in ("get_global_grad_norm", "get_grad_norm"):
            method = getattr(obj, method_name, None)
            if method is None:
                continue
            try:
                value = method()
            except Exception as exc:
                out[f"{label}.{method_name}.error"] = type(exc).__name__
                continue
            if isinstance(value, torch.Tensor):
                out[f"{label}.{method_name}"] = float(value.detach().float().item())
            elif value is not None:
                out[f"{label}.{method_name}"] = float(value)
    for attr in ("averaged_gradients", "grad_partitions_flat_buffer"):
        value = getattr(optimizer, attr, None)
        if isinstance(value, list):
            out[f"{attr}_len"] = len(value)
            out[f"{attr}_summaries"] = [tensor_summary(item) for item in value[:4] if isinstance(item, torch.Tensor)]
        elif isinstance(value, torch.Tensor):
            out[attr] = tensor_summary(value)
    return out


def tensor_sample(t: torch.Tensor, sample_size: int = 8192) -> torch.Tensor:
    flat = t.detach().float().reshape(-1)
    if flat.numel() <= sample_size:
        return flat.cpu().clone()
    idx = torch.linspace(0, flat.numel() - 1, steps=sample_size, device="cpu")
    idx = idx.round().long().clamp_(0, flat.numel() - 1).to(flat.device)
    return flat.index_select(0, idx).cpu().clone()


def fp32_master_samples(optimizer, sample_size: int = 8192) -> list[dict[str, Any]]:
    groups = getattr(optimizer, "single_partition_of_fp32_groups", None)
    out = []
    if not isinstance(groups, list):
        return out
    for group_idx, tensor in enumerate(groups):
        if not isinstance(tensor, torch.Tensor):
            continue
        sample = tensor_sample(tensor, sample_size=sample_size)
        out.append({"group_idx": group_idx, "sample": sample, "summary": tensor_summary(sample)})
    return out


def sample_deltas(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prev = {item["group_idx"]: item for item in before}
    out = []
    for item in after:
        old = prev.get(item["group_idx"])
        if old is None:
            continue
        delta = item["sample"] - old["sample"]
        out.append(
            {
                "group_idx": item["group_idx"],
                "max_abs": float(delta.abs().max().item()),
                "mean_abs": float(delta.abs().mean().item()),
                "norm": float(delta.norm().item()),
                "nonzero": int((delta != 0).sum().item()),
                "numel": int(delta.numel()),
            }
        )
    return out


def strip_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in item.items() if k != "sample"} for item in samples]


def per_token_logps(model, input_ids, attention_mask, logits_to_keep: int) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]
    labels = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def minionerec_loss(model, fixture: dict[str, torch.Tensor], beta: float) -> torch.Tensor:
    prompt_ids = fixture["prompt_ids"]
    prompt_mask = fixture["prompt_mask"]
    completion_ids = fixture["completion_ids"]
    completion_mask = fixture["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logps = per_token_logps(model, input_ids, attention_mask, completion_ids.shape[1])
    ref = fixture["ref_per_token_logps"].to(logps.device)
    adv = fixture["advantages"].to(logps.device)
    kl = torch.exp(ref - logps) - (ref - logps) - 1
    pg = torch.exp(logps - logps.detach()) * adv.unsqueeze(1)
    per_token_loss = -(pg - beta * kl)
    mask = completion_mask.to(logps.device)
    return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()


def move_fixture(fixture: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in fixture.items() if isinstance(v, torch.Tensor)}


def slice_fixture(fixture: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in fixture.items():
        out[k] = v[start:end] if isinstance(v, torch.Tensor) and v.shape[:1] == fixture["prompt_ids"].shape[:1] else v
    return out


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def deepspeed_backward(engine, loss: torch.Tensor) -> None:
    try:
        engine.backward(loss, scale_wrt_gas=False)
    except TypeError:
        engine.backward(loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=99)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument(
        "--force-lr",
        type=float,
        default=None,
        help="If set, use this LR on every step (skips warmup schedule).",
    )
    parser.add_argument("--rank-slice", action="store_true", help="Use a disjoint fixture slice per distributed rank.")
    parser.add_argument("--micro-batch-size", type=int, default=0, help="If >0, split local fixture into equal microbatches.")
    parser.add_argument(
        "--loss-divisor",
        type=float,
        default=1.0,
        help="Divide each micro loss before backward; original HF trainer uses GAS=2.",
    )
    args = parser.parse_args()

    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    raw_fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    model_path = args.model or raw_fixture["model"]
    fixture = move_fixture(raw_fixture, device)
    if args.rank_slice:
        world_size = torch.distributed.get_world_size()
        n_rows = int(fixture["prompt_ids"].shape[0])
        if n_rows % world_size != 0:
            raise ValueError(f"fixture rows {n_rows} not divisible by world_size {world_size}")
        per_rank = n_rows // world_size
        fixture = slice_fixture(fixture, rank() * per_rank, (rank() + 1) * per_rank)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    ds_config = {
        "train_micro_batch_size_per_gpu": int(args.micro_batch_size or fixture["prompt_ids"].shape[0]),
        "gradient_accumulation_steps": int(
            max(1, fixture["prompt_ids"].shape[0] // int(args.micro_batch_size or fixture["prompt_ids"].shape[0]))
        ),
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": False,
            "reduce_scatter": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
        },
        "gradient_clipping": args.clip_grad,
    }
    torch_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=torch_optimizer,
        config=ds_config,
    )
    out = Path(args.out_dir) / f"deepspeed_rank{rank()}.jsonl"

    for step in range(args.steps):
        if args.force_lr is not None:
            lr = float(args.force_lr)
        elif args.warmup_steps <= 0:
            lr = args.base_lr
        else:
            lr = args.base_lr * step / max(1, args.warmup_steps)
        set_lr(optimizer, lr)
        before_params = {name: p.detach().float().clone() for name, p in selected_params(engine)}
        before_master = fp32_master_samples(optimizer)
        if args.micro_batch_size and args.micro_batch_size > 0:
            local_n = int(fixture["prompt_ids"].shape[0])
            if local_n % args.micro_batch_size != 0:
                raise ValueError(f"local rows {local_n} not divisible by micro_batch_size {args.micro_batch_size}")
            losses = []
            for start in range(0, local_n, args.micro_batch_size):
                micro = slice_fixture(fixture, start, start + args.micro_batch_size)
                loss = minionerec_loss(engine, micro, args.beta)
                deepspeed_backward(engine, loss / float(args.loss_divisor))
                # DeepSpeed increments its GAS micro-step in engine.step(); it
                # only updates weights on the accumulation boundary.
                engine.step()
                losses.append(float(loss.detach().float().item()))
            loss = torch.tensor(float(sum(losses) / len(losses)), device=device)
        else:
            loss = minionerec_loss(engine, fixture, args.beta)
            deepspeed_backward(engine, loss)
            engine.step()
        pre_clip = {
            name: {"param": tensor_summary(p.detach()), "grad": tensor_summary(p.grad)}
            for name, p in selected_params(engine)
        }
        pre_global = global_grad_norm(engine)
        ds_grad_norm = deepspeed_grad_norm(engine, optimizer)
        after_master = fp32_master_samples(optimizer)
        deltas = {}
        for name, p in selected_params(engine):
            delta = p.detach().float() - before_params[name]
            deltas[name] = {
                "max_abs": float(delta.abs().max().item()),
                "mean_abs": float(delta.abs().mean().item()),
                "norm": float(delta.norm().item()),
                "param_dtype_after": str(p.dtype),
                "param_norm_after": float(p.detach().float().norm().item()),
            }
        write_jsonl(
            out,
            {
                "backend": "deepspeed_zero2",
                "rank": rank(),
                "step": step,
                "lr": lr,
                "loss": float(loss.detach().float().item()),
                "rank_slice": bool(args.rank_slice),
                "local_rows": int(fixture["prompt_ids"].shape[0]),
                "micro_batch_size": int(args.micro_batch_size),
                "loss_divisor": float(args.loss_divisor),
                "pre_clip_global_grad_norm_local_view": pre_global,
                "deepspeed_grad_norm": ds_grad_norm,
                "selected_grad_norm_local_view": grad_norm(engine),
                "pre_clip": pre_clip,
                "optimizer_class": type(optimizer).__name__,
                "fp32_master_samples": strip_samples(after_master),
                "fp32_master_sample_deltas": sample_deltas(before_master, after_master),
                "visible_deltas": deltas,
            },
        )


if __name__ == "__main__":
    main()
