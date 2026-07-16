#!/usr/bin/env python3
"""Run the fixed MiniOneRec update fixture through verl-GR style DDP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from verl_gr.workers.optimizer import FP32MasterOptimizer, build_actor_optimizer
from verl.workers.config.optimizer import FSDPOptimizerConfig


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
        out.update(norm=float(f.norm().item()), mean=float(f.mean().item()), max_abs=float(f.abs().max().item()))
    else:
        out.update(checksum=int(d.long().sum().item()), min=int(d.min().item()), max=int(d.max().item()))
    return out


def selected_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    module = model.module if hasattr(model, "module") else model
    return [(name, p) for name, p in module.named_parameters() if any(s in name for s in PARAM_SUBSTRINGS)]


def global_grad_norm(model) -> float:
    total = None
    module = model.module if hasattr(model, "module") else model
    for p in module.parameters():
        if p.grad is None:
            continue
        v = p.grad.detach().float().norm().pow(2)
        total = v if total is None else total + v
    return float(total.sqrt().item()) if total is not None else 0.0


def selected_grad_norm(model) -> float:
    total = None
    for _, p in selected_params(model):
        if p.grad is None:
            continue
        v = p.grad.detach().float().norm().pow(2)
        total = v if total is None else total + v
    return float(total.sqrt().item()) if total is not None else 0.0


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
    parser.add_argument(
        "--optimizer",
        default="adamw_torch_fp32_master",
        choices=("adamw_torch_fp32_master", "paged_adamw_32bit", "AdamW"),
        help="MiniOneRec production path uses paged_adamw_32bit; H75 default kept for back-compat.",
    )
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
        help="Divide each micro loss before backward; use num local micros for verl production parity.",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    raw_fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    model_path = args.model or raw_fixture["model"]
    fixture = move_fixture(raw_fixture, device)
    if args.rank_slice:
        world_size = dist.get_world_size()
        n_rows = int(fixture["prompt_ids"].shape[0])
        if n_rows % world_size != 0:
            raise ValueError(f"fixture rows {n_rows} not divisible by world_size {world_size}")
        per_rank = n_rows // world_size
        fixture = slice_fixture(fixture, rank() * per_rank, (rank() + 1) * per_rank)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model.train()
    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    optim_cfg = FSDPOptimizerConfig(
        optimizer=args.optimizer,
        lr=args.base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        clip_grad=args.clip_grad,
    )
    if args.optimizer == "adamw_torch_fp32_master":
        optimizer = FP32MasterOptimizer(
            ddp.module.parameters(),
            torch.optim.AdamW,
            {"lr": args.base_lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        )
        backend_name = "verl_ddp_fp32_master"
    else:
        optimizer = build_actor_optimizer(ddp.module.parameters(), optim_cfg)
        backend_name = f"verl_ddp_{args.optimizer}"
    out = Path(args.out_dir) / f"verl_ddp_rank{rank()}.jsonl"

    for step in range(args.steps):
        if args.force_lr is not None:
            lr = float(args.force_lr)
        elif args.warmup_steps <= 0:
            lr = args.base_lr
        else:
            lr = args.base_lr * step / max(1, args.warmup_steps)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        before_visible = {name: p.detach().float().clone() for name, p in selected_params(ddp)}
        before_master = {}
        master_lookup = getattr(optimizer, "master_param_for_visible", None)
        for name, p in selected_params(ddp):
            if master_lookup is None:
                break
            master = master_lookup(p)
            if master is not None:
                before_master[name] = master.detach().float().clone()
        if args.micro_batch_size and args.micro_batch_size > 0:
            local_n = int(fixture["prompt_ids"].shape[0])
            if local_n % args.micro_batch_size != 0:
                raise ValueError(f"local rows {local_n} not divisible by micro_batch_size {args.micro_batch_size}")
            losses = []
            for start in range(0, local_n, args.micro_batch_size):
                micro = slice_fixture(fixture, start, start + args.micro_batch_size)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    micro_loss = minionerec_loss(ddp, micro, args.beta)
                    scaled_loss = micro_loss / float(args.loss_divisor)
                scaled_loss.backward()
                losses.append(float(micro_loss.detach().float().item()))
            loss_value = float(sum(losses) / len(losses))
            loss = torch.tensor(loss_value, device=device)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = minionerec_loss(ddp, fixture, args.beta)
            loss.backward()
        pre_clip = {
            name: {"param": tensor_summary(p.detach()), "grad": tensor_summary(p.grad)}
            for name, p in selected_params(ddp)
        }
        pre_global = global_grad_norm(ddp)
        returned_preclip = torch.nn.utils.clip_grad_norm_(ddp.module.parameters(), max_norm=args.clip_grad)
        post_clip = {
            name: {"grad": tensor_summary(p.grad)}
            for name, p in selected_params(ddp)
        }
        post_global = global_grad_norm(ddp)
        optimizer.step()
        visible_deltas = {}
        master_deltas = {}
        for name, p in selected_params(ddp):
            visible_delta = p.detach().float() - before_visible[name]
            visible_deltas[name] = {
                "max_abs": float(visible_delta.abs().max().item()),
                "mean_abs": float(visible_delta.abs().mean().item()),
                "norm": float(visible_delta.norm().item()),
                "param_dtype_after": str(p.dtype),
                "param_norm_after": float(p.detach().float().norm().item()),
            }
            if master_lookup is None or name not in before_master:
                continue
            master = master_lookup(p)
            if master is not None:
                master_delta = master.detach().float() - before_master[name]
                master_deltas[name] = {
                    "max_abs": float(master_delta.abs().max().item()),
                    "mean_abs": float(master_delta.abs().mean().item()),
                    "norm": float(master_delta.norm().item()),
                    "param_dtype_after": str(master.dtype),
                    "param_norm_after": float(master.detach().float().norm().item()),
                }
        inner = getattr(optimizer, "inner_optimizer", optimizer)
        write_jsonl(
            out,
            {
                "backend": backend_name,
                "rank": rank(),
                "step": step,
                "lr": lr,
                "loss": float(loss.detach().float().item()),
                "rank_slice": bool(args.rank_slice),
                "local_rows": int(fixture["prompt_ids"].shape[0]),
                "micro_batch_size": int(args.micro_batch_size),
                "loss_divisor": float(args.loss_divisor),
                "pre_clip_global_grad_norm": pre_global,
                "returned_preclip_grad_norm": float(returned_preclip.detach().float().item()),
                "post_clip_global_grad_norm": post_global,
                "selected_grad_norm_post_clip": selected_grad_norm(ddp),
                "pre_clip": pre_clip,
                "post_clip": post_clip,
                "optimizer_class": type(optimizer).__name__,
                "optimizer_module": type(optimizer).__module__,
                "inner_optimizer_class": type(inner).__name__,
                "inner_optimizer_module": type(inner).__module__,
                "visible_deltas": visible_deltas,
                "fp32_master_deltas": master_deltas,
            },
        )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
