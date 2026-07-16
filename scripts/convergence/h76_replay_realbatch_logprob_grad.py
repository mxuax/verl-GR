#!/usr/bin/env python3
"""Replay H76 real MiniOneRec batch through the original padded logprob/loss path."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


DEFAULT_MODEL = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390"
DEFAULT_PARAM_SUBSTRINGS = ("lm_head.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.27.mlp.down_proj.weight")


def stats(a: torch.Tensor, b: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> dict[str, Any]:
    x = a.detach().float()
    if b is not None:
        x = x - b.detach().float()
    if mask is not None:
        x = x[mask.bool()]
    out: dict[str, Any] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if x.numel() == 0:
        return out
    out.update(
        mean=float(x.mean().item()),
        std=float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        min=float(x.min().item()),
        max=float(x.max().item()),
        max_abs=float(x.abs().max().item()),
        mean_abs=float(x.abs().mean().item()),
    )
    return out


def per_token_logps(model, input_ids, attention_mask, logits_to_keep: int, position_ids=None) -> torch.Tensor:
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits_to_keep": logits_to_keep + 1,
    }
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    logits = model(**kwargs).logits
    logits = logits[:, :-1, :]
    labels = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def minionerec_loss(logps, ref, adv, mask, beta: float) -> torch.Tensor:
    kl = torch.exp(ref - logps) - (ref - logps) - 1
    pg = torch.exp(logps - logps.detach()) * adv
    per_token_loss = -(pg - beta * kl)
    return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()


def selected_gradients(model, substrings: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out = {}
    for name, param in model.named_parameters():
        if param.grad is not None and any(s in name for s in substrings):
            out[name] = param.grad.detach().float().cpu().clone()
    return out


def best_match_by_uid(replay: torch.Tensor, ref: torch.Tensor, uids: list[str] | None) -> dict[str, Any] | None:
    if not uids or len(uids) != replay.shape[0]:
        return None
    groups: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[str(uid)].append(i)
    best = []
    identity = []
    for idxs in groups.values():
        r = replay[idxs].float()
        d = ref[idxs].float()
        # pairwise mean abs over response length
        dist = (r[:, None, :] - d[None, :, :]).abs().mean(dim=-1)
        best.extend(dist.min(dim=1).values.tolist())
        identity.extend(dist.diag().tolist())
    best_t = torch.tensor(best)
    identity_t = torch.tensor(identity)
    return {
        "identity_mean_abs": float(identity_t.mean().item()),
        "identity_max_abs": float(identity_t.max().item()),
        "best_same_uid_mean_abs": float(best_t.mean().item()),
        "best_same_uid_max_abs": float(best_t.max().item()),
        "improvement_ratio": float(identity_t.mean().item() / best_t.mean().item()) if best_t.mean().item() != 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--disable-flash-sdp", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--position-mode",
        choices=("leftpad_no_position", "leftpad_with_position", "compact_with_position", "compact_mask_eos_as_pad"),
        default="leftpad_no_position",
    )
    parser.add_argument("--grad-substr", action="append", default=list(DEFAULT_PARAM_SUBSTRINGS))
    args = parser.parse_args()

    if args.disable_flash_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    payload = torch.load(args.fixture, map_location="cpu", weights_only=False)
    tensors = payload["tensor"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.train()

    input_ids_cpu = tensors["input_ids"]
    attention_mask_cpu = tensors["attention_mask"]
    position_ids_cpu = tensors.get("position_ids")
    if args.position_mode.startswith("compact"):
        pad_token = int(input_ids_cpu[0][attention_mask_cpu[0].bool()][-1].item())
        rows = []
        pos_rows = []
        max_len = int(attention_mask_cpu.sum(dim=1).max().item())
        for ids, mask in zip(input_ids_cpu, attention_mask_cpu, strict=True):
            valid = ids[mask.bool()]
            pad_len = max_len - valid.numel()
            rows.append(torch.cat([valid, torch.full((pad_len,), pad_token, dtype=ids.dtype)]))
            pos_rows.append(torch.cat([torch.arange(valid.numel(), dtype=torch.long), torch.zeros((pad_len,), dtype=torch.long)]))
        input_ids_cpu = torch.stack(rows, dim=0)
        position_ids_cpu = torch.stack(pos_rows, dim=0)
        if args.position_mode == "compact_mask_eos_as_pad":
            attention_mask_cpu = (input_ids_cpu != pad_token).long()
        else:
            attention_mask_cpu = torch.arange(max_len).unsqueeze(0) < torch.tensor(
                [int(m.sum().item()) for m in tensors["attention_mask"]]
            ).unsqueeze(1)
            attention_mask_cpu = attention_mask_cpu.long()

    input_ids = input_ids_cpu.to(device)
    attention_mask = attention_mask_cpu.to(device)
    position_ids = None
    if args.position_mode in {"leftpad_with_position", "compact_with_position", "compact_mask_eos_as_pad"}:
        if position_ids_cpu is None:
            raise KeyError("position_ids missing from fixture")
        if args.position_mode == "leftpad_with_position":
            position_ids = position_ids_cpu.to(device)
        else:
            position_ids = position_ids_cpu.to(device)
    response_mask = tensors["response_mask"].to(device).float()
    ref = tensors["ref_log_prob"].to(device).float()
    adv = tensors["advantages"].to(device).float()
    logits_to_keep = tensors["responses"].shape[1]

    logp_chunks = []
    losses = []
    model.zero_grad(set_to_none=True)
    autocast_enabled = args.dtype == "bf16" and device.type == "cuda"
    for start in range(0, input_ids.shape[0], args.micro_batch_size):
        end = min(start + args.micro_batch_size, input_ids.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logps = per_token_logps(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                logits_to_keep,
                None if position_ids is None else position_ids[start:end],
            )
            loss = minionerec_loss(
                logps,
                ref[start:end],
                adv[start:end],
                response_mask[start:end],
                args.beta,
            )
        # Match full-batch mean over sequences while using micro-batches.
        (loss * ((end - start) / input_ids.shape[0])).backward()
        logp_chunks.append(logps.detach().float().cpu())
        losses.append(float(loss.detach().float().cpu().item()))

    replay_logps = torch.cat(logp_chunks, dim=0)
    ref_cpu = ref.detach().cpu()
    mask_cpu = response_mask.detach().cpu()
    uids = None
    if isinstance(payload.get("non_tensor"), dict) and "uid" in payload["non_tensor"]:
        uids = [str(x) for x in payload["non_tensor"]["uid"]]
    grads = selected_gradients(model, tuple(args.grad_substr))
    grad_report = {name: stats(grad) for name, grad in grads.items()}

    report = {
        "fixture": args.fixture,
        "model": args.model,
        "trust_remote_code": bool(args.trust_remote_code),
        "attn_implementation": args.attn_implementation,
        "dtype": args.dtype,
        "disable_flash_sdp": bool(args.disable_flash_sdp),
        "position_mode": args.position_mode,
        "micro_batch_size": args.micro_batch_size,
        "loss_micro_mean": float(sum(losses) / len(losses)),
        "logprob_vs_dumped_ref": stats(replay_logps, ref_cpu, mask_cpu),
        "best_match_by_uid": best_match_by_uid(replay_logps, ref_cpu, uids),
        "replay_logprob": stats(replay_logps, mask=mask_cpu),
        "dumped_ref_logprob": stats(ref_cpu, mask=mask_cpu),
        "gradients": grad_report,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
