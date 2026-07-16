#!/usr/bin/env python3
"""Compare clamped vs unclamped MiniOneRec KL on the same replay batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def stats(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    x = tensor.detach().float()
    if mask is not None:
        x = x[mask.bool()]
    out: dict[str, Any] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if x.numel() == 0:
        return out
    out.update(
        mean=float(x.mean().item()),
        std=float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        min=float(x.min().item()),
        max=float(x.max().item()),
        mean_abs=float(x.abs().mean().item()),
        max_abs=float(x.abs().max().item()),
    )
    return out


def masked_seq_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def per_token_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    position_ids: torch.Tensor | None,
) -> torch.Tensor:
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits_to_keep": logits_to_keep + 1,
    }
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    logits = model(**kwargs).logits[:, :-1, :]
    labels = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def low_var_kl(ref_logps: torch.Tensor, current_logps: torch.Tensor, *, clamp: bool) -> torch.Tensor:
    delta = ref_logps - current_logps
    kld = torch.exp(delta) - delta - 1.0
    if clamp:
        kld = torch.clamp(kld, min=-10.0, max=10.0)
    return kld.contiguous()


def build_inputs(tensors: dict[str, torch.Tensor], device: torch.device, use_position_ids: bool):
    input_ids_cpu = tensors["input_ids"]
    input_source = "input_ids"
    if (
        isinstance(input_ids_cpu, torch.Tensor)
        and input_ids_cpu.is_nested
        and isinstance(tensors.get("prompts"), torch.Tensor)
        and isinstance(tensors.get("responses"), torch.Tensor)
        and not tensors["prompts"].is_nested
        and not tensors["responses"].is_nested
    ):
        input_ids_cpu = torch.cat([tensors["prompts"], tensors["responses"]], dim=1)
        input_source = "dense_prompts_plus_responses"

    position_ids = tensors.get("position_ids")
    if use_position_ids:
        if position_ids is None:
            raise KeyError("position_ids missing from fixture")
        position_ids = position_ids.to(device)
    else:
        position_ids = None

    return (
        input_source,
        input_ids_cpu.to(device),
        tensors["attention_mask"].to(device),
        position_ids,
    )


def gather_current_logps(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    logits_to_keep: int,
    micro_batch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    autocast_enabled = dtype == torch.bfloat16 and input_ids.device.type == "cuda"
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], micro_batch_size):
            end = min(start + micro_batch_size, input_ids.shape[0])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                chunks.append(
                    per_token_logps(
                        model,
                        input_ids[start:end],
                        attention_mask[start:end],
                        logits_to_keep,
                        None if position_ids is None else position_ids[start:end],
                    ).float().cpu()
                )
    return torch.cat(chunks, dim=0)


def grad_norm_for_loss(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    ref: torch.Tensor,
    adv: torch.Tensor,
    mask: torch.Tensor,
    logits_to_keep: int,
    micro_batch_size: int,
    dtype: torch.dtype,
    beta: float,
    mode: str,
) -> tuple[float, float]:
    model.zero_grad(set_to_none=True)
    total_loss_value = 0.0
    autocast_enabled = dtype == torch.bfloat16 and input_ids.device.type == "cuda"

    for start in range(0, input_ids.shape[0], micro_batch_size):
        end = min(start + micro_batch_size, input_ids.shape[0])
        weight = (end - start) / input_ids.shape[0]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logps = per_token_logps(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                logits_to_keep,
                None if position_ids is None else position_ids[start:end],
            )
            mb_ref = ref[start:end]
            mb_adv = adv[start:end]
            mb_mask = mask[start:end]
            pg = torch.exp(logps - logps.detach()) * mb_adv
            pg_loss = masked_seq_mean(-pg, mb_mask).mean()
            kl_clamped = masked_seq_mean(low_var_kl(mb_ref, logps, clamp=True), mb_mask).mean()
            kl_unclamped = masked_seq_mean(low_var_kl(mb_ref, logps, clamp=False), mb_mask).mean()

            if mode == "pg":
                loss = pg_loss
            elif mode == "kl_clamped":
                loss = beta * kl_clamped
            elif mode == "kl_unclamped":
                loss = beta * kl_unclamped
            elif mode == "total_clamped":
                loss = pg_loss + beta * kl_clamped
            elif mode == "total_unclamped":
                loss = pg_loss + beta * kl_unclamped
            else:
                raise ValueError(f"unknown mode: {mode}")

        (loss * weight).backward()
        total_loss_value += float(loss.detach().float().item()) * weight

    norm_sq = torch.tensor(0.0, device=input_ids.device)
    for param in model.parameters():
        if param.grad is not None:
            norm_sq += param.grad.detach().float().pow(2).sum()
    return total_loss_value, float(torch.sqrt(norm_sq).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--disable-flash-sdp", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--use-position-ids", action="store_true")
    args = parser.parse_args()

    if args.disable_flash_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    payload = torch.load(args.fixture, map_location="cpu", weights_only=False)
    tensors = payload["tensor"]
    required = ["input_ids", "attention_mask", "responses", "response_mask", "ref_log_prob", "advantages"]
    missing = [key for key in required if key not in tensors]
    if missing:
        raise KeyError(f"fixture missing tensor keys: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": bool(args.trust_remote_code)}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.train()

    input_source, input_ids, attention_mask, position_ids = build_inputs(tensors, device, args.use_position_ids)
    logits_to_keep = int(tensors["responses"].shape[1])
    mask = tensors["response_mask"].to(device).float()
    ref = tensors["ref_log_prob"].to(device).float()
    adv = tensors["advantages"].to(device).float()
    old = tensors.get("old_log_probs")
    rollout = tensors.get("rollout_log_probs")
    if old is not None:
        old = old.to(device).float()
    if rollout is not None:
        rollout = rollout.to(device).float()

    current_cpu = gather_current_logps(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logits_to_keep=logits_to_keep,
        micro_batch_size=args.micro_batch_size,
        dtype=dtype,
    )
    current = current_cpu.to(device)
    kl_unclamped = low_var_kl(ref, current, clamp=False)
    kl_clamped = low_var_kl(ref, current, clamp=True)
    pg = torch.exp(current - current.detach()) * adv
    lengths = mask.sum(dim=-1)

    grad_modes = ["pg", "kl_clamped", "kl_unclamped", "total_clamped", "total_unclamped"]
    grads = {}
    for mode in grad_modes:
        value, norm = grad_norm_for_loss(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ref=ref,
            adv=adv,
            mask=mask,
            logits_to_keep=logits_to_keep,
            micro_batch_size=args.micro_batch_size,
            dtype=dtype,
            beta=args.beta,
            mode=mode,
        )
        grads[mode] = {"loss_value": value, "grad_global_norm": norm}

    by_length = {}
    for length in sorted(lengths.unique().tolist()):
        row_mask = lengths == length
        token_mask = mask[row_mask]
        by_length[str(int(length))] = {
            "rows": int(row_mask.sum().item()),
            "tokens": int(token_mask.sum().item()),
            "advantage": stats(adv[row_mask], token_mask),
            "current_minus_ref": stats(current[row_mask] - ref[row_mask], token_mask),
            "current_minus_old": stats(current[row_mask] - old[row_mask], token_mask) if old is not None else None,
            "kl_unclamped": stats(kl_unclamped[row_mask], token_mask),
            "kl_clamped": stats(kl_clamped[row_mask], token_mask),
            "kl_saturated_frac": float(((kl_unclamped[row_mask] > 10.0) & token_mask.bool()).sum().item() / token_mask.sum().clamp_min(1.0).item()),
        }

    report: dict[str, Any] = {
        "fixture": args.fixture,
        "fixture_step": payload.get("global_step"),
        "fixture_tag": payload.get("tag"),
        "model": args.model,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "disable_flash_sdp": bool(args.disable_flash_sdp),
        "input_source": input_source,
        "beta": float(args.beta),
        "n_rows": int(current.shape[0]),
        "mask_tokens": int(mask.sum().item()),
        "length_counts": {str(int(v)): int((lengths == v).sum().item()) for v in sorted(lengths.unique().tolist())},
        "current_logprob": stats(current, mask),
        "ref_logprob": stats(ref, mask),
        "old_logprob": stats(old, mask) if old is not None else None,
        "rollout_logprob": stats(rollout, mask) if rollout is not None else None,
        "current_minus_ref": stats(current - ref, mask),
        "current_minus_old": stats(current - old, mask) if old is not None else None,
        "old_minus_ref": stats(old - ref, mask) if old is not None else None,
        "kl_unclamped_token": stats(kl_unclamped, mask),
        "kl_clamped_token": stats(kl_clamped, mask),
        "kl_unclamped_seq": stats(masked_seq_mean(kl_unclamped, mask)),
        "kl_clamped_seq": stats(masked_seq_mean(kl_clamped, mask)),
        "kl_saturated_frac": float(((kl_unclamped > 10.0) & mask.bool()).sum().item() / mask.sum().clamp_min(1.0).item()),
        "pg_seq": stats(masked_seq_mean(pg, mask)),
        "loss_seq_pg": float(masked_seq_mean(-pg, mask).mean().item()),
        "loss_seq_kl_clamped_contribution": float(args.beta * masked_seq_mean(kl_clamped, mask).mean().item()),
        "loss_seq_kl_unclamped_contribution": float(args.beta * masked_seq_mean(kl_unclamped, mask).mean().item()),
        "gradients": grads,
        "by_length": by_length,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
