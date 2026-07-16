#!/usr/bin/env python3
"""Replay a dumped MiniOneRec batch and report actor/ref/rollout KL contracts."""

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
        max_abs=float(x.abs().max().item()),
        mean_abs=float(x.abs().mean().item()),
    )
    return out


def diff_stats(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    return stats(a.detach().float() - b.detach().float(), mask)


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


def masked_seq_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def minionerec_terms(
    *,
    current_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    low_var_kl = torch.exp(ref_logps - current_logps) - (ref_logps - current_logps) - 1
    # Original MiniOneRec REINFORCE uses exp(logp - logp.detach()), not
    # exp(logp - old_logp). old_logps are kept only for rollout/ref diagnostics.
    pg = torch.exp(current_logps - current_logps.detach()) * advantages
    per_token_loss = -(pg - beta * low_var_kl)
    return {
        "low_var_kl": low_var_kl,
        "pg": pg,
        "per_token_loss": per_token_loss,
        "seq_kl": masked_seq_mean(low_var_kl, mask),
        "seq_pg": masked_seq_mean(pg, mask),
        "seq_loss": masked_seq_mean(per_token_loss, mask),
    }


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

    input_source = "input_ids"
    input_ids_cpu = tensors["input_ids"]
    if (
        isinstance(input_ids_cpu, torch.Tensor)
        and input_ids_cpu.is_nested
        and isinstance(tensors.get("prompts"), torch.Tensor)
        and isinstance(tensors.get("responses"), torch.Tensor)
        and not tensors["prompts"].is_nested
        and not tensors["responses"].is_nested
    ):
        # Mirrors CompletionOnlyLogprobMixin._build_padded_completion_inputs:
        # post-padding no-padding batches keep dense prompts/responses for the
        # MiniOneRec padded HF forward contract.
        input_ids_cpu = torch.cat([tensors["prompts"], tensors["responses"]], dim=1)
        input_source = "dense_prompts_plus_responses"
    input_ids = input_ids_cpu.to(device)
    attention_mask = tensors["attention_mask"].to(device)
    position_ids = tensors.get("position_ids")
    if args.use_position_ids:
        if position_ids is None:
            raise KeyError("position_ids missing from fixture")
        position_ids = position_ids.to(device)
    else:
        position_ids = None
    mask = tensors["response_mask"].to(device).float()
    ref = tensors["ref_log_prob"].to(device).float()
    adv = tensors["advantages"].to(device).float()
    rollout = tensors.get("rollout_log_probs")
    old = tensors.get("old_log_probs")
    if rollout is not None:
        rollout = rollout.to(device).float()
    if old is not None:
        old = old.to(device).float()
    else:
        old = rollout if rollout is not None else torch.zeros_like(ref)

    logits_to_keep = int(tensors["responses"].shape[1])
    chunks = []
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
            terms = minionerec_terms(
                current_logps=logps,
                ref_logps=ref[start:end],
                old_logps=old[start:end],
                advantages=adv[start:end],
                mask=mask[start:end],
                beta=args.beta,
            )
            loss = terms["seq_loss"].mean()
        (loss * ((end - start) / input_ids.shape[0])).backward()
        chunks.append(logps.detach().float().cpu())

    current = torch.cat(chunks, dim=0).to(device)
    terms = minionerec_terms(
        current_logps=current,
        ref_logps=ref,
        old_logps=old,
        advantages=adv,
        mask=mask,
        beta=args.beta,
    )

    grad_norm_sq = torch.tensor(0.0, device=device)
    selected_grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        grad_norm_sq += grad.pow(2).sum()
        if any(key in name for key in ("lm_head.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.27.mlp.down_proj.weight")):
            selected_grads[name] = stats(grad.cpu())

    report: dict[str, Any] = {
        "fixture": args.fixture,
        "fixture_step": payload.get("global_step"),
        "fixture_tag": payload.get("tag"),
        "model": args.model,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "disable_flash_sdp": bool(args.disable_flash_sdp),
        "use_position_ids": bool(args.use_position_ids),
        "input_source": input_source,
        "beta": float(args.beta),
        "n_rows": int(current.shape[0]),
        "response_len": int(current.shape[1]),
        "mask_tokens": int(mask.sum().item()),
        "current_logprob": stats(current, mask),
        "ref_logprob": stats(ref, mask),
        "old_logprob": stats(old, mask),
        "current_minus_ref": diff_stats(current, ref, mask),
        "current_minus_old": diff_stats(current, old, mask),
        "old_minus_ref": diff_stats(old, ref, mask),
        "low_var_kl_token": stats(terms["low_var_kl"], mask),
        "low_var_kl_seq": stats(terms["seq_kl"]),
        "pg_token": stats(terms["pg"], mask),
        "pg_seq": stats(terms["seq_pg"]),
        "loss_seq_mean": float(terms["seq_loss"].mean().item()),
        "loss_token_mask_mean": float((terms["per_token_loss"] * mask).sum().item() / mask.sum().clamp_min(1.0).item()),
        "grad_global_norm": float(torch.sqrt(grad_norm_sq).item()),
        "selected_gradients": selected_grads,
    }
    if rollout is not None:
        report["rollout_logprob"] = stats(rollout, mask)
        report["rollout_minus_ref"] = diff_stats(rollout, ref, mask)
        report["rollout_minus_old"] = diff_stats(rollout, old, mask)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
