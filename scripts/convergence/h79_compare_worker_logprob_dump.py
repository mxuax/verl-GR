#!/usr/bin/env python3
"""Compare worker-side completion-only logprob dump with standalone replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


DEFAULT_MODEL = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390"


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    d = a.detach().float() - b.detach().float()
    return {
        "shape": list(a.shape),
        "a_dtype": str(a.dtype),
        "b_dtype": str(b.dtype),
        "mean": float(d.mean().item()),
        "mean_abs": float(d.abs().mean().item()),
        "max_abs": float(d.abs().max().item()),
        "min": float(d.min().item()),
        "max": float(d.max().item()),
        "std": float(d.std(unbiased=False).item()) if d.numel() > 1 else 0.0,
    }


def param_summaries(model) -> dict[str, dict[str, Any]]:
    substrings = (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.27.mlp.down_proj.weight",
        "lm_head.weight",
    )
    out = {}
    for name, param in model.named_parameters():
        if any(s in name for s in substrings):
            data = param.detach().float()
            flat = data.reshape(-1)
            out[name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "norm": float(data.norm().item()),
                "mean": float(data.mean().item()),
                "sample": flat[: min(16, flat.numel())].cpu().tolist(),
            }
    return out


def compare_param_summaries(worker: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name in sorted(set(worker) | set(replay)):
        w = worker.get(name)
        r = replay.get(name)
        if w is None or r is None:
            out[name] = {"missing": "worker" if w is None else "replay"}
            continue
        sample_w = torch.tensor(w.get("sample", []), dtype=torch.float32)
        sample_r = torch.tensor(r.get("sample", []), dtype=torch.float32)
        out[name] = {
            "worker_dtype": w.get("dtype"),
            "replay_dtype": r.get("dtype"),
            "norm_abs_diff": abs(float(w.get("norm", 0.0)) - float(r.get("norm", 0.0))),
            "mean_abs_diff": abs(float(w.get("mean", 0.0)) - float(r.get("mean", 0.0))),
            "sample_max_abs_diff": float((sample_w - sample_r).abs().max().item()) if sample_w.numel() and sample_w.shape == sample_r.shape else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--disable-flash-sdp", action="store_true")
    parser.add_argument("--train-mode", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.disable_flash_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    payload = torch.load(args.worker_dump, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(args.train_mode)

    input_ids = payload["input_ids_padded"].to(device)
    attention_mask = payload["attention_mask"].to(device)
    logits_to_keep = int(payload["logits_to_keep"])
    labels = payload["labels"].to(device)
    autocast_enabled = args.dtype == "bf16" and device.type == "cuda" and not args.no_autocast
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep).logits
        if logits.shape[1] > logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        completion_logits = logits[:, :-1, :][:, -(logits_to_keep - 1):, :].float()
        replay_label_logits = completion_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        replay_logsumexp = torch.logsumexp(completion_logits, dim=-1)
        replay_logps = replay_label_logits - replay_logsumexp
        topk_values, topk_indices = torch.topk(completion_logits, k=payload["topk_values"].shape[-1], dim=-1)

    worker_params = payload.get("param_summaries", {})
    replay_params = param_summaries(model)
    report = {
        "worker_dump": args.worker_dump,
        "model": args.model,
        "dtype": args.dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "attn_implementation": args.attn_implementation,
        "disable_flash_sdp": bool(args.disable_flash_sdp),
        "train_mode": bool(args.train_mode),
        "autocast_enabled": bool(autocast_enabled),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "label_logits_diff": diff_stats(replay_label_logits.cpu(), payload["label_logits"]),
        "logsumexp_diff": diff_stats(replay_logsumexp.cpu(), payload["logsumexp"]),
        "logprob_diff": diff_stats(replay_logps.cpu(), payload["completion_logps"]),
        "topk_values_diff": diff_stats(topk_values.cpu(), payload["topk_values"]),
        "topk_indices_equal_fraction": float((topk_indices.cpu() == payload["topk_indices"]).float().mean().item()),
        "param_summaries": compare_param_summaries(worker_params, replay_params),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
