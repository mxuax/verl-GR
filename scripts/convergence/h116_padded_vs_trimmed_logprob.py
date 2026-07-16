#!/usr/bin/env python3
"""Compare completion logprobs: content-trimmed vs left-padded-to-2560 (same tokens)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def per_token_logps(model, input_ids, attention_mask, logits_to_keep: int) -> torch.Tensor:
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
    ).logits
    logits = logits[:, :-1, :].float()
    labels = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-rows", type=int, default=8)
    parser.add_argument("--pad-to", type=int, default=2560)
    parser.add_argument("--pad-token-id", type=int, default=151645)
    args = parser.parse_args()

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    n = min(args.num_rows, fixture["prompt_ids"].shape[0])
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        fixture["model"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prompt = fixture["prompt_ids"][:n].to(device)
    pmask = fixture["prompt_mask"][:n].to(device)
    comp = fixture["completion_ids"][:n].to(device)
    cmask = fixture["completion_mask"][:n].to(device)
    keep = comp.shape[1]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logps_trim = per_token_logps(
            model, torch.cat([prompt, comp], dim=1), torch.cat([pmask, cmask], dim=1), keep
        )

    pad_len = args.pad_to - prompt.shape[1]
    if pad_len > 0:
        pad_ids = torch.full((n, pad_len), args.pad_token_id, device=device, dtype=prompt.dtype)
        pad_mask = torch.zeros((n, pad_len), device=device, dtype=pmask.dtype)
        prompt_pad = torch.cat([pad_ids, prompt], dim=1)
        pmask_pad = torch.cat([pad_mask, pmask], dim=1)
    else:
        prompt_pad, pmask_pad = prompt, pmask

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logps_pad = per_token_logps(
            model,
            torch.cat([prompt_pad, comp], dim=1),
            torch.cat([pmask_pad, cmask], dim=1),
            keep,
        )

    mask = cmask.bool()
    diff = (logps_trim - logps_pad).abs()
    ref = fixture["ref_per_token_logps"][:n].to(device)
    out = {
        "n": n,
        "trimmed_prompt_len": int(prompt.shape[1]),
        "padded_prompt_len": int(prompt_pad.shape[1]),
        "trim_vs_pad_max_abs": float(diff[mask].max()),
        "trim_vs_pad_mean_abs": float(diff[mask].float().mean()),
        "trim_vs_pad_p99_abs": float(torch.quantile(diff[mask].float(), 0.99)),
        "trim_vs_fixture_ref_max_abs": float((logps_trim - ref).abs()[mask].max()),
        "pad_vs_fixture_ref_max_abs": float((logps_pad - ref).abs()[mask].max()),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
