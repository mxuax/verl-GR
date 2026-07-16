#!/usr/bin/env python3
"""Convert a MiniOneRec realbatch dump into an H75-style update fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_MODEL = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--realbatch", required=True, help="path to step*_post_padding.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-rows", type=int, default=32)
    parser.add_argument("--row-offset", type=int, default=0)
    args = parser.parse_args()

    raw = torch.load(args.realbatch, map_location="cpu", weights_only=False)
    tensors = raw["tensor"]
    prompts = tensors["prompts"]
    responses = tensors["responses"]
    response_mask = tensors["response_mask"].long()
    attention_mask = tensors["attention_mask"].long()
    ref = tensors["ref_log_prob"].float()
    advantages = tensors["advantages"].float()

    n = min(args.num_rows, prompts.shape[0] - args.row_offset)
    sl = slice(args.row_offset, args.row_offset + n)
    prompt_ids = prompts[sl].contiguous()
    completion_ids = responses[sl].contiguous()
    completion_mask = response_mask[sl].contiguous()
    prompt_len = prompt_ids.shape[1]
    prompt_mask = attention_mask[sl, :prompt_len].contiguous()
    # Realbatch dumps are left-padded to max model length (e.g. 2560). Keep only
    # the trailing content window so update probes do not OOM on pad tokens.
    content_lens = prompt_mask.sum(dim=1)
    keep_prompt = int(content_lens.max().item())
    if keep_prompt < prompt_len:
        prompt_ids = prompt_ids[:, -keep_prompt:].contiguous()
        prompt_mask = prompt_mask[:, -keep_prompt:].contiguous()
    # Per-sequence advantage (token-constant in MiniOneRec GRPO dump).
    adv_seq = advantages[sl, 0].contiguous()
    ref_logps = ref[sl].contiguous()

    fixture = {
        "model": args.model,
        "source_realbatch": str(Path(args.realbatch).resolve()),
        "row_offset": args.row_offset,
        "num_rows": n,
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "ref_per_token_logps": ref_logps,
        "advantages": adv_seq,
        "trimmed_prompt_len": keep_prompt,
        "raw_prompt_len": prompt_len,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, out)
    print(
        {
            "out": str(out),
            "num_rows": n,
            "prompt_shape": list(prompt_ids.shape),
            "raw_prompt_len": prompt_len,
            "trimmed_prompt_len": keep_prompt,
            "completion_shape": list(completion_ids.shape),
            "completion_mask_sum": completion_mask.sum(dim=1).tolist()[:8],
            "adv_mean": float(adv_seq.mean()),
            "adv_std": float(adv_seq.std(unbiased=False)),
        }
    )


if __name__ == "__main__":
    main()
