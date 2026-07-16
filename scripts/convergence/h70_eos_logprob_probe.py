#!/usr/bin/env python3
"""Probe completion-only logprob parity on EOS-containing MiniOneRec outputs.

H30 verified the loss formula on SID strings without generated EOS. H69 fails
through empty/EOS-only generations, so this probe checks the exact surface that
matters: normal SID+EOS and EOS-only completions when pad_token_id == eos_token_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390"


def _pad(seqs: list[list[int]], pad_id: int, *, left: bool) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    rows = []
    masks = []
    for seq in seqs:
        n_pad = max_len - len(seq)
        if left:
            rows.append([pad_id] * n_pad + seq)
            masks.append([0] * n_pad + [1] * len(seq))
        else:
            rows.append(seq + [pad_id] * n_pad)
            masks.append([1] * len(seq) + [0] * n_pad)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


def _selective_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return logits.float().log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _full_logits_to_keep_logps(model, input_ids, attention_mask, completion_ids):
    logits_to_keep = completion_ids.shape[1]
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]
    logits = logits[:, -logits_to_keep:, :]
    return _selective_logps(logits, completion_ids)


def _flatten_valid(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tensor[mask.bool()]


def _rmpad_selected_logps_from_full_logits(full_logits, input_ids, attention_mask, prompt_lens, response_lens):
    # Simulate completion_only rmpad selection: logit positions
    # [offset + prompt_len - 1, ..., offset + prompt_len + response_len - 2].
    valid_logits = _flatten_valid(full_logits, attention_mask)
    valid_ids = _flatten_valid(input_ids, attention_mask)
    rolled_labels = torch.roll(valid_ids, shifts=-1, dims=0)
    cu = torch.cat([torch.zeros(1, device=input_ids.device, dtype=torch.long), attention_mask.sum(dim=1).cumsum(0)])
    max_resp = int(response_lens.max().item())
    out = valid_logits.new_zeros((input_ids.shape[0], max_resp))
    for i, (p_len, r_len) in enumerate(zip(prompt_lens.tolist(), response_lens.tolist(), strict=True)):
        if r_len <= 0:
            continue
        start = int(cu[i].item()) + int(p_len) - 1
        idx = torch.arange(start, start + int(r_len), device=input_ids.device)
        logps = _selective_logps(valid_logits.index_select(0, idx), rolled_labels.index_select(0, idx))
        out[i, : int(r_len)] = logps
    return out


def _stats(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    diff = (a - b).detach().float()
    sel = mask.bool()
    if sel.any():
        diff = diff[sel]
    return {
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="logs/convergence/h70_eos_logprob_probe.json")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)
    eos_id = int(tok.eos_token_id)

    prompts = [
        "User history: industrial tape, measuring tool. Recommend the next item SID.\nResponse:\n",
        "User history: lab kit, microscope slide. Recommend the next item SID.\nResponse:\n",
        "User history: safety glove, storage case. Recommend the next item SID.\nResponse:\n",
    ]
    completion_texts = ["<a_1><b_2><c_3>", "<a_4><b_5><c_6>", ""]
    prompt_ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
    completion_ids = [tok.encode(c, add_special_tokens=False) + [eos_id] for c in completion_texts]
    p_ids, p_mask = _pad(prompt_ids, pad_id, left=True)
    c_ids, c_mask = _pad(completion_ids, pad_id, left=False)

    input_ids = torch.cat([p_ids, c_ids], dim=1).to(device)
    attention_mask = torch.cat([p_mask, c_mask], dim=1).to(device)
    completion_ids_t = c_ids.to(device)
    completion_mask = c_mask.to(device)
    prompt_lens = p_mask.sum(dim=1).to(device)
    response_lens = c_mask.sum(dim=1).to(device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model.eval()

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and dtype == torch.bfloat16))
    with torch.no_grad(), autocast:
        full_logps = _full_logits_to_keep_logps(model, input_ids, attention_mask, completion_ids_t)
        full_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        # Pad one dummy logit row per sequence end so flat indexing shape mirrors rolled labels.
        full_logits_for_flat = torch.cat(
            [full_logits, torch.zeros_like(full_logits[:, :1, :])],
            dim=1,
        )
        rmpad_logps = _rmpad_selected_logps_from_full_logits(
            full_logits_for_flat,
            input_ids,
            attention_mask,
            prompt_lens,
            response_lens,
        )
        # Ref fallback currently rebuilds attention_mask as input_ids != pad_id.
        bad_ref_mask = (input_ids != pad_id).to(attention_mask.dtype)
        bad_ref_logps = _full_logits_to_keep_logps(model, input_ids, bad_ref_mask, completion_ids_t)

    eos_positions = completion_ids_t.eq(eos_id) & completion_mask.bool()
    report = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
            "pad_equals_eos": pad_id == eos_id,
        },
        "completion_texts": completion_texts,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask.cpu().tolist(),
        "full_vs_rmpad": _stats(full_logps, rmpad_logps, completion_mask),
        "full_vs_bad_ref_mask": _stats(full_logps, bad_ref_logps, completion_mask),
        "eos_full_vs_rmpad": _stats(full_logps, rmpad_logps, eos_positions),
        "eos_full_vs_bad_ref_mask": _stats(full_logps, bad_ref_logps, eos_positions),
        "per_token": {
            "full": full_logps.detach().float().cpu().tolist(),
            "rmpad": rmpad_logps.detach().float().cpu().tolist(),
            "bad_ref_mask": bad_ref_logps.detach().float().cpu().tolist(),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
