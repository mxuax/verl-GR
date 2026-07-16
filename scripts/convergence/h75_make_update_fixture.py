#!/usr/bin/env python3
"""Create a fixed MiniOneRec update fixture shared by DeepSpeed and DDP probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390"


def _pad(seqs: list[list[int]], pad_id: int, *, left: bool) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in seqs)
    rows, masks = [], []
    for seq in seqs:
        pad_len = max_len - len(seq)
        if left:
            rows.append([pad_id] * pad_len + seq)
            masks.append([0] * pad_len + [1] * len(seq))
        else:
            rows.append(seq + [pad_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


def _per_token_logps(model, input_ids, attention_mask, logits_to_keep: int) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]
    labels = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="logs/convergence/h75_update_fixture.pt")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    prompts = [
        "### User Input: \nThe user has interacted with items <a_210><b_116><c_2>, <a_202><b_25><c_167> in chronological order. Can you predict the next possible item that the user may expect?\n\n### Response:\n",
        "### User Input: \nThe user has interacted with items <a_251><b_29><c_100>, <a_71><b_69><c_199> in chronological order. Can you predict the next possible item that the user may expect?\n\n### Response:\n",
        "### User Input: \nThe user has interacted with items <a_104><b_167><c_96>, <a_210><b_12><c_10> in chronological order. Can you predict the next possible item that the user may expect?\n\n### Response:\n",
        "### User Input: \nThe user has interacted with items <a_165><b_59><c_5>, <a_228><b_18><c_189> in chronological order. Can you predict the next possible item that the user may expect?\n\n### Response:\n",
    ]
    completions = [
        "<a_210><b_156><c_39>",
        "<a_165><b_214><c_30>",
        "<a_206><b_171><c_23>",
        "",
    ]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)
    eos_id = int(tok.eos_token_id)

    prompt_ids = [tok.encode(prompts[i % len(prompts)], add_special_tokens=False) for i in range(args.batch_size)]
    completion_ids = [
        tok.encode(completions[i % len(completions)], add_special_tokens=False) + [eos_id]
        for i in range(args.batch_size)
    ]
    prompt_ids_t, prompt_mask = _pad(prompt_ids, pad_id, left=True)
    completion_ids_t, completion_mask = _pad(completion_ids, pad_id, left=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model.eval()

    input_ids = torch.cat([prompt_ids_t, completion_ids_t], dim=1).to(device)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(device)
    logits_to_keep = completion_ids_t.shape[1]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and args.dtype == "bf16"):
        logps = _per_token_logps(model, input_ids, attention_mask, logits_to_keep)
    pattern = torch.linspace(-0.02, 0.02, steps=logps.numel(), device=logps.device).reshape_as(logps)
    ref_logps = (logps.detach() + pattern).float().cpu()

    if args.batch_size == 1:
        advantages = torch.ones(1, dtype=torch.float32)
    else:
        advantages = torch.linspace(-1.0, 1.0, steps=args.batch_size, dtype=torch.float32)

    fixture = {
        "model": args.model,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
        "prompt_ids": prompt_ids_t.cpu(),
        "prompt_mask": prompt_mask.cpu(),
        "completion_ids": completion_ids_t.cpu(),
        "completion_mask": completion_mask.cpu(),
        "ref_per_token_logps": ref_logps,
        "advantages": advantages,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, out)
    print(f"wrote {out}")
    print({
        "prompt_shape": list(prompt_ids_t.shape),
        "completion_shape": list(completion_ids_t.shape),
        "completion_mask_sum": completion_mask.sum(dim=1).tolist(),
        "advantages": advantages.tolist(),
    })


if __name__ == "__main__":
    main()
