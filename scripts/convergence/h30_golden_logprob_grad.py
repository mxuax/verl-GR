#!/usr/bin/env python3
"""H30: same-weight MiniOneRec loss/logprob/gradient parity probe.

This script deliberately avoids a full Trainer initialization.  It imports the
original MiniOneRec ``ReReTrainer`` class, creates a minimal instance with the
fields used by ``compute_loss``, and calls the original method directly.  The
verl-style side replays the same tensors with the equivalent seq-mean-token-mean
formula.  The output is JSON with concrete loss, logprob, KL, and gradient
differences.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390"
DEFAULT_MINIONEREC_ROOT = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec"


class _DummyAccelerator:
    def gather_for_metrics(self, value):
        return value


@dataclass
class DiffStats:
    max_abs: float
    mean_abs: float
    rel_l2: float
    cosine: float | None = None


def _tensor_stats(a: torch.Tensor, b: torch.Tensor) -> DiffStats:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = af - bf
    max_abs = diff.abs().max().item() if diff.numel() else 0.0
    mean_abs = diff.abs().mean().item() if diff.numel() else 0.0
    denom = bf.norm().item()
    rel_l2 = diff.norm().item() / max(denom, 1e-12)
    cosine = None
    if af.numel() and af.norm().item() > 0 and bf.norm().item() > 0:
        cosine = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    return DiffStats(max_abs=max_abs, mean_abs=mean_abs, rel_l2=rel_l2, cosine=cosine)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, DiffStats):
        return obj.__dict__
    if isinstance(obj, torch.Tensor):
        return obj.detach().float().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
    return obj


def _pad_1d(seqs: list[list[int]], pad_id: int, *, left: bool) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in seqs)
    ids = []
    mask = []
    for seq in seqs:
        pad_len = max_len - len(seq)
        if left:
            row = [pad_id] * pad_len + seq
            row_mask = [0] * pad_len + [1] * len(seq)
        else:
            row = seq + [pad_id] * pad_len
            row_mask = [1] * len(seq) + [0] * pad_len
        ids.append(row)
        mask.append(row_mask)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def build_batch(tokenizer, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    prompts = [
        "User history: item A, item B, item C. Recommend the next item SID.\nResponse:\n",
        "User history: science tool, lab kit, storage case. Recommend the next item SID.\nResponse:\n",
        "User history: industrial tape, measuring tool, safety glove. Recommend the next item SID.\nResponse:\n",
        "User history: microscope slide, pipette tip, nitrile glove. Recommend the next item SID.\nResponse:\n",
    ]
    completions = [
        "<a_1><b_2><c_3>",
        "<a_4><b_5><c_6>",
        "<a_7><b_8><c_9>",
        "<a_10><b_11><c_12>",
    ]
    prompt_ids = [tokenizer.encode(prompts[i % len(prompts)], add_special_tokens=False) for i in range(batch_size)]
    completion_ids = [
        tokenizer.encode(completions[i % len(completions)], add_special_tokens=False) for i in range(batch_size)
    ]
    if any(len(seq) == 0 for seq in completion_ids):
        raise RuntimeError("At least one synthetic completion tokenized to an empty sequence.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    p_ids, p_mask = _pad_1d(prompt_ids, pad_id, left=True)
    c_ids, c_mask = _pad_1d(completion_ids, pad_id, left=False)
    return {
        "prompt_ids": p_ids.to(device),
        "prompt_mask": p_mask.to(device),
        "completion_ids": c_ids.to(device),
        "completion_mask": c_mask.to(device),
    }


def local_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]
    labels = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return torch.log_softmax(logits.float(), dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def make_probe_trainer(re_trainer_cls, beta: float, dapo: bool, gspo: bool):
    probe = object.__new__(re_trainer_cls)
    probe.beta = beta
    probe.dapo = dapo
    probe.gspo = gspo
    probe.accelerator = _DummyAccelerator()
    probe._metrics = {"completion_length": [], "kl": []}
    return probe


def compute_verl_style_loss(logps, ref_logps, advantages, completion_mask, beta: float, dapo: bool = False):
    per_token_kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    per_token_pg = torch.exp(logps - logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_pg - beta * per_token_kl)
    if dapo:
        return (per_token_loss * completion_mask).sum() / completion_mask.sum(), per_token_kl
    return ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean(), per_token_kl


def selected_gradients(model, substrings: list[str]) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if any(s in name for s in substrings):
            grads[name] = param.grad.detach().float().cpu().clone()
    return grads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--minionerec-root", default=os.environ.get("MINIONEREC_ROOT", DEFAULT_MINIONEREC_ROOT))
    parser.add_argument("--out", default="logs/convergence/h30_golden_logprob_grad.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--dapo", action="store_true")
    parser.add_argument(
        "--grad-substr",
        action="append",
        default=["lm_head.weight", "embed_tokens.weight", "layers.0.self_attn.q_proj.weight"],
        help="Parameter-name substring to include in gradient diff. Can be repeated.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.minionerec_root).resolve()))
    from minionerec_trainer import ReReTrainer  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model.train()

    probe = make_probe_trainer(ReReTrainer, beta=args.beta, dapo=args.dapo, gspo=False)
    batch = build_batch(tokenizer, args.batch_size, device)
    input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
    attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
    logits_to_keep = batch["completion_ids"].shape[1]
    advantages = torch.linspace(-1.0, 1.0, steps=args.batch_size, device=device)
    if args.batch_size == 1:
        advantages = torch.ones(1, device=device)

    autocast_enabled = args.dtype == "bf16" and device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled)
    with torch.no_grad(), autocast_ctx:
        orig_logps_for_ref = probe._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        local_logps = local_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        pattern = torch.linspace(-0.02, 0.02, steps=orig_logps_for_ref.numel(), device=device).reshape_as(orig_logps_for_ref)
        ref_logps = (orig_logps_for_ref.detach() + pattern).detach()
        initial_kl = torch.exp(ref_logps - orig_logps_for_ref) - (ref_logps - orig_logps_for_ref) - 1

    inputs = {
        **batch,
        "ref_per_token_logps": ref_logps,
        "advantages": advantages,
    }

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        orig_loss = probe.compute_loss(model, inputs)
    orig_loss.backward()
    orig_grads = selected_gradients(model, args.grad_substr)

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        replay_logps = probe._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        verl_loss, verl_kl = compute_verl_style_loss(
            replay_logps,
            ref_logps,
            advantages,
            batch["completion_mask"],
            beta=args.beta,
            dapo=args.dapo,
        )
    verl_loss.backward()
    verl_grads = selected_gradients(model, args.grad_substr)

    grad_report = {}
    for name in sorted(set(orig_grads) | set(verl_grads)):
        if name not in orig_grads or name not in verl_grads:
            grad_report[name] = {"missing": "orig" if name not in orig_grads else "verl"}
        else:
            grad_report[name] = _tensor_stats(orig_grads[name], verl_grads[name])

    report = {
        "config": vars(args),
        "device": str(device),
        "model_dtype": str(model_dtype),
        "batch": {
            "batch_size": args.batch_size,
            "prompt_shape": list(batch["prompt_ids"].shape),
            "completion_shape": list(batch["completion_ids"].shape),
            "completion_mask_sum": batch["completion_mask"].sum(dim=1).detach().cpu().tolist(),
            "advantages": advantages.detach().float().cpu().tolist(),
        },
        "loss": {
            "orig": float(orig_loss.detach().float().cpu()),
            "verl_style": float(verl_loss.detach().float().cpu()),
            "abs_diff": abs(float(orig_loss.detach().float().cpu()) - float(verl_loss.detach().float().cpu())),
        },
        "logprob": {
            "orig_method_vs_local": _tensor_stats(orig_logps_for_ref, local_logps),
            "orig_method_vs_replay": _tensor_stats(orig_logps_for_ref, replay_logps),
        },
        "kl": {
            "initial_mean": float(((initial_kl * batch["completion_mask"]).sum(dim=1) / batch["completion_mask"].sum(dim=1)).mean().cpu()),
            "verl_replay_mean": float(((verl_kl * batch["completion_mask"]).sum(dim=1) / batch["completion_mask"].sum(dim=1)).mean().detach().float().cpu()),
            "diff": _tensor_stats(initial_kl, verl_kl),
        },
        "gradients": grad_report,
        "orig_metrics": probe._metrics,
        "pass": {
            "loss_abs_diff_lt_1e-6": abs(float(orig_loss.detach().float().cpu()) - float(verl_loss.detach().float().cpu())) < 1e-6,
            "logprob_local_max_abs_lt_1e-5": _tensor_stats(orig_logps_for_ref, local_logps).max_abs < 1e-5,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonify(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonify(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
