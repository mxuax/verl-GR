#!/usr/bin/env python3
"""Analyze a dumped MiniOneRec real rollout/update batch."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


DEFAULT_MODEL = "/home/fq9hpsac/fq9hpsacuser04/workspace/MiniOneRec/output_dir/xxx/checkpoint-390"


def stats(t: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {"shape": list(t.shape), "dtype": str(t.dtype)}
    if t.numel() == 0:
        return out
    if t.is_floating_point():
        f = t.float()
        out.update(
            mean=float(f.mean().item()),
            std=float(f.std(unbiased=False).item()) if f.numel() > 1 else 0.0,
            min=float(f.min().item()),
            max=float(f.max().item()),
            sum=float(f.sum().item()),
        )
    else:
        out.update(min=int(t.min().item()), max=int(t.max().item()), sum=int(t.long().sum().item()))
    return out


def row_masked_max_diff(values: torch.Tensor, mask: torch.Tensor) -> float:
    diffs = []
    for row, row_mask in zip(values.float(), mask.bool(), strict=True):
        valid = row[row_mask]
        if valid.numel() <= 1:
            diffs.append(torch.zeros(()))
        else:
            diffs.append(valid.max() - valid.min())
    return float(torch.stack(diffs).abs().max().item()) if diffs else 0.0


def sequence_advantages_from_rewards(rewards: torch.Tensor, mask: torch.Tensor, uids: list[str], epsilon: float) -> torch.Tensor:
    scores = rewards.float().sum(dim=-1).clone()
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for uid, score in zip(uids, scores, strict=True):
        groups[uid].append(score)
    means = {}
    stds = {}
    for uid, items in groups.items():
        if len(items) == 1:
            means[uid] = torch.tensor(0.0)
            stds[uid] = torch.tensor(1.0)
        else:
            stacked = torch.stack(items)
            means[uid] = stacked.mean()
            stds[uid] = stacked.std()
    out = torch.empty_like(scores)
    for i, uid in enumerate(uids):
        out[i] = (scores[i] - means[uid]) / (stds[uid] + epsilon)
    return out.unsqueeze(-1) * mask.float()


def minionerec_loss_from_logps(logps: torch.Tensor, ref: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    kl = torch.exp(ref - logps) - (ref - logps) - 1
    pg = torch.exp(logps - logps.detach()) * adv
    return (-((pg - beta * kl) * mask).sum(dim=1) / mask.sum(dim=1)).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--decode-samples", type=int, default=12)
    args = parser.parse_args()

    pre = torch.load(args.pre, map_location="cpu", weights_only=False)
    post = torch.load(args.post, map_location="cpu", weights_only=False)
    pre_t = pre["tensor"]
    post_t = post["tensor"]
    non = pre.get("non_tensor", {})

    mask = pre_t["response_mask"].long()
    rewards = pre_t["token_level_rewards"].float()
    adv = pre_t["advantages"].float()
    ref = pre_t["ref_log_prob"].float()
    uids = [str(x) for x in non.get("uid", list(range(mask.shape[0])))]
    uid_counts = Counter(uids)

    recomputed_adv = sequence_advantages_from_rewards(rewards, mask, uids, args.epsilon)
    adv_diff = (adv - recomputed_adv).abs()
    response_mask_vs_loss_mask = None
    if "loss_mask" in post_t:
        response_mask_vs_loss_mask = stats((post_t["loss_mask"].long() - mask).abs())

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    decoded = []
    for i in range(min(args.decode_samples, int(mask.shape[0]))):
        ids = pre_t["responses"][i][mask[i].bool()].tolist()
        decoded.append(
            {
                "idx": i,
                "uid": uids[i] if i < len(uids) else str(i),
                "response_ids": ids,
                "response_text": tok.decode(ids, skip_special_tokens=True),
                "reward_sum": float(rewards[i].sum().item()),
                "adv_first": float(adv[i][mask[i].bool()][0].item()) if mask[i].sum() else None,
            }
        )

    report = {
        "paths": {"pre": args.pre, "post": args.post},
        "tensor_stats": {key: stats(value) for key, value in pre_t.items()},
        "post_tensor_stats": {key: stats(value) for key, value in post_t.items() if key in {"loss_mask", "advantages", "ref_log_prob", "responses"}},
        "uid": {
            "num_rows": len(uids),
            "num_groups": len(uid_counts),
            "group_size_counts": dict(Counter(uid_counts.values())),
            "first_groups": list(uid_counts.items())[:8],
        },
        "mask": {
            "row_sum_counts": dict(Counter(mask.sum(dim=1).tolist())),
            "response_mask_vs_post_loss_mask_abs": response_mask_vs_loss_mask,
        },
        "reward": {
            "nonzero_token_position_counts": dict(Counter(torch.nonzero(rewards != 0, as_tuple=False)[:, 1].tolist())),
            "sequence_reward_stats": stats(rewards.sum(dim=-1)),
        },
        "advantage": {
            "masked_row_max_diff": row_masked_max_diff(adv, mask),
            "recomputed_from_rewards_max_abs": float(adv_diff.max().item()),
            "recomputed_from_rewards_mean_abs": float(adv_diff.mean().item()),
            "sequence_advantage_stats": stats(adv[:, 0]),
        },
        "loss_identity_given_same_logps": {
            "verl_token_adv_loss": float(minionerec_loss_from_logps(ref, ref, adv, mask, args.beta).item()),
            "recomputed_seq_adv_loss": float(minionerec_loss_from_logps(ref, ref, recomputed_adv, mask, args.beta).item()),
        },
        "decoded_samples": decoded,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
