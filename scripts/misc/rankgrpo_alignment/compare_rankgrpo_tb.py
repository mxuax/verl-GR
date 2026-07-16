#!/usr/bin/env python3
"""Compare Rank-GRPO TensorBoard scalars between verl-gr and TRL reference runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir: str | Path) -> dict[str, list[tuple[int, float]]]:
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def _pick_series(scalars: dict[str, list[tuple[int, float]]], candidates: list[str]) -> str | None:
    for tag in candidates:
        if tag in scalars and scalars[tag]:
            return tag
    return None


def compare_pair(
    fork: dict[str, list[tuple[int, float]]],
    orig: dict[str, list[tuple[int, float]]],
    *,
    fork_tag: str,
    orig_tag: str,
    limit: int,
) -> None:
    fs = fork.get(fork_tag, [])
    os_ = orig.get(orig_tag, [])
    print(f"\n=== {fork_tag}  vs  {orig_tag} ===")
    print(f"fork n={len(fs)}  orig n={len(os_)}")
    if not fs or not os_:
        print("  (missing series)")
        return
    n = min(limit, len(fs), len(os_))
    for i in range(n):
        print(f"  [{i}] fork step {fs[i][0]:>6} = {fs[i][1]:.6g} | orig step {os_[i][0]:>6} = {os_[i][1]:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork", required=True, help="verl-gr tensorboard logdir")
    parser.add_argument("--orig", required=True, help="TRL reference tensorboard logdir")
    parser.add_argument("--limit", type=int, default=15, help="points to print per metric")
    args = parser.parse_args()

    fork = load_scalars(args.fork)
    orig = load_scalars(args.orig)

    pairs = [
        ("actor/kl_loss", "train/kl"),
        ("logprob_gate/rollout_minus_ref/abs_mean", "train/logprob_gate/rollout_minus_ref/abs_mean"),
        ("logprob_gate/rollout_minus_rollout/abs_mean", "train/logprob_gate/rollout_minus_rollout/abs_mean"),
        ("actor/debug/logprob_diff_abs", "train/actor/debug/logprob_diff_abs"),
        ("dbg/logp_actor_mean", "train/dbg/logp_actor_mean"),
        ("dbg/logp_ref_mean", "train/dbg/logp_ref_mean"),
        ("train/rankgrpo/completions/mean_length", "train/completions/mean_length"),
        ("train/rankgrpo/completions/clipped_ratio", "train/completions/clipped_ratio"),
        ("train/rankgrpo/reward_total", "train/reward_total"),
        ("train/rankgrpo/reward", "train/reward"),
    ]

    print(f"fork tags: {len(fork)}  orig tags: {len(orig)}")
    for ft, ot in pairs:
        if ot is None:
            if ft in fork:
                print(f"\n=== fork-only {ft} (first {args.limit}) ===")
                for step, val in fork[ft][: args.limit]:
                    print(f"  step {step}: {val:.6g}")
            continue
        fork_tag = _pick_series(fork, [ft]) or ft
        orig_tag = _pick_series(orig, [ot]) or ot
        compare_pair(fork, orig, fork_tag=fork_tag, orig_tag=orig_tag, limit=args.limit)


if __name__ == "__main__":
    main()
