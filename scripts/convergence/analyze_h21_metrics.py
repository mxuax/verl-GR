#!/usr/bin/env python3
"""Summarize VERL_GR_METRICS_JSONL for H21 alignment monitoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def pick(metrics: dict, *keys: str):
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def summarize(rows: list[dict], tail: int | None = None) -> str:
    if not rows:
        return "metrics: (empty)"
    if tail is not None and tail > 0:
        rows = rows[-tail:]

    lines: list[str] = []
    for row in rows:
        step = row.get("step")
        m = row.get("metrics") or {}
        grad = pick(m, "actor/grad_norm", "grad_norm")
        kl = pick(m, "actor/kl_loss", "kl_loss")
        loss = pick(m, "actor/policy_loss", "policy_loss", "loss")
        reward = pick(
            m,
            "minionerec/total_reward_mean",
            "minionerec_reward_mean",
            "reward/mean",
        )
        invalid = pick(m, "minionerec_invalid_sid", "minionerec/invalid_sid_mean")
        valid = pick(m, "minionerec_valid_sid", "minionerec/valid_sid_mean")
        lr = pick(m, "actor/lr", "lr")
        parts = [f"step={step}"]
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        if reward is not None:
            parts.append(f"reward={float(reward):.4f}")
        if invalid is not None:
            parts.append(f"invalid_sid={float(invalid):.4f}")
        if valid is not None:
            parts.append(f"valid_sid={float(valid):.4f}")
        if grad is not None:
            parts.append(f"grad_norm={float(grad):.4f}")
        if kl is not None:
            parts.append(f"kl={float(kl):.6f}")
        if loss is not None:
            parts.append(f"loss={float(loss):.6f}")
        lines.append("metrics: " + " ".join(parts))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=5)
    args = parser.parse_args()
    rows = load_metrics(args.metrics)
    print(summarize(rows, tail=args.tail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
