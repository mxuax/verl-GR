#!/usr/bin/env python3
"""Build TRL logprob gate sidecar from Rank-GRPO train log console metrics."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def _parse_train_log(log_path: Path) -> dict[str, dict[int, float]]:
    """Parse HF Trainer log dict lines containing logprob_gate metrics."""
    pattern = re.compile(r"\|\s*(\d+)/\d+\s+\[")
    payload: dict[str, dict[int, float]] = {
        "train/logprob_gate/rollout_minus_ref/abs_mean": {},
        "train/actor/debug/logprob_diff_abs": {},
    }
    step = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            step = int(m.group(1))
        if "logprob_gate/rollout_minus_ref/abs_mean" not in line:
            continue
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            row = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        rr = row.get("logprob_gate/rollout_minus_ref/abs_mean")
        ar = row.get("actor/debug/logprob_diff_abs")
        if rr is not None:
            payload["train/logprob_gate/rollout_minus_ref/abs_mean"][step] = float(rr)
        if ar is not None:
            payload["train/actor/debug/logprob_diff_abs"][step] = float(ar)
    return {
        tag: {str(k): v for k, v in sorted(vals.items())}
        for tag, vals in payload.items()
        if vals
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", required=True, help="Rank-GRPO train_*.log path")
    parser.add_argument("--output", required=True, help="Output rankgrpo_gate_sidecar.json")
    args = parser.parse_args()

    payload = _parse_train_log(Path(args.train_log))
    if not payload.get("train/logprob_gate/rollout_minus_ref/abs_mean"):
        raise SystemExit(f"No logprob metrics found in {args.train_log}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    n = len(payload["train/logprob_gate/rollout_minus_ref/abs_mean"])
    print(f"wrote {out} ({n} steps)")


if __name__ == "__main__":
    main()
