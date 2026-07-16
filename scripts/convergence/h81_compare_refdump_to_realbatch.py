#!/usr/bin/env python3
"""Match a ref worker logprob dump back to the dumped realbatch rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    d = a.float() - b.float()
    return {
        "shape": list(a.shape),
        "mean": float(d.mean().item()),
        "mean_abs": float(d.abs().mean().item()),
        "max_abs": float(d.abs().max().item()),
        "min": float(d.min().item()),
        "max": float(d.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--realbatch", required=True)
    parser.add_argument("--refdump", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    real = torch.load(args.realbatch, map_location="cpu", weights_only=False)
    dump = torch.load(args.refdump, map_location="cpu", weights_only=False)
    tensors = real["tensor"]
    full_input = tensors["input_ids"]
    full_ref = tensors["ref_log_prob"]
    dump_input = dump["input_ids_padded"]
    dump_logps = dump["completion_logps"]

    matches = []
    for i, row in enumerate(dump_input):
        equal = (full_input == row.unsqueeze(0)).all(dim=1).nonzero(as_tuple=False).flatten()
        matches.append(equal.tolist())

    selected = []
    for equal in matches:
        if len(equal) != 1:
            continue
        selected.append(equal[0])

    report: dict[str, Any] = {
        "realbatch": args.realbatch,
        "refdump": args.refdump,
        "dump_rows": int(dump_input.shape[0]),
        "matches": matches,
        "unique_match_count": len(selected),
    }
    if len(selected) == dump_input.shape[0]:
        matched_ref = full_ref[selected]
        report["dump_logps_vs_realbatch_ref"] = stats(dump_logps, matched_ref)
        report["matched_realbatch_indices"] = selected

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
