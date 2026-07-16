#!/usr/bin/env python3
"""Compare H75 DeepSpeed ZeRO-2 and verl-GR DDP update-delta probe outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for file in sorted(path.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def by_step_backend(events: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    out = {}
    for event in events:
        backend = str(event.get("backend", "unknown"))
        step = int(event.get("step", -1))
        rank = int(event.get("rank", 0))
        out[(backend, step, rank)] = event
    return out


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def collect_visible(event: dict[str, Any]) -> dict[str, dict[str, float]]:
    return event.get("visible_deltas", {}) or {}


def collect_master(event: dict[str, Any]) -> dict[str, Any]:
    named = event.get("fp32_master_deltas")
    if isinstance(named, dict):
        return named
    sampled = event.get("fp32_master_sample_deltas")
    if isinstance(sampled, list):
        return {f"group_{item.get('group_idx')}": item for item in sampled}
    return {}


def metric_block(ds: dict[str, Any], ddp: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "loss": {
            "deepspeed": ds.get("loss"),
            "verl_ddp": ddp.get("loss"),
            "abs_diff": abs(float(ds.get("loss", 0.0)) - float(ddp.get("loss", 0.0))),
        },
        "grad_norm": {
            "deepspeed_pre_clip_local_view": ds.get("pre_clip_global_grad_norm_local_view"),
            "deepspeed_internal": ds.get("deepspeed_grad_norm"),
            "verl_ddp_pre_clip": ddp.get("pre_clip_global_grad_norm"),
            "verl_ddp_post_clip": ddp.get("post_clip_global_grad_norm"),
            "pre_clip_ratio_ddp_over_ds": ratio(
                ddp.get("pre_clip_global_grad_norm"),
                ds.get("pre_clip_global_grad_norm_local_view"),
            ),
        },
        "visible_deltas": {},
        "fp32_master_deltas": {},
    }
    ds_visible = collect_visible(ds)
    ddp_visible = collect_visible(ddp)
    for name in sorted(set(ds_visible) | set(ddp_visible)):
        dsv = ds_visible.get(name, {})
        ddpv = ddp_visible.get(name, {})
        out["visible_deltas"][name] = {
            "deepspeed_norm": dsv.get("norm"),
            "verl_ddp_norm": ddpv.get("norm"),
            "ratio_ddp_over_ds": ratio(ddpv.get("norm"), dsv.get("norm")),
            "deepspeed_max_abs": dsv.get("max_abs"),
            "verl_ddp_max_abs": ddpv.get("max_abs"),
        }
    ds_master = collect_master(ds)
    ddp_master = collect_master(ddp)
    for name in sorted(set(ds_master) | set(ddp_master)):
        dsm = ds_master.get(name, {})
        ddpm = ddp_master.get(name, {})
        out["fp32_master_deltas"][name] = {
            "deepspeed_norm": dsm.get("norm"),
            "verl_ddp_norm": ddpm.get("norm"),
            "ratio_ddp_over_ds": ratio(ddpm.get("norm"), dsm.get("norm")),
            "deepspeed_max_abs": dsm.get("max_abs"),
            "verl_ddp_max_abs": ddpm.get("max_abs"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    events = load_events(Path(args.dump_dir))
    index = by_step_backend(events)
    ds = index.get(("deepspeed_zero2", args.step, args.rank))
    ddp = index.get(("verl_ddp_fp32_master", args.step, args.rank))
    if ds is None or ddp is None:
        available = sorted(index)
        raise SystemExit(f"missing comparison events for step={args.step}, rank={args.rank}; available={available}")
    report = {
        "step": args.step,
        "rank": args.rank,
        "deepspeed_file_backend": ds.get("backend"),
        "verl_file_backend": ddp.get("backend"),
        "comparison": metric_block(ds, ddp),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
