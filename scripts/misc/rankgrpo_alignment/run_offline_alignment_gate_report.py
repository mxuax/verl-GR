#!/usr/bin/env python3
"""Offline per-step alignment gate report from fork TensorBoard vs TRL_REF."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import (
    get_rankgrpo_alignment_accumulator,
    write_rankgrpo_alignment_report,
)


def _load_fork_accumulator(fork_tb: Path, max_step: int | None) -> None:
    acc = get_rankgrpo_alignment_accumulator()
    acc.steps.clear()
    acc.metrics_by_step.clear()

    ea = EventAccumulator(str(fork_tb), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        raise SystemExit(f"No TB scalars in {fork_tb}")

    steps = sorted({e.step for tag in tags for e in ea.Scalars(tag)})
    if max_step is not None:
        steps = [s for s in steps if s <= max_step]
    if not steps:
        raise SystemExit(f"No steps <= {max_step} in {fork_tb}")

    scalar_cache: dict[str, dict[int, float]] = {}
    for tag in tags:
        scalar_cache[tag] = {int(e.step): float(e.value) for e in ea.Scalars(tag)}

    for step in steps:
        metrics = {tag: vals[step] for tag, vals in scalar_cache.items() if step in vals}
        acc.record(step, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fork-tb",
        default=str(_REPO / "tensorboard_log/RankGRPO/logprob_align_v015_TP2_g0_1_s400"),
    )
    parser.add_argument(
        "--trl-ref",
        default=os.environ.get(
            "TRL_REF",
            "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/"
            "logs/debug_precision_verlgr/runs/Jul06_12-52-16_hk01dgx028",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO / "outputs/logprob_align_v015_TP2_g0_1"),
    )
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--experiment", default="offline_gate_report")
    args = parser.parse_args()

    os.environ.setdefault("RUN_DEBUG_STEP", str(args.max_step or 200))
    os.environ["TRL_REF"] = args.trl_ref
    os.environ["VERL_GR_TRL_TB_REF"] = args.trl_ref
    os.environ["VERL_GR_ALIGN_REPORT_DIR"] = args.output_dir
    os.environ["VERL_GR_ALIGN_GATE_EXIT"] = "0"

    _load_fork_accumulator(Path(args.fork_tb), args.max_step)
    result = write_rankgrpo_alignment_report(
        output_dir=args.output_dir,
        trl_tb_dir=args.trl_ref,
        experiment_name=args.experiment,
    )
    if result is None:
        raise SystemExit(1)
    report_path, gate = result
    print(report_path)
    if gate.blocked_reasons:
        print("GATE BLOCKED:", "; ".join(gate.blocked_reasons))
        raise SystemExit(2)
    if not gate.passed:
        n_fail = sum(1 for r in gate.steps if not r.passed)
        print(f"GATE FAIL: {n_fail}/{len(gate.steps)} steps")
        raise SystemExit(2)
    print(f"GATE PASS: {len(gate.steps)} steps")


if __name__ == "__main__":
    main()
