#!/usr/bin/env python3
"""Offline precision report: verl-gr TensorBoard vs TRL reference run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import write_offline_tb_alignment_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fork-tb",
        default=str(_REPO / "tensorboard_log/RankGRPO/logprob_align_v015_TP2_g0_1"),
        help="verl-gr TensorBoard logdir",
    )
    parser.add_argument(
        "--trl-tb",
        default=(
            "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/"
            "logs/debug_precision_verlgr/runs/Jul06_12-05-38_hk01dgx028"
        ),
        help="TRL reference TensorBoard logdir",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO / "outputs/logprob_align_v015_TP2_g0_1"),
        help="Write logs/precision_align_vs_trl_debug.md here",
    )
    parser.add_argument("--experiment", default="logprob_align_v015_TP2_g0_1")
    parser.add_argument("--max-step", type=int, default=100, help="Cap comparison at this step")
    parser.add_argument("--report-stem", default="precision_align_vs_trl_debug")
    args = parser.parse_args()

    path = write_offline_tb_alignment_report(
        fork_tb_dir=args.fork_tb,
        trl_tb_dir=args.trl_tb,
        output_dir=args.output_dir,
        experiment_name=args.experiment,
        max_step=args.max_step,
        report_stem=args.report_stem,
    )
    if path is None:
        raise SystemExit(1)
    print(path)


if __name__ == "__main__":
    main()
