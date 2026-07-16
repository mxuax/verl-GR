#!/usr/bin/env python3
"""Export TRL logprob gate probes from TensorBoard into rankgrpo_gate_sidecar.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import _GATE_LOGPROB_CHECKS, _load_trl_series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trl-ref",
        required=True,
        help="TRL TensorBoard run directory (TRL_REF)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Sidecar path (default: {trl-ref}/rankgrpo_gate_sidecar.json)",
    )
    args = parser.parse_args()

    trl_dir = Path(args.trl_ref)
    out_path = Path(args.output) if args.output else trl_dir / "rankgrpo_gate_sidecar.json"
    payload: dict[str, dict[str, float]] = {}

    for check in _GATE_LOGPROB_CHECKS:
        for tag in check["trl_tags"]:
            series = _load_trl_series(tag, str(trl_dir))
            if series:
                payload[tag] = {str(step): value for step, value in series}
                print(f"exported {len(series)} points for {tag}")
                break

    if not payload:
        raise SystemExit(f"No logprob gate tags found under {trl_dir}")

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
