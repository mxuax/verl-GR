#!/usr/bin/env python3
"""Compare two Nsight Systems NVTX summaries and rank timing differences.

Usage:
  python scripts/compare_nsys_nvtx.py \
    --left mini_nvtxsum.csv --right verl_nvtxsum.csv --topk 30
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NvtxStat:
    name: str
    total_us: float
    avg_us: float
    calls: float


def _normalize_header(h: str) -> str:
    return "".join(ch.lower() for ch in h if ch.isalnum())


def _pick_column(headers: list[str], candidates: list[str]) -> str | None:
    norm = {_normalize_header(h): h for h in headers}
    for cand in candidates:
        key = _normalize_header(cand)
        if key in norm:
            return norm[key]
    return None


def _to_float(v: str) -> float:
    v = (v or "").strip().replace(",", "")
    if not v:
        return 0.0
    try:
        return float(v)
    except ValueError:
        # Nsight may emit "1.23e+06 ns" style strings in some formats.
        for unit, scale in (("ns", 1e-3), ("us", 1.0), ("ms", 1e3), ("s", 1e6)):
            if v.lower().endswith(unit):
                base = v[: -len(unit)].strip()
                return float(base) * scale
        raise


def load_nvtx_csv(path: Path) -> dict[str, NvtxStat]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        headers = [h.strip() for h in reader.fieldnames]

        name_col = _pick_column(
            headers,
            [
                "Range Name",
                "Range",
                "NVTX Range",
                "Name",
            ],
        )
        total_col = _pick_column(
            headers,
            [
                "Total Time (us)",
                "Total Time (ns)",
                "Total Time",
                "Time (us)",
                "Time (ns)",
            ],
        )
        avg_col = _pick_column(
            headers,
            [
                "Avg (us)",
                "Avg (ns)",
                "Average (us)",
                "Average (ns)",
                "Avg",
            ],
        )
        calls_col = _pick_column(headers, ["Instances", "Calls", "Count"])

        if not name_col or not total_col:
            raise ValueError(
                f"Cannot find NVTX name/total columns in {path}. "
                f"Found headers: {headers}"
            )

        out: dict[str, NvtxStat] = {}
        for row in reader:
            name = (row.get(name_col) or "").strip()
            if not name:
                continue
            total_raw = _to_float(row.get(total_col, "0"))
            # If total column is ns, convert to us.
            total_is_ns = "ns" in total_col.lower() and "us" not in total_col.lower()
            total_us = total_raw * (1e-3 if total_is_ns else 1.0)

            if avg_col:
                avg_raw = _to_float(row.get(avg_col, "0"))
                avg_is_ns = "ns" in avg_col.lower() and "us" not in avg_col.lower()
                avg_us = avg_raw * (1e-3 if avg_is_ns else 1.0)
            else:
                avg_us = 0.0

            calls = _to_float(row.get(calls_col, "0")) if calls_col else 0.0
            out[name] = NvtxStat(name=name, total_us=total_us, avg_us=avg_us, calls=calls)
        return out


def fmt_us(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v / 1e6:.3f}s"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.3f}ms"
    return f"{v:.3f}us"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path, help="Baseline NVTX CSV (e.g., MiniOneRec)")
    parser.add_argument("--right", required=True, type=Path, help="Compared NVTX CSV (e.g., verl-GR)")
    parser.add_argument("--left-name", default="left", help="Label for baseline")
    parser.add_argument("--right-name", default="right", help="Label for compared side")
    parser.add_argument("--topk", type=int, default=30, help="Top-K rows by absolute total-time diff")
    parser.add_argument(
        "--contains",
        default="",
        help="Optional comma-separated substrings to keep (e.g. gen.generate,ref.forward)",
    )
    args = parser.parse_args()

    left = load_nvtx_csv(args.left)
    right = load_nvtx_csv(args.right)
    names = sorted(set(left.keys()) | set(right.keys()))

    filters = [x.strip() for x in args.contains.split(",") if x.strip()]
    if filters:
        names = [n for n in names if any(f in n for f in filters)]

    rows = []
    for name in names:
        l = left.get(name, NvtxStat(name, 0.0, 0.0, 0.0))
        r = right.get(name, NvtxStat(name, 0.0, 0.0, 0.0))
        diff = r.total_us - l.total_us
        speed = (r.total_us / l.total_us) if l.total_us > 0 else math.inf
        rows.append((name, l, r, diff, speed))

    rows.sort(key=lambda x: abs(x[3]), reverse=True)
    rows = rows[: max(args.topk, 1)]

    print(
        f"# NVTX diff ({args.left_name} -> {args.right_name})\n"
        f"- left:  {args.left}\n"
        f"- right: {args.right}\n"
        f"- rows:  {len(rows)}\n"
    )
    print(
        "| Range | left total | right total | delta(right-left) | speed(right/left) | "
        "left avg | right avg | left calls | right calls |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, l, r, diff, speed in rows:
        speed_str = "inf" if math.isinf(speed) else f"{speed:.3f}x"
        print(
            f"| {name} | {fmt_us(l.total_us)} | {fmt_us(r.total_us)} | "
            f"{fmt_us(diff)} | {speed_str} | {fmt_us(l.avg_us)} | {fmt_us(r.avg_us)} | "
            f"{l.calls:.0f} | {r.calls:.0f} |"
        )


if __name__ == "__main__":
    main()
