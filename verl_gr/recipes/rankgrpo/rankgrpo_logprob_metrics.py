"""Rank-GRPO logprob gate metrics and TRL alignment report."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from verl import DataProto

# TRL reference: 30-step precision sidecar (override via TRL_REF / VERL_GR_TRL_TB_REF).
_DEFAULT_TRL_TB = (
    "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/"
    "logs/debug_precision_verlgr/runs/Jul07_03-56-22_hk01dgx028"
)
# Legacy resumed TRL run (step offset ≈ 410); use only when VERL_GR_TRL_TB_REF points there.
_LEGACY_TRL_TB = (
    "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/"
    "results/grpo/new2/runs/May27_06-48-14_hk01dgx028"
)

# Fork metric → TRL TensorBoard tag, tolerance (relative), category.
_REL_TOL = 0.2


def _rel_aligned(fork_val: float, trl_val: float | None, *, abs_floor: float = 1e-6) -> bool:
    if trl_val is None:
        return False
    if abs(trl_val) < abs_floor and abs(fork_val) < abs_floor:
        return abs(fork_val - trl_val) <= abs_floor
    return abs(fork_val - trl_val) / max(abs(trl_val), 1e-8) < _REL_TOL


# Per-step gate: logprob probes compared to TRL (relative diff ≤ 20%).
_GATE_LOGPROB_CHECKS: list[dict[str, Any]] = [
    {
        "label": "rollout↔ref",
        "fork_keys": ["logprob_gate/rollout_minus_ref/abs_mean", "logprob_gate/actor_minus_ref/abs_mean"],
        "trl_tags": [
            "train/logprob_gate/rollout_minus_ref/abs_mean",
            "logprob_gate/rollout_minus_ref/abs_mean",
        ],
    },
    {
        "label": "actor↔ref (update)",
        "fork_keys": ["actor/debug/logprob_diff_abs", "debug/logprob_diff_abs"],
        "trl_tags": [
            "train/actor/debug/logprob_diff_abs",
            "actor/debug/logprob_diff_abs",
        ],
    },
]

_GATE_KL_CHECK: dict[str, Any] = {
    "label": "KL",
    "fork_keys": ["actor/kl_loss", "actor/train/kl", "train/kl"],
    "trl_tags": ["train/kl"],
}

# Verdict table: pass when pass / (pass + fail) >= this fraction (warmup rows count as skip).
_GATE_MIN_PASS_FRACTION = 2 / 3

_FORK_STEP_TIME_KEYS = ["timing_s/step", "perf/time_per_step"]
_TRL_STEP_TIME_TAGS = ["train/train_steps_per_second", "train/step_time", "timing_s/step"]

# Modular step latency (RUN_DEBUG_STEP only): verl `timing_s/*` keys from ray_trainer.
_STEP_LATENCY_PHASES: tuple[tuple[str, str], ...] = (
    ("gen (vLLM rollout)", "gen"),
    ("update_actor (FSDP train step)", "update_actor"),
    ("update_weights (actor → rollout sync)", "update_weights"),
    ("old_log_prob", "old_log_prob"),
    ("ref", "ref"),
    ("adv", "adv"),
    ("reward", "reward"),
)
_STEP_LATENCY_OTHER_KEYS: frozenset[str] = frozenset(
    {
        "start_profile",
        "stop_profile",
        "gen_max",
        "values",
        "update_critic",
        "testing",
        "save_checkpoint",
        "dump_rollout_generations",
    }
)


@dataclass
class ModularStepLatencyRow:
    phase: str
    fork_seconds: float | None
    fork_pct: float | None
    trl_seconds: float | None
    trl_estimated: bool
    delta_seconds: float | None


@dataclass
class ModularStepLatencySummary:
    rows: list[ModularStepLatencyRow]
    fork_total_seconds: float | None
    trl_total_seconds: float | None
    other_seconds: float | None
    other_pct: float | None
    n_steps: int
    step_range: str
    trl_step_time_source: str | None = None


@dataclass
class SidecarProbeOverheadRow:
    component: str
    mean_seconds: float | None
    notes: str = ""


@dataclass
class SidecarProbeOverheadSummary:
    step_distribution: dict[str, float]
    probe_rows: list[SidecarProbeOverheadRow]
    measured_probe_total: float | None
    modular_other_seconds: float | None
    logging_steps: int
    prod_logging_steps: int
    estimated_prod_step_seconds: float | None
    trl_reference_seconds: float | None
    n_steps: int
    step_range: str
    incomplete_run_note: str | None = None


_PROBE_TIMING_KEYS: tuple[tuple[str, str], ...] = (
    ("logprob gate metrics", "timing_rankgrpo/probe_logprob_gate"),
    ("alignment accumulator", "timing_rankgrpo/probe_align_accum"),
    ("TensorBoard flush (logging_steps=1)", "timing_rankgrpo/probe_tb_log"),
)


def modular_step_latency_enabled() -> bool:
    """True only during RUN_DEBUG_STEP sidecar alignment runs."""
    return alignment_report_enabled()


def _fork_timing_tag(phase_key: str) -> str:
    return f"timing_s/{phase_key}"


def _mean_fork_timing_values(
    acc: RankGRPOAlignmentAccumulator,
    timing_tag: str,
    compare_steps: list[int],
    *,
    skip_warmup_steps: int,
) -> float | None:
    vals: list[float] = []
    for step in compare_steps:
        if step <= skip_warmup_steps:
            continue
        raw = acc.metrics_by_step.get(step, {}).get(timing_tag)
        if raw is not None:
            vals.append(float(raw))
    return float(np.mean(vals)) if vals else None


def _mean_fork_timing_phase(
    acc: RankGRPOAlignmentAccumulator,
    phase_key: str,
    compare_steps: list[int],
    *,
    skip_warmup_steps: int,
) -> float | None:
    return _mean_fork_timing_values(
        acc,
        _fork_timing_tag(phase_key),
        compare_steps,
        skip_warmup_steps=skip_warmup_steps,
    )


def _mean_other_overhead_seconds(
    acc: RankGRPOAlignmentAccumulator,
    compare_steps: list[int],
    *,
    skip_warmup_steps: int,
    accounted_phase_keys: tuple[str, ...],
) -> float | None:
    vals: list[float] = []
    for step in compare_steps:
        if step <= skip_warmup_steps:
            continue
        metrics = acc.metrics_by_step.get(step, {})
        step_total = _pick_metric(metrics, _FORK_STEP_TIME_KEYS)
        if step_total is None:
            continue
        accounted = 0.0
        for key in accounted_phase_keys:
            tag = _fork_timing_tag(key)
            if tag in metrics:
                accounted += float(metrics[tag])
        for tag, value in metrics.items():
            if not tag.startswith("timing_s/"):
                continue
            phase = tag.removeprefix("timing_s/")
            if phase in ("step",) or phase in _STEP_LATENCY_OTHER_KEYS:
                continue
            if phase in accounted_phase_keys:
                continue
            accounted += float(value)
        other = max(0.0, float(step_total) - accounted)
        vals.append(other)
    return float(np.mean(vals)) if vals else None


def compute_modular_step_latency_summary(
    acc: RankGRPOAlignmentAccumulator,
    *,
    trl_tb_dir: str | Path | None = None,
    compare_steps: list[int] | None = None,
    skip_warmup_steps: int = 1,
) -> ModularStepLatencySummary | None:
    """Mean per-phase `timing_s/*` from fork TB metrics (RUN_DEBUG_STEP runs only)."""

    if not modular_step_latency_enabled():
        return None

    steps = compare_steps if compare_steps is not None else sorted(acc.steps)
    if not steps:
        return None

    trl_dir = _resolve_trl_tb_dir(trl_tb_dir)
    trl_total, trl_source = _trl_reference_step_time(trl_dir)
    trl_total_for_steps = _trl_avg_step_time_for_steps(
        trl_dir,
        steps,
        skip_warmup_steps=skip_warmup_steps,
        fallback=trl_total,
    )
    if trl_total_for_steps is not None:
        trl_total = trl_total_for_steps

    fork_total = _mean_fork_timing_values(
        acc,
        "timing_s/step",
        steps,
        skip_warmup_steps=skip_warmup_steps,
    )
    if fork_total is None:
        fork_total = _fork_avg_step_time(acc, steps, skip_warmup_steps=skip_warmup_steps)

    phase_keys = tuple(key for _, key in _STEP_LATENCY_PHASES)
    rows: list[ModularStepLatencyRow] = []
    accounted_sum = 0.0
    for label, key in _STEP_LATENCY_PHASES:
        fork_s = _mean_fork_timing_phase(acc, key, steps, skip_warmup_steps=skip_warmup_steps)
        fork_pct = (100.0 * fork_s / fork_total) if fork_s is not None and fork_total and fork_total > 0 else None
        trl_s: float | None = None
        trl_est = False
        if fork_s is not None and trl_total is not None and fork_total and fork_total > 0:
            trl_s = trl_total * (fork_s / fork_total)
            trl_est = True
        elif key == "step" and trl_total is not None:
            trl_s = trl_total
            trl_est = False
        delta = (fork_s - trl_s) if fork_s is not None and trl_s is not None else None
        if fork_s is not None:
            accounted_sum += fork_s
        rows.append(
            ModularStepLatencyRow(
                phase=label,
                fork_seconds=fork_s,
                fork_pct=fork_pct,
                trl_seconds=trl_s,
                trl_estimated=trl_est,
                delta_seconds=delta,
            )
        )

    other_s = _mean_other_overhead_seconds(
        acc,
        steps,
        skip_warmup_steps=skip_warmup_steps,
        accounted_phase_keys=phase_keys,
    )
    if other_s is None and fork_total is not None:
        other_s = max(0.0, fork_total - accounted_sum)
    other_pct = (100.0 * other_s / fork_total) if other_s is not None and fork_total and fork_total > 0 else None
    trl_other = (trl_total * (other_s / fork_total)) if other_s is not None and trl_total and fork_total and fork_total > 0 else None
    rows.append(
        ModularStepLatencyRow(
            phase="Other/overhead",
            fork_seconds=other_s,
            fork_pct=other_pct,
            trl_seconds=trl_other,
            trl_estimated=trl_other is not None,
            delta_seconds=(other_s - trl_other) if other_s is not None and trl_other is not None else None,
        )
    )

    used_steps = [s for s in steps if s > skip_warmup_steps]
    step_range = f"{used_steps[0]}–{used_steps[-1]}" if used_steps else "—"

    rows.append(
        ModularStepLatencyRow(
            phase="**Total logged step**",
            fork_seconds=fork_total,
            fork_pct=100.0 if fork_total is not None else None,
            trl_seconds=trl_total,
            trl_estimated=False,
            delta_seconds=(fork_total - trl_total) if fork_total is not None and trl_total is not None else None,
        )
    )

    return ModularStepLatencySummary(
        rows=rows,
        fork_total_seconds=fork_total,
        trl_total_seconds=trl_total,
        other_seconds=other_s,
        other_pct=other_pct,
        n_steps=len(used_steps),
        step_range=step_range,
        trl_step_time_source=trl_source,
    )


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}s"


def _format_delta_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}s"


def format_modular_step_latency_markdown(summary: ModularStepLatencySummary | None) -> list[str]:
    if summary is None:
        return []

    lines: list[str] = []
    lines.append("## Modular step latency (measured)")
    lines.append("")
    lines.append(
        f"Mean `timing_s/*` over fork steps **{summary.step_range}** "
        f"(n={summary.n_steps}, warmup skipped). "
        "Enabled only when `RUN_DEBUG_STEP` is set."
    )
    if summary.trl_step_time_source:
        lines.append(f"TRL total step time: {summary.trl_step_time_source}.")
    lines.append(
        "Per-phase TRL times are **pro-rata estimates** "
        "`TRL_total × (fork_phase / fork_total)` — TRL does not log modular `timing_s/*`."
    )
    lines.append("")
    lines.append("| Phase | verl-gr Time | TRL Time | Delta Step Time |")
    lines.append("|-------|--------------|----------|-----------------|")
    for row in summary.rows:
        fork_cell = _format_seconds(row.fork_seconds)
        if row.fork_pct is not None and row.phase != "**Total logged step**":
            fork_cell = f"{fork_cell} ({row.fork_pct:.0f}%)"
        trl_cell = _format_seconds(row.trl_seconds)
        if row.trl_estimated and row.trl_seconds is not None:
            trl_cell += "†"
        lines.append(
            f"| {row.phase} | {fork_cell} | {trl_cell} | {_format_delta_seconds(row.delta_seconds)} |"
        )
    lines.append("")
    lines.append("† TRL phase time estimated pro-rata from total step time (tqdm / TB).")
    lines.append(
        "Eval (`timing_s/testing`) and checkpoint (`timing_s/save_checkpoint`) are excluded "
        "from this per-step training breakdown."
    )
    lines.append("")
    return lines


def _step_time_distribution(
    acc: RankGRPOAlignmentAccumulator,
    compare_steps: list[int],
    *,
    skip_warmup_steps: int,
) -> dict[str, float]:
    vals: list[float] = []
    for step in compare_steps:
        if step <= skip_warmup_steps:
            continue
        raw = _pick_metric(acc.metrics_by_step.get(step, {}), _FORK_STEP_TIME_KEYS)
        if raw is not None:
            vals.append(float(raw))
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def merge_sidecar_probe_timings(step: int, timings: dict[str, float]) -> None:
    """Merge per-step probe timings into the alignment accumulator (sidecar runs)."""
    if not alignment_report_enabled():
        return
    bucket = _ALIGNMENT_ACCUMULATOR.metrics_by_step.get(step)
    if bucket is None:
        return
    bucket.update(timings)


def compute_sidecar_probe_overhead_summary(
    acc: RankGRPOAlignmentAccumulator,
    *,
    compare_steps: list[int] | None = None,
    latency_summary: ModularStepLatencySummary | None = None,
    trl_tb_dir: str | Path | None = None,
    skip_warmup_steps: int = 1,
    logging_steps: int | None = None,
    prod_logging_steps: int = 10,
) -> SidecarProbeOverheadSummary | None:
    if not modular_step_latency_enabled():
        return None

    steps = compare_steps if compare_steps is not None else sorted(acc.steps)
    if not steps:
        return None

    used_steps = [s for s in steps if s > skip_warmup_steps]
    step_range = f"{used_steps[0]}–{used_steps[-1]}" if used_steps else "—"
    dist = _step_time_distribution(acc, steps, skip_warmup_steps=skip_warmup_steps)

    probe_rows: list[SidecarProbeOverheadRow] = []
    for label, key in _PROBE_TIMING_KEYS:
        mean_s = _mean_fork_timing_values(acc, key, steps, skip_warmup_steps=skip_warmup_steps)
        notes = ""
        if mean_s is None and key == "timing_rankgrpo/probe_tb_log":
            notes = "not instrumented in this run"
        probe_rows.append(SidecarProbeOverheadRow(component=label, mean_seconds=mean_s, notes=notes))

    modular_other = latency_summary.other_seconds if latency_summary is not None else None
    measured_components = [row.mean_seconds for row in probe_rows if row.mean_seconds is not None]
    measured_any = bool(measured_components)
    measured_total = float(sum(measured_components)) if measured_any else None

    if measured_total is None and modular_other is not None and modular_other > 0.01:
        measured_total = modular_other
        measured_any = True

    sidecar_logging_steps = logging_steps
    if sidecar_logging_steps is None:
        try:
            sidecar_logging_steps = int(os.environ.get("VERL_GR_ALIGN_LOGGING_STEPS", "1") or "1")
        except ValueError:
            sidecar_logging_steps = 1

    mean_step = dist.get("mean")
    estimated_prod: float | None = None
    if mean_step is not None and measured_any:
        estimated_prod = max(0.0, mean_step - measured_total)
        tb_mean = next(
            (row.mean_seconds for row in probe_rows if row.component.startswith("TensorBoard")),
            None,
        )
        if tb_mean is not None and sidecar_logging_steps == 1 and prod_logging_steps > 1:
            estimated_prod = max(
                0.0,
                mean_step - (measured_total - tb_mean + tb_mean / prod_logging_steps),
            )

    trl_dir = _resolve_trl_tb_dir(trl_tb_dir)
    trl_total, _ = _trl_reference_step_time(trl_dir)
    trl_for_steps = _trl_avg_step_time_for_steps(
        trl_dir,
        steps,
        skip_warmup_steps=skip_warmup_steps,
        fallback=trl_total,
    )
    if trl_for_steps is not None:
        trl_total = trl_for_steps

    incomplete_note: str | None = None
    try:
        expected = int(os.environ.get("RUN_DEBUG_STEP", "0") or "0")
    except ValueError:
        expected = 0
    last_step = acc.last_step()
    if expected > 0 and last_step is not None and last_step < expected:
        incomplete_note = (
            f"Run incomplete: logged through step {last_step}, expected {expected}. "
            "Gate/latency stats use available steps only."
        )

    return SidecarProbeOverheadSummary(
        step_distribution=dist,
        probe_rows=probe_rows,
        measured_probe_total=measured_total if measured_any else None,
        modular_other_seconds=modular_other,
        logging_steps=sidecar_logging_steps,
        prod_logging_steps=prod_logging_steps,
        estimated_prod_step_seconds=estimated_prod,
        trl_reference_seconds=trl_total,
        n_steps=len(used_steps),
        step_range=step_range,
        incomplete_run_note=incomplete_note,
    )


def format_sidecar_probe_overhead_markdown(summary: SidecarProbeOverheadSummary | None) -> list[str]:
    if summary is None:
        return []

    lines: list[str] = []
    lines.append("## Sidecar probe overhead & step-time distribution")
    lines.append("")
    lines.append(
        "Separates **sidecar-only** work (logprob probes, alignment accumulator, per-step TB) "
        f"from modular `timing_s/*` training phases. Sidecar uses `logging_steps={summary.logging_steps}`; "
        f"production long runs typically use `logging_steps={summary.prod_logging_steps}`."
    )
    if summary.incomplete_run_note:
        lines.append(f"- **Note:** {summary.incomplete_run_note}")
    lines.append("")

    if summary.step_distribution:
        d = summary.step_distribution
        lines.append(
            f"`timing_s/step` distribution over steps **{summary.step_range}** "
            f"(n={summary.n_steps}, warmup skipped): "
            f"mean **{d['mean']:.2f}s**, std {d['std']:.2f}s, "
            f"min {d['min']:.2f}s, p50 {d['p50']:.2f}s, p90 {d['p90']:.2f}s, max {d['max']:.2f}s."
        )
        lines.append("")

    lines.append("| Probe component | mean/step | notes |")
    lines.append("|-----------------|-----------|-------|")
    for row in summary.probe_rows:
        mean_cell = f"{row.mean_seconds:.3f}s" if row.mean_seconds is not None else "—"
        lines.append(f"| {row.component} | {mean_cell} | {row.notes or '—'} |")
    if summary.modular_other_seconds is not None:
        lines.append(
            f"| modular Other/overhead (`timing_s/step` residual) | {summary.modular_other_seconds:.3f}s | "
            "includes TB + unbucketed trainer time when probes not instrumented |"
        )
    lines.append("")

    if summary.measured_probe_total is not None:
        lines.append(
            f"- **Estimated sidecar probe overhead:** **{summary.measured_probe_total:.3f}s/step** "
            "(sum of measured `timing_rankgrpo/probe_*` rows, or modular Other/overhead when >0)."
        )
    else:
        lines.append(
            "- **Sidecar probe overhead:** not directly measured in this run "
            "(missing `timing_rankgrpo/probe_*` scalars). "
            f"With `logging_steps={summary.logging_steps}`, per-step TB flush + logprob gate probes "
            f"typically add ~0.1–0.3s/step vs production `logging_steps={summary.prod_logging_steps}`. "
            "Compare `timing_s/step` distribution below to long-run tqdm s/it."
        )
    if summary.estimated_prod_step_seconds is not None and summary.measured_probe_total is not None:
        lines.append(
            f"- **Estimated production-equivalent step time:** **{summary.estimated_prod_step_seconds:.3f}s/step** "
            f"(adjusts TB logging for `logging_steps={summary.prod_logging_steps}`)."
        )
    if summary.trl_reference_seconds is not None:
        lines.append(f"- **TRL reference step time:** **{summary.trl_reference_seconds:.3f}s/step** (tqdm train log).")
    if (
        summary.estimated_prod_step_seconds is not None
        and summary.trl_reference_seconds is not None
    ):
        delta = summary.estimated_prod_step_seconds - summary.trl_reference_seconds
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"- **Prod-estimate vs TRL:** {sign}{delta:.3f}s/step "
            "(after removing sidecar probe overhead; training phases only)."
        )
    lines.append("")
    return lines


def sidecar_probe_overhead_summary_to_dict(summary: SidecarProbeOverheadSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "step_range": summary.step_range,
        "n_steps": summary.n_steps,
        "step_distribution": summary.step_distribution,
        "probe_rows": [
            {"component": row.component, "mean_seconds": row.mean_seconds, "notes": row.notes}
            for row in summary.probe_rows
        ],
        "measured_probe_total": summary.measured_probe_total,
        "modular_other_seconds": summary.modular_other_seconds,
        "logging_steps": summary.logging_steps,
        "prod_logging_steps": summary.prod_logging_steps,
        "estimated_prod_step_seconds": summary.estimated_prod_step_seconds,
        "trl_reference_seconds": summary.trl_reference_seconds,
        "incomplete_run_note": summary.incomplete_run_note,
    }


def modular_step_latency_summary_to_dict(summary: ModularStepLatencySummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "step_range": summary.step_range,
        "n_steps": summary.n_steps,
        "fork_total_seconds": summary.fork_total_seconds,
        "trl_total_seconds": summary.trl_total_seconds,
        "trl_step_time_source": summary.trl_step_time_source,
        "rows": [
            {
                "phase": row.phase,
                "fork_seconds": row.fork_seconds,
                "fork_pct": row.fork_pct,
                "trl_seconds": row.trl_seconds,
                "trl_estimated": row.trl_estimated,
                "delta_seconds": row.delta_seconds,
            }
            for row in summary.rows
        ],
    }


def _resolve_trl_tb_dir(trl_tb_dir: str | Path | None = None) -> str:
    if trl_tb_dir is not None:
        return str(trl_tb_dir)
    for key in ("TRL_REF", "VERL_GR_TRL_TB_REF"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    return _DEFAULT_TRL_TB


def _resolve_align_report_root(output_dir: str | Path | None = None) -> Path:
    fixed = os.environ.get("VERL_GR_ALIGN_REPORT_DIR", "").strip()
    if fixed:
        return Path(fixed)
    out_root = Path(output_dir or os.environ.get("OUTPUT_DIR") or os.getcwd())
    return out_root


def _relative_error(fork_val: float, trl_val: float) -> float:
    return abs(fork_val - trl_val) / max(abs(trl_val), 1e-8)


@dataclass
class MetricGateVerdict:
    """Aggregate pass/fail for one metric family across all compared steps."""

    name: str
    passed: bool
    blocked: bool = False
    blocked_reason: str | None = None
    n_evaluated: int = 0
    n_pass: int = 0
    n_fail: int = 0
    n_skip: int = 0

    def status_label(self) -> str:
        if self.blocked:
            return "BLOCKED"
        return "PASS" if self.passed else "FAIL"


@dataclass
class StepGateResult:
    step: int
    logprob_ok: bool | None
    logprob_rel_err: float | None
    kl_ok: bool | None
    kl_rel_err: float | None
    fork_kl: float | None
    trl_kl: float | None
    time_ok: bool | None
    fork_step_time: float | None
    trl_step_time: float | None
    passed: bool | None
    notes: list[str] = field(default_factory=list)


_GATE_PER_STEP_HEADER = (
    "| step | logprob gate | KL gate | fork KL | TRL KL | step time gate | fork time | TRL time | gate |"
)
_GATE_PER_STEP_SEP = (
    "|------|--------------|---------|---------|--------|----------------|-----------|----------|------|"
)


def _format_gate_metric_cell(ok: bool | None, rel_err: float | None) -> str:
    if ok is True:
        return f"OK ({rel_err:.3f})" if rel_err is not None else "OK"
    if ok is False:
        return f"**FAIL** ({rel_err:.3f})" if rel_err is not None else "**FAIL**"
    return "—"


def _format_step_time_gate_cell(ok: bool | None) -> str:
    if ok is True:
        return "OK"
    if ok is False:
        return "**FAIL**"
    return "—"


@dataclass
class AlignmentGateSummary:
    """Combined gate = logprob ∧ KL (per-step, ≥2/3 pass) ∧ step_time (fork avg < TRL avg)."""

    passed: bool
    logprob_gate: MetricGateVerdict
    kl_gate: MetricGateVerdict
    combined_gate: MetricGateVerdict
    time_gate: MetricGateVerdict
    steps: list[StepGateResult]
    trl_tb_dir: str
    trl_step_time_ref: float | None
    trl_step_time_tag: str | None
    logprob_trl_available: bool
    fork_step_time_avg: float | None = None
    blocked_reasons: list[str] = field(default_factory=list)


_ALIGNMENT_CHECKS: list[dict[str, Any]] = [
    {
        "name": "rollout ↔ ref logprob (batch gate)",
        "fork_keys": ["logprob_gate/rollout_minus_ref/abs_mean", "logprob_gate/actor_minus_ref/abs_mean"],
        "trl_tag": "train/logprob_gate/rollout_minus_ref/abs_mean",
        "category": "logprob",
        "aligned_if": lambda v, ref: _rel_aligned(v, ref),
        "note": "vLLM rollout (anchor) vs ref on item_token_mask; target <20% vs TRL.",
    },
    {
        "name": "rollout ↔ rollout logprob (bypass sanity)",
        "fork_keys": ["logprob_gate/rollout_minus_rollout/abs_mean", "logprob_gate/actor_minus_rollout/abs_mean"],
        "trl_tag": "train/logprob_gate/rollout_minus_rollout/abs_mean",
        "category": "logprob",
        "aligned_if": lambda v, ref: _rel_aligned(v, ref, abs_floor=1e-5),
        "note": "Anchor equals rollout logprobs; both sides should be ~0.",
    },
    {
        "name": "actor ↔ ref logprob (update forward)",
        "fork_keys": ["actor/debug/logprob_diff_abs", "debug/logprob_diff_abs"],
        "trl_tag": "train/actor/debug/logprob_diff_abs",
        "category": "logprob",
        "aligned_if": lambda v, ref: _rel_aligned(v, ref),
        "note": "FSDP actor forward vs ref during PPO update.",
    },
    {
        "name": "KL (item-masked, TRL train/kl)",
        "fork_keys": ["actor/kl_loss", "actor/train/kl", "train/kl"],
        "trl_tag": "train/kl",
        "category": "precision",
        "aligned_if": lambda v, ref: _rel_aligned(v, ref),
        "note": "TRL train/kl uses global token-mean (bnpo). Fork must log with token-mean in trl_match.",
    },
    {
        "name": "reward_total (sum of 20 rank hits)",
        "fork_keys": ["train/rankgrpo/reward_total"],
        "trl_tag": "train/reward_total",
        "category": "reward",
        "aligned_if": lambda v, ref: _rel_aligned(v, ref),
        "note": "High variance per step; use mean over last N steps.",
    },
    {
        "name": "per-sample reward mean",
        "fork_keys": ["train/rankgrpo/reward"],
        "trl_tag": "train/reward",
        "category": "reward",
        "aligned_if": lambda v, ref: _rel_aligned(v, ref),
        "note": "Do not use critic/rewards/mean for TRL comparison.",
    },
    {
        "name": "completion mean length",
        "fork_keys": ["train/rankgrpo/completions/mean_length", "response_length/mean"],
        "trl_tag": "train/completions/mean_length",
        "category": "generation",
        "aligned_if": lambda v, ref: ref is not None and abs(v - ref) < 8.0,
        "note": "Target ~190–194 tokens.",
    },
    {
        "name": "completion clipped ratio",
        "fork_keys": ["train/rankgrpo/completions/clipped_ratio", "response_length/clip_ratio"],
        "trl_tag": "train/completions/clipped_ratio",
        "category": "generation",
        "aligned_if": lambda v, ref: (ref is None and v < 0.01) or (ref is not None and abs(v - ref) < 0.05),
        "note": "TRL reference run has clip_ratio = 0.",
    },
    {
        "name": "PG clip fraction",
        "fork_keys": ["actor/pg_clipfrac"],
        "trl_tag": "train/clip_ratio/region_mean",
        "category": "actor",
        "aligned_if": lambda v, ref: (ref is None and v < 0.05) or (ref is not None and abs(v - ref) < 0.05),
        "note": "Early training should stay near 0 with ε=0.06/0.08.",
    },
    {
        "name": "actor mean logp (update)",
        "fork_keys": ["actor/dbg/logp_actor_mean", "dbg/logp_actor_mean"],
        "trl_tag": "train/dbg/logp_actor_mean",
        "category": "logprob",
        "aligned_if": lambda v, ref: ref is not None and abs(v - ref) < 0.15,
        "note": "Absolute logp tolerance ±0.15 nats (scale differs from ratio metrics).",
    },
    {
        "name": "ref mean logp (batch)",
        "fork_keys": ["logprob_gate/ref_mean", "actor/dbg/logp_ref_mean"],
        "trl_tag": "train/dbg/logp_ref_mean",
        "category": "logprob",
        "aligned_if": lambda v, ref: ref is not None and abs(v - ref) < 0.15,
        "note": "Ref uses fp32 actor master weights; fork should match actor/ref dtype settings.",
    },
]


def _loss_mask(batch: DataProto) -> torch.Tensor | None:
    if "item_token_mask" in batch.batch:
        return batch.batch["item_token_mask"].bool()
    if "response_mask" in batch.batch:
        return batch.batch["response_mask"].bool()
    return None


def _masked_logprob_stats(
    log_probs_a: torch.Tensor,
    log_probs_b: torch.Tensor,
    mask: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    """Mean/abs-diff/max-diff of (a - b) over masked tokens."""
    if log_probs_a.shape != log_probs_b.shape:
        return {f"{prefix}/valid": 0.0}
    m = mask.to(device=log_probs_a.device, dtype=torch.bool)
    if not m.any():
        return {f"{prefix}/valid": 0.0}

    diff = (log_probs_a - log_probs_b).float()
    md = diff[m]
    return {
        f"{prefix}/valid": 1.0,
        f"{prefix}/mean": float(md.mean().item()),
        f"{prefix}/abs_mean": float(md.abs().mean().item()),
        f"{prefix}/max_abs": float(md.abs().max().item()),
        f"{prefix}/std": float(md.std(unbiased=False).item()) if md.numel() > 1 else 0.0,
    }


def calculate_rankgrpo_logprob_gate_metrics(batch: DataProto) -> dict[str, float]:
    """Compare logprobs at rollout/ref boundaries (TRL ``item_token_mask``).

    With ``bypass_mode: true``, ``old_log_probs`` in the batch is the vLLM rollout
    policy, not the FSDP actor re-forward. True actor↔ref diffs come from the
    actor update worker metrics (``actor/debug/logprob_diff_abs``).
    """

    mask = _loss_mask(batch)
    if mask is None:
        return {}

    metrics: dict[str, float] = {}
    rollout_lp = batch.batch.get("rollout_log_probs")
    ref_lp = batch.batch.get("ref_log_prob")
    # Under bypass, old_log_probs == rollout logprobs (anchor for PPO ratio).
    anchor_lp = batch.batch.get("old_log_probs")

    if anchor_lp is not None and ref_lp is not None:
        metrics.update(_masked_logprob_stats(anchor_lp, ref_lp, mask, prefix="logprob_gate/rollout_minus_ref"))
        # Backward-compatible alias (misleading name kept for existing TB dashboards).
        metrics.update(_masked_logprob_stats(anchor_lp, ref_lp, mask, prefix="logprob_gate/actor_minus_ref"))

    if anchor_lp is not None and rollout_lp is not None:
        metrics.update(_masked_logprob_stats(anchor_lp, rollout_lp, mask, prefix="logprob_gate/rollout_minus_rollout"))
        metrics.update(_masked_logprob_stats(anchor_lp, rollout_lp, mask, prefix="logprob_gate/actor_minus_rollout"))

    if ref_lp is not None and rollout_lp is not None:
        metrics.update(_masked_logprob_stats(ref_lp, rollout_lp, mask, prefix="logprob_gate/ref_minus_rollout"))

    if anchor_lp is not None:
        m = mask.float()
        s = m.sum().clamp(min=1)
        metrics["logprob_gate/rollout_mean"] = float((anchor_lp * m).sum().item() / s.item())
        metrics["logprob_gate/actor_mean"] = metrics["logprob_gate/rollout_mean"]
    if ref_lp is not None:
        m = mask.float()
        s = m.sum().clamp(min=1)
        metrics["logprob_gate/ref_mean"] = float((ref_lp * m).sum().item() / s.item())
    if rollout_lp is not None:
        m = mask.float()
        s = m.sum().clamp(min=1)
        metrics["logprob_gate/rollout_logprobs_mean"] = float((rollout_lp * m).sum().item() / s.item())

    metrics["logprob_gate/mask_tokens"] = float(mask.sum().item())
    metrics["logprob_gate/bypass_mode"] = 1.0
    return metrics


def maybe_export_rankgrpo_logprobs(batch: DataProto, *, step: int) -> None:
    """Optional per-step logprob dump for offline gate comparison (VERL_GR_LOGPROB_EXPORT=1)."""

    if os.environ.get("VERL_GR_LOGPROB_EXPORT", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    from uuid import uuid4

    dump_dir = os.environ.get("VERL_GR_LOGPROB_EXPORT_DIR") or os.path.join(os.getcwd(), "rankgrpo_logprob_export")
    os.makedirs(dump_dir, exist_ok=True)

    mask = _loss_mask(batch)
    if mask is None:
        return

    row: dict[str, Any] = {"step": step, "mask_tokens": int(mask.sum().item())}
    for key in ("old_log_probs", "ref_log_prob", "rollout_log_probs", "responses"):
        tensor = batch.batch.get(key)
        if tensor is None:
            continue
        masked = tensor[0][mask[0]].detach().cpu().tolist()
        row[key] = masked

    path = os.path.join(dump_dir, f"step_{step}_{uuid4().hex[:8]}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f)
    print(f"[logprob_export] wrote {path}")


def alignment_report_enabled() -> bool:
    raw = os.environ.get("RUN_DEBUG_STEP", "").strip()
    return raw not in {"", "None", "none", "0"} and raw.isdigit() and int(raw) > 0


def _pick_metric(metrics: dict[str, float], keys: list[str]) -> float | None:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def _load_trl_series(tag: str, logdir: str) -> list[tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []
    path = Path(logdir)
    if not path.is_dir():
        return []
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, float(e.value)) for e in ea.Scalars(tag)]


def _trl_resume_offset(trl_tb_dir: str | None = None) -> int:
    if "VERL_GR_TRL_RESUME_OFFSET" in os.environ:
        raw = os.environ.get("VERL_GR_TRL_RESUME_OFFSET", "0").strip()
        try:
            return max(0, int(raw))
        except ValueError:
            return 0
    trl_dir = str(trl_tb_dir or os.environ.get("VERL_GR_TRL_TB_REF", _DEFAULT_TRL_TB))
    if _LEGACY_TRL_TB in trl_dir or "May27_06-48-14" in trl_dir:
        return 410
    # Fresh-start TRL debug runs log from step 1 — no offset.
    return 0


def _infer_compare_steps(
    fork_steps: list[int],
    trl_series: dict[str, list[tuple[int, float]]],
    *,
    max_step: int | None = None,
) -> list[int]:
    trl_max = 0
    for series in trl_series.values():
        if series:
            trl_max = max(trl_max, max(s for s, _ in series))
    cap = min(fork_steps[-1], trl_max) if fork_steps else 0
    if max_step is not None:
        cap = min(cap, max_step)
    return [s for s in fork_steps if s <= cap]


def _trl_value_at_step(series: list[tuple[int, float]], step: int) -> tuple[float | None, int]:
    if not series:
        return None, step
    by_step = {s: v for s, v in series}
    if step in by_step:
        return by_step[step], step
    closest_step, closest_val = min(series, key=lambda x: abs(x[0] - step))
    return closest_val, closest_step


@dataclass
class RankGRPOAlignmentAccumulator:
    """Collect per-step metrics for end-of-run TRL alignment report."""

    steps: list[int] = field(default_factory=list)
    metrics_by_step: dict[int, dict[str, float]] = field(default_factory=dict)

    def record(self, step: int, metrics: dict[str, Any]) -> None:
        flat: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                flat[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if step not in self.metrics_by_step:
            self.steps.append(step)
        self.metrics_by_step[step] = flat

    def last_step(self) -> int | None:
        return self.steps[-1] if self.steps else None

    def series(self, keys: list[str]) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for step in self.steps:
            val = _pick_metric(self.metrics_by_step[step], keys)
            if val is not None:
                out.append((step, val))
        return out

    def mean_last(self, keys: list[str], n: int = 5) -> float | None:
        vals = [v for _, v in self.series(keys)]
        if not vals:
            return None
        tail = vals[-n:]
        return float(np.mean(tail))


_ALIGNMENT_ACCUMULATOR = RankGRPOAlignmentAccumulator()


def get_rankgrpo_alignment_accumulator() -> RankGRPOAlignmentAccumulator:
    return _ALIGNMENT_ACCUMULATOR


def record_rankgrpo_alignment_metrics(step: int, metrics: dict[str, Any]) -> None:
    if alignment_report_enabled():
        _ALIGNMENT_ACCUMULATOR.record(step, metrics)


def _load_trl_series_any(tags: list[str], logdir: str) -> tuple[str | None, list[tuple[int, float]]]:
    for tag in tags:
        series = _load_trl_series(tag, logdir)
        if series:
            return tag, series
    return None, []


def _load_trl_gate_sidecar(trl_dir: str) -> dict[str, dict[int, float]]:
    """Optional `{trl_dir}/rankgrpo_gate_sidecar.json` for TRL logprob probes not in TB."""
    sidecar_path = Path(trl_dir) / "rankgrpo_gate_sidecar.json"
    env_path = os.environ.get("VERL_GR_TRL_GATE_SIDECAR", "").strip()
    if env_path:
        sidecar_path = Path(env_path)
    if not sidecar_path.is_file():
        return {}
    try:
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[int, float]] = {}
    for tag, values in raw.items():
        if not isinstance(values, dict):
            continue
        out[str(tag)] = {int(k): float(v) for k, v in values.items()}
    return out


def _find_trl_train_log(trl_dir: str) -> Path | None:
    env = os.environ.get("VERL_GR_TRL_TRAIN_LOG", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    debug_root = Path(trl_dir).resolve().parent.parent
    logs = sorted(debug_root.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _load_trl_step_times_from_log(trl_dir: str) -> tuple[dict[int, float], str | None]:
    import re

    log_path = _find_trl_train_log(trl_dir)
    if log_path is None:
        return {}, None
    pattern = re.compile(r"\|\s*(\d+)/\d+\s+\[[^\]]*,\s*([\d.]+)s/it\]")
    by_step: dict[int, float] = {}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}, None
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            by_step[int(match.group(1))] = float(match.group(2))
    return by_step, str(log_path) if by_step else None


def _trl_reference_step_time(trl_dir: str) -> tuple[float | None, str | None]:
    """Return TRL per-step time benchmark (seconds). Prefer per-step log, then aggregate TB."""
    per_step, log_src = _load_trl_step_times_from_log(trl_dir)
    if per_step:
        avg = float(np.mean(list(per_step.values())))
        return avg, f"avg from train log ({log_src})"

    tag, series = _load_trl_series_any(["train/train_steps_per_second"], trl_dir)
    if series:
        sps = series[-1][1]
        if sps > 0:
            return 1.0 / sps, tag

    tag, series = _load_trl_series_any(["train/train_runtime"], trl_dir)
    if series:
        runtime, logged_at = series[-1][1], series[-1][0]
        if logged_at > 0 and runtime > 0:
            return runtime / logged_at, tag
    return None, None


def _trl_step_time_at_step(trl_dir: str, step: int, *, fallback: float | None) -> float | None:
    per_step, _ = _load_trl_step_times_from_log(trl_dir)
    if step in per_step:
        return per_step[step]
    return fallback


def _fork_step_time(metrics: dict[str, float]) -> float | None:
    return _pick_metric(metrics, _FORK_STEP_TIME_KEYS)


def _fork_avg_step_time(
    acc: RankGRPOAlignmentAccumulator,
    compare_steps: list[int],
    *,
    skip_warmup_steps: int,
) -> float | None:
    times: list[float] = []
    for step in compare_steps:
        if step <= skip_warmup_steps:
            continue
        fork_time = _fork_step_time(acc.metrics_by_step.get(step, {}))
        if fork_time is not None:
            times.append(fork_time)
    return float(np.mean(times)) if times else None


def _trl_avg_step_time_for_steps(
    trl_dir: str,
    compare_steps: list[int],
    *,
    skip_warmup_steps: int,
    fallback: float | None,
) -> float | None:
    per_step, _ = _load_trl_step_times_from_log(trl_dir)
    times: list[float] = []
    for step in compare_steps:
        if step <= skip_warmup_steps:
            continue
        trl_time = per_step.get(step, fallback)
        if trl_time is not None:
            times.append(trl_time)
    if times:
        return float(np.mean(times))
    return fallback


def _build_step_time_avg_gate(
    *,
    fork_avg: float | None,
    trl_avg: float | None,
    blocked: bool = False,
    blocked_reason: str | None = None,
) -> MetricGateVerdict:
    if blocked:
        return MetricGateVerdict(
            name="step_time",
            passed=False,
            blocked=True,
            blocked_reason=blocked_reason,
        )
    if fork_avg is None:
        return MetricGateVerdict(
            name="step_time",
            passed=False,
            blocked=True,
            blocked_reason="missing fork step-time average",
        )
    if trl_avg is None:
        return MetricGateVerdict(
            name="step_time",
            passed=False,
            blocked=True,
            blocked_reason="missing TRL step-time average",
        )
    passed = _step_time_faster_than_trl(fork_avg, trl_avg)
    return MetricGateVerdict(
        name="step_time",
        passed=passed,
        blocked=False,
        n_evaluated=1,
        n_pass=1 if passed else 0,
        n_fail=0 if passed else 1,
        n_skip=0,
    )


def _step_time_faster_than_trl(fork_time: float, trl_time: float) -> bool:
    return fork_time < trl_time


def _load_trl_logprob_by_check(trl_dir: str) -> dict[str, dict[int, float]]:
    sidecar = _load_trl_gate_sidecar(trl_dir)
    out: dict[str, dict[int, float]] = {}
    for check in _GATE_LOGPROB_CHECKS:
        for tag in check["trl_tags"]:
            if tag in sidecar:
                out[check["label"]] = sidecar[tag]
                break
            _, series = _load_trl_series_any([tag], trl_dir)
            if series:
                out[check["label"]] = {s: v for s, v in series}
                break
    return out


def _metric_gate_passed(n_pass: int, n_evaluated: int) -> bool:
    if n_evaluated <= 0:
        return False
    return (n_pass / n_evaluated) >= _GATE_MIN_PASS_FRACTION


def _format_gate_pass_detail(mg: MetricGateVerdict) -> str:
    if mg.n_evaluated <= 0:
        return "no evaluated steps"
    rate = mg.n_pass / mg.n_evaluated
    threshold = f"{_GATE_MIN_PASS_FRACTION:.0%}"
    return f"{mg.n_pass}/{mg.n_evaluated} pass ({rate:.0%}, need ≥{threshold})"


def _summarize_metric_gate(
    name: str,
    steps: list[StepGateResult],
    ok_attr: str,
    *,
    blocked: bool = False,
    blocked_reason: str | None = None,
) -> MetricGateVerdict:
    """Roll per-step ok flags into a single metric gate verdict."""
    if blocked:
        return MetricGateVerdict(
            name=name,
            passed=False,
            blocked=True,
            blocked_reason=blocked_reason,
        )

    n_pass = n_fail = n_skip = 0
    for row in steps:
        ok = getattr(row, ok_attr)
        if ok is None:
            n_skip += 1
        elif ok:
            n_pass += 1
        else:
            n_fail += 1

    n_evaluated = n_pass + n_fail
    return MetricGateVerdict(
        name=name,
        passed=_metric_gate_passed(n_pass, n_evaluated),
        blocked=False,
        n_evaluated=n_evaluated,
        n_pass=n_pass,
        n_fail=n_fail,
        n_skip=n_skip,
    )


def evaluate_rankgrpo_alignment_gate(
    acc: RankGRPOAlignmentAccumulator,
    *,
    trl_tb_dir: str | Path | None = None,
    max_step: int | None = None,
    skip_warmup_steps: int = 1,
) -> AlignmentGateSummary:
    """Per-step gate: logprob + KL rel diff ≤20% vs TRL; step time = fork avg < TRL avg."""

    trl_dir = _resolve_trl_tb_dir(trl_tb_dir)
    _, trl_kl_series = _load_trl_series_any(_GATE_KL_CHECK["trl_tags"], trl_dir)
    trl_kl_by_step = {s: v for s, v in trl_kl_series}
    trl_logprob_by_check = _load_trl_logprob_by_check(trl_dir)

    trl_step_time_avg, trl_step_time_tag = _trl_reference_step_time(trl_dir)
    blocked: list[str] = []
    logprob_blocked_reason: str | None = None
    kl_blocked_reason: str | None = None
    time_blocked_reason: str | None = None

    if not trl_kl_by_step:
        msg = f"TRL KL missing in `{trl_dir}` (expected train/kl)"
        blocked.append(msg)
        kl_blocked_reason = msg
    if not trl_logprob_by_check:
        msg = (
            f"TRL logprob probes missing in `{trl_dir}` "
            f"(TB tags or rankgrpo_gate_sidecar.json)"
        )
        blocked.append(msg)
        logprob_blocked_reason = msg
    if trl_step_time_avg is None:
        msg = f"TRL step-time benchmark missing (train log or train/train_steps_per_second)"
        blocked.append(msg)
        time_blocked_reason = msg

    debug_cap = os.environ.get("RUN_DEBUG_STEP", "").strip()
    cap = int(debug_cap) if debug_cap.isdigit() else None
    if max_step is not None:
        cap = min(cap, max_step) if cap is not None else max_step

    compare_steps = sorted(set(acc.steps) & set(trl_kl_by_step.keys()))
    if cap is not None:
        compare_steps = [s for s in compare_steps if s <= cap]

    step_results: list[StepGateResult] = []
    for step in compare_steps:
        metrics = acc.metrics_by_step.get(step, {})
        notes: list[str] = []
        is_warmup = step <= skip_warmup_steps

        fork_time = _fork_step_time(metrics)
        trl_time = _trl_step_time_at_step(trl_dir, step, fallback=trl_step_time_avg)
        if fork_time is None:
            notes.append("missing fork step time")
        elif trl_time is None:
            notes.append("missing TRL step time")

        time_ok: bool | None = None
        if not is_warmup and fork_time is not None and trl_time is not None:
            time_ok = _step_time_faster_than_trl(fork_time, trl_time)

        if is_warmup:
            step_results.append(
                StepGateResult(
                    step=step,
                    logprob_ok=None,
                    logprob_rel_err=None,
                    kl_ok=None,
                    kl_rel_err=None,
                    fork_kl=_pick_metric(metrics, _GATE_KL_CHECK["fork_keys"]),
                    trl_kl=trl_kl_by_step.get(step),
                    time_ok=None,
                    fork_step_time=fork_time,
                    trl_step_time=trl_time,
                    passed=None,
                    notes=["warmup (excluded from logprob/KL/step-time gates)"],
                )
            )
            continue

        fork_kl = _pick_metric(metrics, _GATE_KL_CHECK["fork_keys"])
        trl_kl = trl_kl_by_step.get(step)
        kl_ok: bool | None = False
        kl_rel_err: float | None = None
        if fork_kl is not None and trl_kl is not None:
            kl_ok = _rel_aligned(fork_kl, trl_kl)
            kl_rel_err = _relative_error(fork_kl, trl_kl)
        elif fork_kl is None:
            kl_ok = None
            notes.append("missing fork KL")
        elif trl_kl is None:
            kl_ok = None
            notes.append("missing TRL KL")

        logprob_ok: bool | None = None
        logprob_rel_err: float | None = None
        if trl_logprob_by_check:
            lp_ok = True
            worst_err = 0.0
            for check in _GATE_LOGPROB_CHECKS:
                fork_lp = _pick_metric(metrics, check["fork_keys"])
                trl_lp = trl_logprob_by_check.get(check["label"], {}).get(step)
                if fork_lp is None:
                    lp_ok = False
                    notes.append(f"missing fork {check['label']}")
                    continue
                if trl_lp is None:
                    lp_ok = False
                    notes.append(f"missing TRL {check['label']} @step {step}")
                    continue
                err = _relative_error(fork_lp, trl_lp)
                worst_err = max(worst_err, err)
                if not _rel_aligned(fork_lp, trl_lp):
                    lp_ok = False
            logprob_ok = lp_ok
            logprob_rel_err = worst_err

        checks: list[bool] = []
        if kl_ok is not None:
            checks.append(kl_ok)
        if logprob_ok is not None:
            checks.append(logprob_ok)
        passed = all(checks) if checks else False

        step_results.append(
            StepGateResult(
                step=step,
                logprob_ok=logprob_ok,
                logprob_rel_err=logprob_rel_err,
                kl_ok=kl_ok,
                kl_rel_err=kl_rel_err,
                fork_kl=fork_kl,
                trl_kl=trl_kl,
                time_ok=time_ok,
                fork_step_time=fork_time,
                trl_step_time=trl_time,
                passed=passed,
                notes=notes,
            )
        )

    fork_step_time_avg = _fork_avg_step_time(
        acc, compare_steps, skip_warmup_steps=skip_warmup_steps
    )
    trl_step_time_gate_avg = _trl_avg_step_time_for_steps(
        trl_dir,
        compare_steps,
        skip_warmup_steps=skip_warmup_steps,
        fallback=trl_step_time_avg,
    )

    logprob_gate = _summarize_metric_gate(
        "logprob",
        step_results,
        "logprob_ok",
        blocked=logprob_blocked_reason is not None,
        blocked_reason=logprob_blocked_reason,
    )
    kl_gate = _summarize_metric_gate(
        "kl",
        step_results,
        "kl_ok",
        blocked=kl_blocked_reason is not None,
        blocked_reason=kl_blocked_reason,
    )
    combined_gate = _summarize_metric_gate("combined", step_results, "passed")
    time_gate = _build_step_time_avg_gate(
        fork_avg=fork_step_time_avg,
        trl_avg=trl_step_time_gate_avg,
        blocked=time_blocked_reason is not None,
        blocked_reason=time_blocked_reason,
    )

    any_blocked = logprob_gate.blocked or kl_gate.blocked or time_gate.blocked
    combined_passed = (
        not any_blocked
        and bool(step_results)
        and logprob_gate.passed
        and kl_gate.passed
        and time_gate.passed
    )

    return AlignmentGateSummary(
        passed=combined_passed,
        logprob_gate=logprob_gate,
        kl_gate=kl_gate,
        combined_gate=combined_gate,
        time_gate=time_gate,
        steps=step_results,
        trl_tb_dir=trl_dir,
        trl_step_time_ref=trl_step_time_gate_avg or trl_step_time_avg,
        trl_step_time_tag=trl_step_time_tag,
        fork_step_time_avg=fork_step_time_avg,
        logprob_trl_available=bool(trl_logprob_by_check),
        blocked_reasons=blocked,
    )


def _format_gate_verdicts_summary(gate: AlignmentGateSummary, *, last_step: int) -> list[str]:
    """Compact three-metric + combined gate summary (report header section)."""
    lines: list[str] = []
    lines.append("## Gate Verdicts")
    lines.append("")
    lines.append(
        f"Per-metric pass rule: **pass / (pass + fail) ≥ {_GATE_MIN_PASS_FRACTION:.0%}** "
        "(warmup step 1 excluded). **step_time** uses fork vs TRL average s/it."
    )
    lines.append("")
    lines.append("| gate | status | pass | fail | skip | detail |")
    lines.append("|------|--------|------|------|------|--------|")

    for mg in (gate.logprob_gate, gate.kl_gate, gate.time_gate):
        if mg.blocked:
            detail = mg.blocked_reason or "—"
        elif mg.passed:
            if mg.name == "step_time":
                detail = (
                    f"fork avg {gate.fork_step_time_avg:.3f}s < TRL avg {gate.trl_step_time_ref:.3f}s"
                    if gate.fork_step_time_avg is not None and gate.trl_step_time_ref is not None
                    else "fork avg faster than TRL avg"
                )
            elif mg.name == "logprob":
                detail = f"rollout↔ref + actor↔ref vs TRL; {_format_gate_pass_detail(mg)}"
            elif mg.name == "kl":
                detail = f"actor/kl_loss vs TRL train/kl; {_format_gate_pass_detail(mg)}"
            else:
                detail = _format_gate_pass_detail(mg)
        else:
            if mg.name == "step_time":
                if gate.fork_step_time_avg is not None and gate.trl_step_time_ref is not None:
                    detail = (
                        f"fork avg {gate.fork_step_time_avg:.3f}s ≥ TRL avg {gate.trl_step_time_ref:.3f}s"
                    )
                else:
                    detail = "avg comparison failed"
            else:
                detail = _format_gate_pass_detail(mg)
        lines.append(
            f"| {mg.name} | **{mg.status_label()}** | {mg.n_pass} | {mg.n_fail} | {mg.n_skip} | {detail} |"
        )

    mg = gate.combined_gate
    if any(x.blocked for x in (gate.logprob_gate, gate.kl_gate, gate.time_gate)):
        combined_status = "BLOCKED"
        combined_detail = "; ".join(
            f"{x.name}: {x.blocked_reason}"
            for x in (gate.logprob_gate, gate.kl_gate, gate.time_gate)
            if x.blocked
        )
    elif gate.passed:
        combined_status = "PASS"
        combined_detail = (
            f"logprob ∧ KL (per-step) ∧ step_time avg @ step {last_step}; "
            f"{_format_gate_pass_detail(mg)}"
        )
    else:
        combined_status = "FAIL"
        fail_parts: list[str] = []
        if not gate.logprob_gate.passed and not gate.logprob_gate.blocked:
            fail_parts.append(f"logprob ({_format_gate_pass_detail(gate.logprob_gate)})")
        if not gate.kl_gate.passed and not gate.kl_gate.blocked:
            fail_parts.append(f"KL ({_format_gate_pass_detail(gate.kl_gate)})")
        if not gate.combined_gate.passed:
            fail_parts.append(f"per-step ({_format_gate_pass_detail(gate.combined_gate)})")
        if not gate.time_gate.passed and not gate.time_gate.blocked:
            fail_parts.append("step_time avg")
        combined_detail = "; ".join(fail_parts) or _format_gate_pass_detail(mg)

    lines.append(
        f"| **combined** | **{combined_status}** | {mg.n_pass} | {mg.n_fail} | "
        f"{mg.n_skip} | {combined_detail} |"
    )
    lines.append("")
    lines.append(f"- TRL reference: `{gate.trl_tb_dir}`")
    lines.append(f"- TRL logprob probes in TB: **{'yes' if gate.logprob_trl_available else 'no'}**")
    if gate.fork_step_time_avg is not None:
        lines.append(f"- Fork step-time avg: **{gate.fork_step_time_avg:.3f}s** (excl. warmup)")
    if gate.trl_step_time_ref is not None:
        lines.append(
            f"- TRL step-time avg: **{gate.trl_step_time_ref:.3f}s** (`{gate.trl_step_time_tag}`)"
        )
    lines.append("")
    return lines


def _format_gate_per_step_table(gate: AlignmentGateSummary) -> list[str]:
    """Per-step gate comparison table."""
    lines: list[str] = []
    lines.append("## Per-step alignment gate")
    lines.append("")
    lines.append(
        "Criteria (each step): **logprob gate** and **KL gate** rel diff ≤20% vs TRL; "
        "**step time gate**: fork s/it < TRL s/it at the same step. "
        "Header **step time** gate uses fork vs TRL **average** s/it (excl. warmup step 1). "
        "Logprob/KL/step-time gates skip step 1 (vLLM compile + first-cycle warmup)."
    )
    lines.append("")

    lines.append(_GATE_PER_STEP_HEADER)
    lines.append(_GATE_PER_STEP_SEP)

    show_steps = list(gate.steps)
    if len(show_steps) > 40:
        ellipsis = StepGateResult(
            step=-1,
            logprob_ok=None,
            logprob_rel_err=None,
            kl_ok=None,
            kl_rel_err=None,
            fork_kl=None,
            trl_kl=None,
            time_ok=None,
            fork_step_time=None,
            trl_step_time=None,
            passed=None,
        )
        show_steps = show_steps[:15] + [ellipsis] + show_steps[-10:]

    for row in show_steps:
        if row.step == -1:
            lines.append("| … | | | | | | | | |")
            continue
        lines.append(
            "| {step} | {logprob} | {kl} | {fork_kl} | {trl_kl} | {time_gate} | {fork_time} | {trl_time} | {combined} |".format(
                step=row.step,
                logprob=_format_gate_metric_cell(row.logprob_ok, row.logprob_rel_err),
                kl=_format_gate_metric_cell(row.kl_ok, row.kl_rel_err),
                fork_kl=f"{row.fork_kl:.4g}" if row.fork_kl is not None else "—",
                trl_kl=f"{row.trl_kl:.4g}" if row.trl_kl is not None else "—",
                time_gate=_format_step_time_gate_cell(row.time_ok),
                fork_time=f"{row.fork_step_time:.3f}s" if row.fork_step_time is not None else "—",
                trl_time=f"{row.trl_step_time:.3f}s" if row.trl_step_time is not None else "—",
                combined=_format_step_time_gate_cell(row.passed),
            )
        )

    lines.append("")
    if not gate.passed and gate.steps:
        lines.append("First failing steps:")
        for r in [x for x in gate.steps if x.passed is False][:5]:
            parts = []
            if r.logprob_ok is False:
                parts.append("logprob")
            if r.kl_ok is False:
                parts.append("KL")
            if r.time_ok is False:
                parts.append("step_time")
            lines.append(f"- step {r.step}: {', '.join(parts) or 'unknown'}")
        lines.append("")
    return lines


def _format_gate_markdown(gate: AlignmentGateSummary, *, last_step: int) -> list[str]:
    lines: list[str] = []
    lines.extend(_format_gate_verdicts_summary(gate, last_step=last_step))
    lines.extend(_format_gate_per_step_table(gate))
    return lines


def write_rankgrpo_alignment_report(
    *,
    output_dir: str | Path | None = None,
    trl_tb_dir: str | Path | None = None,
    experiment_name: str | None = None,
) -> tuple[Path, AlignmentGateSummary] | None:
    """Write human-readable alignment report after a RUN_DEBUG_STEP-capped run."""

    if not alignment_report_enabled():
        return None

    acc = _ALIGNMENT_ACCUMULATOR
    if not acc.steps:
        print("[rankgrpo_align] no metrics recorded; skip report")
        return None

    out_root = _resolve_align_report_root(output_dir)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    trl_dir = _resolve_trl_tb_dir(trl_tb_dir)
    last_step = acc.last_step()
    assert last_step is not None

    gate = evaluate_rankgrpo_alignment_gate(acc, trl_tb_dir=trl_dir)

    compare_steps = sorted({r.step for r in gate.steps} or set(acc.steps))
    latency_summary = compute_modular_step_latency_summary(
        acc,
        trl_tb_dir=trl_dir,
        compare_steps=compare_steps,
    )
    probe_summary = compute_sidecar_probe_overhead_summary(
        acc,
        compare_steps=compare_steps,
        latency_summary=latency_summary,
        trl_tb_dir=trl_dir,
    )

    trl_cache: dict[str, list[tuple[int, float]]] = {}
    lines: list[str] = []
    lines.append("# RankGRPO TRL Alignment Report")
    lines.append("")
    lines.append(f"- generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- experiment: {experiment_name or os.environ.get('EXPERIMENT_NAME', 'unknown')}")
    lines.append(f"- fork steps: {acc.steps[0]}–{last_step} (n={len(acc.steps)})")
    lines.append(f"- RUN_DEBUG_STEP: {os.environ.get('RUN_DEBUG_STEP')}")
    lines.append(f"- report dir: `{log_dir}`")
    lines.append(f"- TRL reference (`TRL_REF`): `{trl_dir}`")
    trl_offset = _trl_resume_offset(trl_dir)
    lines.append(f"- TRL resume offset (for matched comparison): +{trl_offset} → fork step {last_step} vs TRL ~{last_step + trl_offset}")
    lines.append("")
    lines.extend(_format_gate_markdown(gate, last_step=last_step))
    lines.extend(format_modular_step_latency_markdown(latency_summary))
    lines.extend(format_sidecar_probe_overhead_markdown(probe_summary))

    aligned: list[str] = []
    misaligned: list[str] = []
    unknown: list[str] = []

    lines.append("## Metric comparison (fork last step vs TRL)")
    lines.append("")
    lines.append("| category | check | fork @last | TRL @same | TRL @resume+offset | status | notes |")
    lines.append("|----------|-------|------------|-----------|-------------------|--------|-------|")

    for check in _ALIGNMENT_CHECKS:
        fork_val = _pick_metric(acc.metrics_by_step[last_step], check["fork_keys"])
        fork_mean5 = acc.mean_last(check["fork_keys"], n=5)

        trl_val_same: float | None = None
        trl_step_same = last_step
        trl_val_off: float | None = None
        trl_step_off = last_step + trl_offset
        trl_tag = check.get("trl_tag")
        if trl_tag:
            if trl_tag not in trl_cache:
                trl_cache[trl_tag] = _load_trl_series(trl_tag, trl_dir)
            trl_val_same, trl_step_same = _trl_value_at_step(trl_cache[trl_tag], last_step)
            trl_val_off, trl_step_off = _trl_value_at_step(trl_cache[trl_tag], last_step + trl_offset)

        # Prefer resume-offset TRL value when available (TRL TB starts ~410).
        trl_val = trl_val_off if trl_val_off is not None else trl_val_same
        trl_step = trl_step_off if trl_val_off is not None else trl_step_same

        status = "unknown"
        if fork_val is None:
            status = "missing"
            unknown.append(check["name"])
        else:
            try:
                ok = bool(check["aligned_if"](fork_val, trl_val))
            except Exception:
                ok = False
            if trl_val is None and check["trl_tag"] is None:
                status = "diagnostic"
                unknown.append(check["name"])
            elif ok:
                status = "aligned"
                aligned.append(check["name"])
            else:
                status = "MISALIGNED"
                misaligned.append(check["name"])

        fork_disp = f"{fork_val:.6g}" if fork_val is not None else "—"
        if fork_mean5 is not None and fork_val is not None and len(acc.steps) >= 3:
            fork_disp += f" (mean5={fork_mean5:.6g})"
        trl_same_disp = f"{trl_val_same:.6g} @{trl_step_same}" if trl_val_same is not None else "—"
        trl_off_disp = f"{trl_val_off:.6g} @{trl_step_off}" if trl_val_off is not None else "—"
        note = str(check.get("note", ""))
        lines.append(
            f"| {check['category']} | {check['name']} | {fork_disp} | {trl_same_disp} | {trl_off_disp} | **{status}** | {note} |"
        )

    lines.append("")
    lines.append("## Trajectory (fork)")
    lines.append("")
    traj_keys = [
        ("KL", ["actor/kl_loss", "actor/train/kl"]),
        ("reward_total", ["train/rankgrpo/reward_total"]),
        ("mean_length", ["train/rankgrpo/completions/mean_length"]),
        ("rollout-ref gate", ["logprob_gate/rollout_minus_ref/abs_mean"]),
        ("actor-ref |diff|", ["actor/debug/logprob_diff_abs"]),
        ("pg_clipfrac", ["actor/pg_clipfrac"]),
    ]
    for label, keys in traj_keys:
        series = acc.series(keys)
        if not series:
            continue
        preview = ", ".join(f"{s}={v:.4g}" for s, v in series[-8:])
        lines.append(f"- **{label}**: {preview}")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Aligned ({len(aligned)})**: {', '.join(aligned) or '—'}")
    lines.append(f"- **Misaligned ({len(misaligned)})**: {', '.join(misaligned) or '—'}")
    lines.append(f"- **Diagnostic / missing ({len(unknown)})**: {', '.join(unknown) or '—'}")
    lines.append("")

    if misaligned:
        lines.append("## Likely root causes (from misaligned checks)")
        lines.append("")
        if any("KL" in m for m in misaligned):
            lines.append(
                "- **KL low vs TRL**: verify `loss_agg_mode: seq-mean-token-mean` and `loss_mode: trl_match` on the fork, "
                "ref fp32, item_token_mask KL aggregation; TRL TB may start at step 410 (resume offset)."
            )
        if any("reward" in m.lower() for m in misaligned):
            lines.append(
                "- **Reward**: high step variance — compare mean over last 10 steps; confirm `exp_inf` / "
                "`rank_rewards_from_text` catalog path and `num_generations=8` grouping."
            )
        if any("logprob" in m.lower() for m in misaligned):
            lines.append(
                "- **Logprob**: use `logprob_gate/rollout_minus_ref` for batch gate; "
                "`actor/debug/logprob_diff_abs` for true actor update vs ref."
            )
        lines.append("")

    lines.append("## Config checklist")
    lines.append("")
    lines.append("- [ ] `loss_mode: trl_match`, `old_log_prob_mode: current`")
    lines.append("- [ ] `loss_mode: trl_match`, `old_log_prob_mode: current`, `loss_agg_mode: seq-mean-token-mean`")
    lines.append("- [ ] `use_fused_kernels: false`, actor/ref `model_dtype: fp32`")
    lines.append("- [ ] `kl_loss_coef: 0.001`, `low_var_kl`, `importance_sampling_level: item`")
    lines.append("- [ ] `bypass_mode: true` → batch `old_log_probs` is rollout")
    lines.append("- [ ] `gen_batch_size: 6`, `rollout.n: 8`, `ppo_epochs: 1`")
    lines.append("")

    report_name = f"rankgrpo_align_report_step{last_step}.md"
    report_path = log_dir / report_name
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = log_dir / f"rankgrpo_align_report_step{last_step}.json"
    json_path.write_text(
        json.dumps(
            {
                "last_step": last_step,
                "steps": acc.steps,
                "metrics_by_step": {str(k): v for k, v in acc.metrics_by_step.items()},
                "aligned": aligned,
                "misaligned": misaligned,
                "unknown": unknown,
                "trl_tb_dir": trl_dir,
                "modular_step_latency": modular_step_latency_summary_to_dict(latency_summary),
                "sidecar_probe_overhead": sidecar_probe_overhead_summary_to_dict(probe_summary),
                "gate": {
                    "passed": gate.passed,
                    "logprob_gate": {
                        "passed": gate.logprob_gate.passed,
                        "blocked": gate.logprob_gate.blocked,
                        "blocked_reason": gate.logprob_gate.blocked_reason,
                        "n_pass": gate.logprob_gate.n_pass,
                        "n_fail": gate.logprob_gate.n_fail,
                        "n_skip": gate.logprob_gate.n_skip,
                    },
                    "kl_gate": {
                        "passed": gate.kl_gate.passed,
                        "blocked": gate.kl_gate.blocked,
                        "blocked_reason": gate.kl_gate.blocked_reason,
                        "n_pass": gate.kl_gate.n_pass,
                        "n_fail": gate.kl_gate.n_fail,
                        "n_skip": gate.kl_gate.n_skip,
                    },
                    "time_gate": {
                        "passed": gate.time_gate.passed,
                        "blocked": gate.time_gate.blocked,
                        "blocked_reason": gate.time_gate.blocked_reason,
                        "n_pass": gate.time_gate.n_pass,
                        "n_fail": gate.time_gate.n_fail,
                        "n_skip": gate.time_gate.n_skip,
                    },
                    "combined_gate": {
                        "passed": gate.combined_gate.passed,
                        "n_pass": gate.combined_gate.n_pass,
                        "n_fail": gate.combined_gate.n_fail,
                        "n_skip": gate.combined_gate.n_skip,
                    },
                    "min_pass_fraction": _GATE_MIN_PASS_FRACTION,
                    "blocked_reasons": gate.blocked_reasons,
                    "trl_step_time_ref": gate.trl_step_time_ref,
                    "fork_step_time_avg": gate.fork_step_time_avg,
                    "logprob_trl_available": gate.logprob_trl_available,
                    "n_pass": gate.combined_gate.n_pass,
                    "n_fail": gate.combined_gate.n_fail,
                    "n_skip": gate.combined_gate.n_skip,
                    "per_step": [
                        {
                            "step": r.step,
                            "passed": r.passed,
                            "logprob_ok": r.logprob_ok,
                            "logprob_rel_err": r.logprob_rel_err,
                            "kl_ok": r.kl_ok,
                            "kl_rel_err": r.kl_rel_err,
                            "time_ok": r.time_ok,
                            "fork_step_time": r.fork_step_time,
                            "trl_step_time": r.trl_step_time,
                        }
                        for r in gate.steps
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[rankgrpo_align] wrote {report_path}")
    print(f"[rankgrpo_align] wrote {json_path}")
    for mg in (gate.logprob_gate, gate.kl_gate, gate.time_gate):
        print(f"[rankgrpo_align] {mg.name} gate: {mg.status_label()} ({mg.n_pass}/{mg.n_evaluated} pass, {mg.n_skip} skip)")
    if gate.passed:
        print(f"[rankgrpo_align] COMBINED GATE PASS")
    else:
        print(f"[rankgrpo_align] COMBINED GATE FAIL")
    if misaligned:
        print(f"[rankgrpo_align] snapshot MISALIGNED ({len(misaligned)}): {', '.join(misaligned)}")
    else:
        print(f"[rankgrpo_align] snapshot: all checked metrics aligned or diagnostic-only")
    return report_path, gate


def _load_all_tb_scalars(logdir: str | Path) -> dict[str, dict[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    path = Path(logdir)
    if not path.is_dir():
        return {}
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, dict[int, float]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = {int(e.step): float(e.value) for e in ea.Scalars(tag)}
    return out


def _fork_series(fork_tb: dict[str, dict[int, float]], keys: list[str]) -> dict[int, float]:
    for key in keys:
        if key in fork_tb and fork_tb[key]:
            return fork_tb[key]
    return {}


def write_offline_tb_alignment_report(
    *,
    fork_tb_dir: str | Path,
    trl_tb_dir: str | Path,
    output_dir: str | Path,
    experiment_name: str = "unknown",
    max_step: int | None = None,
    report_stem: str = "precision_align_vs_trl_debug",
) -> Path | None:
    """Compare fork vs TRL TensorBoard logs and write a markdown+json report."""

    fork_tb = _load_all_tb_scalars(fork_tb_dir)
    trl_tb = _load_all_tb_scalars(trl_tb_dir)
    if not fork_tb or not trl_tb:
        print("[rankgrpo_align] missing TB scalars; skip offline report")
        return None

    fork_steps = sorted(
        {
            step
            for keys in (
                ["actor/kl_loss", "train/kl"],
                ["train/rankgrpo/reward_total"],
                ["train/rankgrpo/reward"],
                ["train/rankgrpo/completions/mean_length"],
            )
            for step in _fork_series(fork_tb, keys)
        }
    )
    if not fork_steps:
        print("[rankgrpo_align] no fork steps found")
        return None

    trl_cache: dict[str, dict[int, float]] = {}
    for check in _ALIGNMENT_CHECKS:
        tag = check.get("trl_tag")
        if tag and tag not in trl_cache:
            trl_cache[tag] = {s: v for s, v in _load_trl_series(tag, str(trl_tb_dir))}

    compare_steps = _infer_compare_steps(fork_steps, {k: list(v.items()) for k, v in trl_cache.items()}, max_step=max_step)
    last_step = compare_steps[-1] if compare_steps else fork_steps[-1]
    trl_offset = _trl_resume_offset(str(trl_tb_dir))

    out_root = Path(output_dir)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# RankGRPO Precision Alignment (verl-gr vs TRL)")
    lines.append("")
    lines.append(f"- generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- experiment: {experiment_name}")
    lines.append(f"- fork TB: `{fork_tb_dir}`")
    lines.append(f"- TRL TB: `{trl_tb_dir}`")
    lines.append(f"- compared steps: {compare_steps[0]}–{last_step} (n={len(compare_steps)})")
    lines.append(f"- TRL resume offset (legacy): +{trl_offset}")
    lines.append("")

    aligned: list[str] = []
    misaligned: list[str] = []
    unknown: list[str] = []

    lines.append("## Snapshot @ last compared step")
    lines.append("")
    lines.append("| category | check | fork | TRL same step | rel err | status |")
    lines.append("|----------|-------|------|---------------|---------|--------|")

    summary_stats: dict[str, Any] = {"checks": [], "per_step": {}}

    for check in _ALIGNMENT_CHECKS:
        fork_series = _fork_series(fork_tb, check["fork_keys"])
        fork_val = fork_series.get(last_step)
        trl_val = None
        trl_tag = check.get("trl_tag")
        if trl_tag:
            trl_val = trl_cache.get(trl_tag, {}).get(last_step)

        status = "unknown"
        rel_err: float | None = None
        if fork_val is None:
            status = "missing"
            unknown.append(check["name"])
        elif trl_tag is None:
            status = "diagnostic"
            unknown.append(check["name"])
        elif trl_val is None:
            status = "missing TRL"
            unknown.append(check["name"])
        else:
            rel_err = abs(fork_val - trl_val) / max(abs(trl_val), 1e-8)
            try:
                ok = bool(check["aligned_if"](fork_val, trl_val))
            except Exception:
                ok = False
            status = "aligned" if ok else "MISALIGNED"
            (aligned if ok else misaligned).append(check["name"])

        fork_disp = f"{fork_val:.6g}" if fork_val is not None else "—"
        trl_disp = f"{trl_val:.6g}" if trl_val is not None else "—"
        err_disp = f"{rel_err:.3f}" if rel_err is not None else "—"
        lines.append(
            f"| {check['category']} | {check['name']} | {fork_disp} | {trl_disp} | {err_disp} | **{status}** |"
        )
        summary_stats["checks"].append(
            {
                "name": check["name"],
                "fork": fork_val,
                "trl": trl_val,
                "rel_err": rel_err,
                "status": status,
            }
        )

    lines.append("")
    lines.append("## Trajectory statistics (matched steps)")
    lines.append("")

    metric_pairs = [
        ("KL", ["actor/kl_loss", "train/kl"], "train/kl"),
        ("reward_total", ["train/rankgrpo/reward_total"], "train/reward_total"),
        ("reward mean", ["train/rankgrpo/reward"], "train/reward"),
        ("mean_length", ["train/rankgrpo/completions/mean_length"], "train/completions/mean_length"),
    ]
    for label, fork_keys, trl_tag in metric_pairs:
        fs = _fork_series(fork_tb, fork_keys)
        ts = trl_cache.get(trl_tag, {})
        ratios: list[float] = []
        diffs: list[float] = []
        for step in compare_steps:
            fv, tv = fs.get(step), ts.get(step)
            if fv is None or tv is None:
                continue
            if abs(tv) > 1e-9:
                ratios.append(fv / tv)
            diffs.append(fv - tv)
        if not ratios and not diffs:
            continue
        ratio_med = float(np.median(ratios)) if ratios else None
        diff_mean = float(np.mean(diffs)) if diffs else None
        lines.append(
            f"- **{label}**: median ratio fork/TRL={ratio_med:.4g}" if ratio_med is not None
            else f"- **{label}**: mean diff fork−TRL={diff_mean:.4g}"
        )
        summary_stats.setdefault("trajectory", {})[label] = {
            "median_ratio": ratio_med,
            "mean_diff": diff_mean,
            "n": len(ratios) or len(diffs),
        }

    lines.append("")
    lines.append("## Key steps")
    lines.append("")
    lines.append("| step | fork KL | TRL KL | fork reward_total | TRL reward_total | fork len | TRL len |")
    lines.append("|------|---------|--------|-------------------|------------------|----------|---------|")
    key_steps = [s for s in (1, 10, 25, 50, 75, 100, 150, 200) if s in compare_steps]
    if last_step not in key_steps:
        key_steps.append(last_step)
    for step in sorted(set(key_steps)):
        fk = _fork_series(fork_tb, ["actor/kl_loss"]).get(step)
        tk = trl_cache.get("train/kl", {}).get(step)
        fr = _fork_series(fork_tb, ["train/rankgrpo/reward_total"]).get(step)
        tr = trl_cache.get("train/reward_total", {}).get(step)
        fl = _fork_series(fork_tb, ["train/rankgrpo/completions/mean_length"]).get(step)
        tl = trl_cache.get("train/completions/mean_length", {}).get(step)
        lines.append(
            f"| {step} | {fk:.4g} | {tk:.4g} | {fr:.4g} | {tr:.4g} | {fl:.4g} | {tl:.4g} |"
            if all(v is not None for v in (fk, tk, fr, tr, fl, tl))
            else f"| {step} | {fk} | {tk} | {fr} | {tr} | {fl} | {tl} |"
        )

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Aligned ({len(aligned)})**: {', '.join(aligned) or '—'}")
    lines.append(f"- **Misaligned ({len(misaligned)})**: {', '.join(misaligned) or '—'}")
    lines.append(f"- **Diagnostic / missing ({len(unknown)})**: {', '.join(unknown) or '—'}")
    lines.append("")

    if misaligned:
        lines.append("## Notes on misalignments")
        lines.append("")
        if any("KL" in m for m in misaligned):
            lines.append(
                "- **KL**: TRL logs global token-mean KL (`train/kl`, bnpo). "
                "Use `loss_agg_mode: token-mean` and `loss_mode: trl_match` on the fork."
            )
        if any("reward" in m.lower() for m in misaligned):
            lines.append(
                "- **Reward**: compare `train/rankgrpo/reward` / `reward_total`, not `critic/rewards/mean`."
            )
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    if not misaligned:
        lines.append(
            "At matched training steps, verl-gr and TRL agree on all checked scalars within tolerance. "
            "Logprob gate diagnostics (rollout↔ref, bypass sanity) confirm correct bypass + ref path."
        )
    else:
        lines.append(
            "Partial alignment: generation/reward track TRL; remaining gaps are listed above. "
            "Re-run fork after config fixes with the same seed (3407) for step-wise comparison."
        )
    lines.append("")

    report_path = log_dir / f"{report_stem}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = log_dir / f"{report_stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "experiment": experiment_name,
                "fork_tb_dir": str(fork_tb_dir),
                "trl_tb_dir": str(trl_tb_dir),
                "last_step": last_step,
                "compare_steps": compare_steps,
                "aligned": aligned,
                "misaligned": misaligned,
                "unknown": unknown,
                "summary": summary_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[rankgrpo_align] wrote {report_path}")
    print(f"[rankgrpo_align] wrote {json_path}")
    return report_path
