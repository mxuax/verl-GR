"""Long-horizon convergence gates vs TRL reference (KL + eval/reward).

Two layers:

1. **Exit report** (``write_convergence_gate_report``): offline TB compare @200/400/600 (+ optional steps).
   Zero training-step cost; runs once when ``fit()`` finishes.

2. **KL growth early-stop** (``maybe_abort_on_kl_growth_failure``): during long runs,
   abort if fork KL stays far below TRL (policy not moving) or eval lags TRL badly.

3. **Length blowout early-stop** (``maybe_abort_on_length_blowout``): abort when
   ``eos_rate`` collapses, ``clip_ratio`` spikes, or ``overflow_token_ratio`` grows.
   Disable with ``VERL_GR_LENGTH_GATE=0``.

   Cost: O(1) dict lookup per logging step; optional TB read of TRL ref once per gate step.

This is **not** the step-30 sidecar / ``VERL_GR_ALIGN_DEBUG`` path (replay, fingerprint,
logprob export) — that path is debug-only and expensive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONVERGENCE_CHECK_STEPS: tuple[int, ...] = (200, 400, 600)

# Minimum fork_kl / trl_kl at TRL optimizer steps. Failed trlonpolicy was ~0.02→0.002.
KL_GROWTH_RATIO_FLOORS: dict[int, float] = {
    200: 0.05,
    600: 0.05,
    2000: 0.05,
}

# Absolute KL floors (independent of TRL TB). Failed trlonpolicy stayed ~2e-4.
# Prod long_jul8_3 was ~1.8e-3 @200 and ~2.0e-2 @600.
KL_ABS_FLOORS: dict[int, float] = {
    200: 0.001,
    600: 0.005,
    2000: 0.01,
}

# eval/reward_total must not lag TRL by more than this (when eval is in the log batch).
EVAL_MAX_LAG_VS_TRL: dict[int, float] = {
    200: 0.05,
    600: 0.05,
    2000: 0.08,
}

# f167a49 workingbranch logs actor/kl_loss and actor/dbg/kl_tok_mean (not actor/train/kl).
_FORK_KL_KEYS = (
    "actor/dbg/kl_tok_mean",
    "actor/kl_loss",
    "actor/train/kl",
    "train/kl",
)
_FORK_EVAL_KEYS = ("eval/reward_total",)
_FORK_ROLLOUT_GAP_KEYS = (
    "actor/rollout_train_logprob_gap",
    "logprob_gate/actor_minus_rollout/abs_mean",
    "actor/debug/logprob_diff_abs",
)
_FORK_EOS_RATE_KEYS = ("train/rankgrpo/items/eos_rate",)
_FORK_CLIP_RATIO_KEYS = (
    "response_length/clip_ratio",
    "train/rankgrpo/completions/clipped_ratio",
)
_FORK_OVERFLOW_RATIO_KEYS = ("train/rankgrpo/items/overflow_token_ratio",)
_TRL_KL_TAG = "train/kl"
_TRL_EVAL_TAG = "eval/reward_total"
_TRL_KL_CACHE: dict[int, float] | None = None
_TRL_EVAL_CACHE: dict[int, float] | None = None
_FORK_KL_AT_GATE: dict[int, float] = {}
_ABORT_REPORT_WRITTEN = False


@dataclass
class ConvergenceStepResult:
    step: int
    fork_kl: float | None = None
    trl_kl: float | None = None
    kl_rel_err: float | None = None
    kl_ok: bool | None = None
    fork_eval: float | None = None
    trl_eval: float | None = None
    eval_delta: float | None = None
    eval_ok: bool | None = None
    eval_beats_trl: bool | None = None
    fork_rollout_gap: float | None = None


@dataclass
class ConvergenceGateSummary:
    passed: bool
    steps: list[ConvergenceStepResult] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    fork_tb_dir: str = ""
    trl_tb_dir: str = ""


def _convergence_check_steps() -> tuple[int, ...]:
    raw = os.environ.get("VERL_GR_CONVERGENCE_STEPS", "").strip()
    if not raw:
        return CONVERGENCE_CHECK_STEPS
    steps: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            steps.append(int(part))
        except ValueError:
            continue
    return tuple(sorted(set(steps))) or CONVERGENCE_CHECK_STEPS


def _rel_aligned(fork_val: float, trl_val: float, *, tol: float = 0.20) -> bool:
    denom = max(abs(trl_val), 1e-8)
    return abs(fork_val - trl_val) / denom <= tol


def _load_tb_scalar_series(tb_dir: str | Path, tag: str) -> dict[int, float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    path = Path(tb_dir)
    if not path.is_dir():
        return {}
    files = list(path.glob("events.out.tfevents.*"))
    if not files:
        files = list(path.glob("**/events.out.tfevents.*"))
    if not files:
        return {}
    main = max(files, key=lambda p: p.stat().st_size)
    ea = EventAccumulator(str(main))
    try:
        ea.Reload()
    except Exception:
        return {}
    if tag not in ea.Tags().get("scalars", []):
        return {}
    return {int(e.step): float(e.value) for e in ea.Scalars(tag)}


def _pick_series(series_map: dict[int, float], keys: tuple[str, ...], tb_dir: str | Path) -> dict[int, float]:
    for key in keys:
        loaded = _load_tb_scalar_series(tb_dir, key)
        if loaded:
            return loaded
    return series_map


def _resolve_trl_tb_dir(explicit: str | Path | None = None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    for env_name in ("VERL_GR_TRL_TB_REF", "TRL_REF"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            return raw
    return ""


def _resolve_fork_tb_dir(experiment_name: str | None = None) -> str:
    explicit = os.environ.get("VERL_GR_FORK_TB_DIR", "").strip()
    if explicit and Path(explicit).is_dir():
        return explicit

    exp = experiment_name or os.environ.get("EXPERIMENT_NAME", "")
    output_dir = os.environ.get("OUTPUT_DIR", "").strip()
    if output_dir:
        tb_under_output = Path(output_dir) / "tensorboard"
        if tb_under_output.is_dir():
            return str(tb_under_output)

    root = os.environ.get("VERL_GR_ROOT", "")
    if root and exp:
        legacy = Path(root) / "tensorboard_log" / "RankGRPO" / exp
        if legacy.is_dir():
            return str(legacy)
        return str(legacy)
    return ""


def evaluate_convergence_gate(
    *,
    fork_tb_dir: str | Path | None = None,
    trl_tb_dir: str | Path | None = None,
    steps: tuple[int, ...] | None = None,
    eval_abs_tol: float = 0.02,
) -> ConvergenceGateSummary:
    check_steps = steps or _convergence_check_steps()
    fork_dir = str(fork_tb_dir or _resolve_fork_tb_dir())
    trl_dir = str(trl_tb_dir or _resolve_trl_tb_dir())
    blocked: list[str] = []
    if not fork_dir or not Path(fork_dir).is_dir():
        blocked.append(f"fork TB missing: {fork_dir or '(unset)'}")
    if not trl_dir or not Path(trl_dir).is_dir():
        blocked.append(f"TRL TB missing: {trl_dir or '(unset)'}")

    fork_kl = _pick_series({}, _FORK_KL_KEYS, fork_dir) if fork_dir else {}
    fork_eval = _pick_series({}, _FORK_EVAL_KEYS, fork_dir) if fork_dir else {}
    fork_gap = _pick_series({}, _FORK_ROLLOUT_GAP_KEYS, fork_dir) if fork_dir else {}
    trl_kl = _load_tb_scalar_series(trl_dir, _TRL_KL_TAG) if trl_dir else {}
    trl_eval = _load_tb_scalar_series(trl_dir, _TRL_EVAL_TAG) if trl_dir else {}

    if not trl_kl:
        blocked.append(f"TRL `{_TRL_KL_TAG}` missing in `{trl_dir}`")
    if not trl_eval:
        blocked.append(f"TRL `{_TRL_EVAL_TAG}` missing in `{trl_dir}`")

    results: list[ConvergenceStepResult] = []
    all_ok = not blocked
    for step in check_steps:
        row = ConvergenceStepResult(step=step)
        row.fork_kl = fork_kl.get(step)
        row.trl_kl = trl_kl.get(step)
        row.fork_eval = fork_eval.get(step)
        row.trl_eval = trl_eval.get(step)
        row.fork_rollout_gap = fork_gap.get(step)

        if row.fork_kl is not None and row.trl_kl is not None:
            row.kl_rel_err = abs(row.fork_kl - row.trl_kl) / max(abs(row.trl_kl), 1e-8)
            row.kl_ok = _rel_aligned(row.fork_kl, row.trl_kl)
            all_ok = all_ok and bool(row.kl_ok)
        else:
            all_ok = False

        if row.fork_eval is not None and row.trl_eval is not None:
            row.eval_delta = row.fork_eval - row.trl_eval
            row.eval_beats_trl = row.fork_eval > row.trl_eval
            row.eval_ok = abs(row.eval_delta) <= eval_abs_tol
            all_ok = all_ok and bool(row.eval_ok)
        else:
            all_ok = False

        results.append(row)

    return ConvergenceGateSummary(
        passed=all_ok and bool(results),
        steps=results,
        blocked_reasons=blocked,
        fork_tb_dir=fork_dir,
        trl_tb_dir=trl_dir,
    )


def _online_watchdog_enabled() -> bool:
    raw = os.environ.get("VERL_GR_KL_GROWTH_GATE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _parse_step_float_map(env_name: str, default: dict[int, float]) -> dict[int, float]:
    """Parse ``200:0.05,600:0.05`` style env maps."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return dict(default)
    out: dict[int, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        step_s, val_s = part.split(":", 1)
        try:
            out[int(step_s.strip())] = float(val_s.strip())
        except ValueError:
            continue
    return out or dict(default)


def _kl_growth_floors() -> dict[int, float]:
    return _parse_step_float_map("VERL_GR_KL_GROWTH_FLOORS", KL_GROWTH_RATIO_FLOORS)


def _kl_abs_floors() -> dict[int, float]:
    return _parse_step_float_map("VERL_GR_KL_ABS_FLOORS", KL_ABS_FLOORS)


def _eval_max_lag() -> dict[int, float]:
    return _parse_step_float_map("VERL_GR_EVAL_MAX_LAG", EVAL_MAX_LAG_VS_TRL)


def _trl_series_at(
    cache_name: str,
    tag: str,
    step: int,
    *,
    trl_tb_dir: str | Path | None = None,
) -> float | None:
    global _TRL_KL_CACHE, _TRL_EVAL_CACHE
    cache = _TRL_KL_CACHE if cache_name == "kl" else _TRL_EVAL_CACHE
    if cache is None:
        trl_dir = str(trl_tb_dir or _resolve_trl_tb_dir())
        cache = _load_tb_scalar_series(trl_dir, tag) if trl_dir else {}
        if cache_name == "kl":
            _TRL_KL_CACHE = cache
        else:
            _TRL_EVAL_CACHE = cache
    if step in cache:
        return cache[step]
    if not cache:
        return None
    best = min(cache, key=lambda s: abs(s - step))
    if abs(best - step) <= max(20, step // 20):
        return cache[best]
    return None


def _trl_kl_at(step: int, *, trl_tb_dir: str | Path | None = None) -> float | None:
    return _trl_series_at("kl", _TRL_KL_TAG, step, trl_tb_dir=trl_tb_dir)


def _trl_eval_at(step: int, *, trl_tb_dir: str | Path | None = None) -> float | None:
    return _trl_series_at("eval", _TRL_EVAL_TAG, step, trl_tb_dir=trl_tb_dir)


def _metric_float(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in metrics:
            continue
        try:
            return float(metrics[key])
        except (TypeError, ValueError):
            continue
    return None


def _fork_kl_from_metrics(metrics: dict[str, Any]) -> float | None:
    return _metric_float(metrics, _FORK_KL_KEYS)


def _fork_eval_from_metrics(metrics: dict[str, Any]) -> float | None:
    return _metric_float(metrics, _FORK_EVAL_KEYS)


def _write_online_abort_report(
    *,
    trl_step: int,
    reasons: list[str],
    fork_kl: float | None,
    trl_kl: float | None,
    fork_eval: float | None,
    trl_eval: float | None,
    rollout_gap: float | None,
    report_name: str = "rankgrpo_online_watchdog.md",
    title: str = "RankGRPO Online Watchdog — ABORT",
    disable_note: str = "Disable online abort: `VERL_GR_KL_GROWTH_GATE=0`.",
) -> Path | None:
    global _ABORT_REPORT_WRITTEN
    if _ABORT_REPORT_WRITTEN:
        return None
    out_root = Path(os.environ.get("OUTPUT_DIR", "") or ".")
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / report_name
    lines = [
        f"# {title}",
        "",
        f"- generated: {datetime.now(timezone.utc).isoformat()}",
        f"- experiment: {os.environ.get('EXPERIMENT_NAME', 'unknown')}",
        f"- TRL step: {trl_step}",
        f"- fork_kl: {fork_kl}",
        f"- trl_kl: {trl_kl}",
        f"- fork_eval: {fork_eval}",
        f"- trl_eval: {trl_eval}",
        f"- rollout_train_logprob_gap: {rollout_gap}",
        "",
        "## Reasons",
        "",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is the **online** early-stop path (logging-step O(1) checks).",
            "- Exit-time `rankgrpo_convergence_gate.md` is offline-only and does not abort.",
            f"- {disable_note}",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _ABORT_REPORT_WRITTEN = True
    print(f"[rankgrpo] online watchdog abort report: {path}", flush=True)
    return path


def maybe_abort_on_kl_growth_failure(
    trl_step: int,
    metrics: dict[str, Any],
    *,
    trl_tb_dir: str | Path | None = None,
) -> None:
    """Online long-run watchdog: abort when KL/eval show policy is not learning.

    Checks (only at configured TRL steps, default 200/600/2000):

    1. Absolute KL floor (works even if TRL TB is missing)
    2. fork_kl / trl_kl ratio floor
    3. Self-trend: KL @ later gate must not collapse vs earlier gate
    4. eval/reward_total lag vs TRL (only when eval is present in this log batch)

    Called from the trainer logging path. Raises ``SystemExit(3)`` on failure.
    Overhead: O(1) metric lookups; TRL TB loaded once into process cache.
    """
    if not _online_watchdog_enabled():
        return
    step = int(trl_step)
    ratio_floors = _kl_growth_floors()
    abs_floors = _kl_abs_floors()
    eval_lags = _eval_max_lag()
    gate_steps = set(ratio_floors) | set(abs_floors) | set(eval_lags)
    if step not in gate_steps:
        return

    fork_kl = _fork_kl_from_metrics(metrics)
    fork_eval = _fork_eval_from_metrics(metrics)
    rollout_gap = _metric_float(metrics, _FORK_ROLLOUT_GAP_KEYS)
    trl_kl = _trl_kl_at(step, trl_tb_dir=trl_tb_dir)
    trl_eval = _trl_eval_at(step, trl_tb_dir=trl_tb_dir) if fork_eval is not None else None

    if fork_kl is not None:
        _FORK_KL_AT_GATE[step] = fork_kl

    reasons: list[str] = []

    abs_floor = abs_floors.get(step)
    if fork_kl is not None and abs_floor is not None and fork_kl < abs_floor:
        reasons.append(
            f"absolute KL too low: fork_kl={fork_kl:.6f} < floor {abs_floor} @step {step}"
        )

    ratio_floor = ratio_floors.get(step)
    if fork_kl is not None and trl_kl is not None and trl_kl > 0 and ratio_floor is not None:
        ratio = fork_kl / trl_kl
        if ratio < ratio_floor:
            reasons.append(
                f"KL ratio vs TRL too low: {ratio:.4f} < {ratio_floor} "
                f"(fork={fork_kl:.6f}, trl={trl_kl:.6f})"
            )

    prev_steps = sorted(s for s in _FORK_KL_AT_GATE if s < step)
    if fork_kl is not None and prev_steps:
        prev = prev_steps[-1]
        prev_kl = _FORK_KL_AT_GATE[prev]
        if prev_kl > 0 and fork_kl < 0.5 * prev_kl:
            reasons.append(
                f"KL collapsed vs earlier gate: @{step}={fork_kl:.6f} < 0.5×@{prev}={prev_kl:.6f}"
            )

    lag = eval_lags.get(step)
    if fork_eval is not None and trl_eval is not None and lag is not None:
        if fork_eval < trl_eval - lag:
            reasons.append(
                f"eval lag vs TRL: fork={fork_eval:.4f} trl={trl_eval:.4f} "
                f"delta={fork_eval - trl_eval:+.4f} < -{lag}"
            )

    if not reasons:
        bits = [f"@TRL step {step}"]
        if fork_kl is not None:
            bits.append(f"fork_kl={fork_kl:.6f}")
        if trl_kl is not None:
            bits.append(f"trl_kl={trl_kl:.6f}")
            if fork_kl is not None and trl_kl > 0:
                bits.append(f"ratio={fork_kl / trl_kl:.4f}")
        if fork_eval is not None:
            bits.append(f"fork_eval={fork_eval:.4f}")
            if trl_eval is not None:
                bits.append(f"trl_eval={trl_eval:.4f}")
                if fork_eval > trl_eval:
                    bits.append("eval_beats_trl=YES")
        print(f"[rankgrpo] online watchdog OK: {', '.join(bits)}", flush=True)
        return

    msg = (
        f"[rankgrpo] online watchdog FAILED @TRL step {step}: "
        + "; ".join(reasons)
        + ". Aborting to save compute. Disable with VERL_GR_KL_GROWTH_GATE=0."
    )
    print(msg, flush=True)
    _write_online_abort_report(
        trl_step=step,
        reasons=reasons,
        fork_kl=fork_kl,
        trl_kl=trl_kl,
        fork_eval=fork_eval,
        trl_eval=trl_eval,
        rollout_gap=rollout_gap,
    )
    raise SystemExit(3)


def _length_gate_enabled() -> bool:
    return os.environ.get("VERL_GR_LENGTH_GATE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _length_gate_min_step() -> int:
    try:
        return int(os.environ.get("VERL_GR_LENGTH_GATE_MIN_STEP", "200"))
    except ValueError:
        return 200


def _length_gate_float(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def maybe_abort_on_length_blowout(step: int, metrics: dict[str, Any]) -> None:
    """Abort when rollout completions show length blowout (no EOS, max-length clip, overflow).

    Monitors every logging step after ``VERL_GR_LENGTH_GATE_MIN_STEP`` (default 200).
    Disable with ``VERL_GR_LENGTH_GATE=0``.
    """
    if not _length_gate_enabled() or step < _length_gate_min_step():
        return

    eos_rate = _metric_float(metrics, _FORK_EOS_RATE_KEYS)
    clip_ratio = _metric_float(metrics, _FORK_CLIP_RATIO_KEYS)
    overflow_ratio = _metric_float(metrics, _FORK_OVERFLOW_RATIO_KEYS)
    if eos_rate is None and clip_ratio is None and overflow_ratio is None:
        return

    min_eos = _length_gate_float("VERL_GR_MIN_EOS_RATE", 0.5)
    max_clip = _length_gate_float("VERL_GR_MAX_CLIP_RATIO", 0.1)
    max_overflow = _length_gate_float("VERL_GR_MAX_OVERFLOW_RATIO", 0.2)

    reasons: list[str] = []
    if eos_rate is not None and eos_rate < min_eos:
        reasons.append(f"eos_rate too low: {eos_rate:.4f} < {min_eos}")
    if clip_ratio is not None and clip_ratio > max_clip:
        reasons.append(f"clip_ratio too high: {clip_ratio:.4f} > {max_clip}")
    if overflow_ratio is not None and overflow_ratio > max_overflow:
        reasons.append(f"overflow_token_ratio too high: {overflow_ratio:.4f} > {max_overflow}")

    if not reasons:
        bits = [f"@step {step}"]
        if eos_rate is not None:
            bits.append(f"eos_rate={eos_rate:.4f}")
        if clip_ratio is not None:
            bits.append(f"clip_ratio={clip_ratio:.4f}")
        if overflow_ratio is not None:
            bits.append(f"overflow_ratio={overflow_ratio:.4f}")
        print(f"[rankgrpo] length watchdog OK: {', '.join(bits)}", flush=True)
        return

    msg = (
        f"[rankgrpo] length watchdog FAILED @step {step}: "
        + "; ".join(reasons)
        + ". Aborting to prevent runaway completions. Disable with VERL_GR_LENGTH_GATE=0."
    )
    print(msg, flush=True)
    _write_online_abort_report(
        trl_step=step,
        reasons=reasons,
        fork_kl=_metric_float(metrics, _FORK_KL_KEYS),
        trl_kl=None,
        fork_eval=_metric_float(metrics, _FORK_EVAL_KEYS),
        trl_eval=None,
        rollout_gap=_metric_float(metrics, _FORK_ROLLOUT_GAP_KEYS),
        report_name="rankgrpo_length_watchdog.md",
        title="RankGRPO Length Blowout Watchdog",
    )
    raise SystemExit(3)


def write_convergence_gate_report(
    *,
    output_dir: str | Path,
    experiment_name: str | None = None,
    trl_tb_dir: str | Path | None = None,
    fork_tb_dir: str | Path | None = None,
) -> Path | None:
    if os.environ.get("VERL_GR_CONVERGENCE_GATE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None

    summary = evaluate_convergence_gate(
        fork_tb_dir=fork_tb_dir,
        trl_tb_dir=trl_tb_dir,
    )
    out_root = Path(output_dir)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    final_step = max((row.step for row in summary.steps), default=0)
    final_row = next((row for row in summary.steps if row.step == final_step), None)
    beats_trl_final = (
        final_row is not None
        and final_row.fork_eval is not None
        and final_row.trl_eval is not None
        and final_row.fork_eval > final_row.trl_eval
    )

    lines: list[str] = [
        "# RankGRPO Long-Horizon Convergence Gate",
        "",
        f"- generated: {datetime.now(timezone.utc).isoformat()}",
        f"- experiment: {experiment_name or os.environ.get('EXPERIMENT_NAME', 'unknown')}",
        f"- fork TB: `{summary.fork_tb_dir}`",
        f"- TRL TB: `{summary.trl_tb_dir}`",
        f"- gate passed (KL+eval tolerance): **{summary.passed}**",
        f"- final step eval beats TRL: **{beats_trl_final}**",
        "",
        "> Exit-time offline report only (reads TensorBoard). Not the step-30 ALIGN_DEBUG sidecar path.",
        "> Long-run target: **fork eval/reward_total > TRL** at final milestone.",
        "",
    ]
    if summary.blocked_reasons:
        lines.append("## Blocked / missing data")
        lines.append("")
        for reason in summary.blocked_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines.extend(
        [
            "## Milestones (TRL optimizer steps)",
            "",
            "| step | fork KL | TRL KL | KL rel err | KL ok | fork eval | TRL eval | Δ eval | beats TRL | eval ok | rollout↔train |",
            "|------|---------|--------|------------|-------|-----------|----------|--------|-----------|---------|---------------|",
        ]
    )
    for row in summary.steps:
        lines.append(
            "| {step} | {fkl} | {tkl} | {klre} | {klok} | {fev} | {tev} | {ed} | {beats} | {eok} | {gap} |".format(
                step=row.step,
                fkl=f"{row.fork_kl:.6f}" if row.fork_kl is not None else "—",
                tkl=f"{row.trl_kl:.6f}" if row.trl_kl is not None else "—",
                klre=f"{row.kl_rel_err:.4f}" if row.kl_rel_err is not None else "—",
                klok=row.kl_ok if row.kl_ok is not None else "—",
                fev=f"{row.fork_eval:.6f}" if row.fork_eval is not None else "—",
                tev=f"{row.trl_eval:.6f}" if row.trl_eval is not None else "—",
                ed=f"{row.eval_delta:+.6f}" if row.eval_delta is not None else "—",
                beats=row.eval_beats_trl if row.eval_beats_trl is not None else "—",
                eok=row.eval_ok if row.eval_ok is not None else "—",
                gap=f"{row.fork_rollout_gap:.6f}" if row.fork_rollout_gap is not None else "—",
            )
        )

    lines.extend(
        [
            "",
            "## Criteria",
            "",
            "- KL milestones: relative error vs TRL `train/kl` ≤ 20%",
            "- eval/reward_total milestones: |fork − TRL| ≤ 0.02 (alignment tolerance)",
            "- **Success target**: fork eval/reward_total > TRL at final step (see beats TRL column)",
            "- Fork KL tags: `actor/dbg/kl_tok_mean`, `actor/kl_loss`",
            "- Online early-stop (`VERL_GR_KL_GROWTH_GATE`): aborts mid-run if",
            f"  abs KL < configured floors, or fork/TRL KL ratio too low,",
            f"  or eval lags TRL by more than configured lag (when eval is logged).",
            "  Abort writes `logs/rankgrpo_online_watchdog.md`.",
            "",
        ]
    )

    report_path = log_dir / "rankgrpo_convergence_gate.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[rankgrpo] convergence gate report: {report_path} (passed={summary.passed}, beats_trl_final={beats_trl_final})")
    return report_path
