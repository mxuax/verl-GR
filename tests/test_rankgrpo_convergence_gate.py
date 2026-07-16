"""Long-horizon / online convergence gate tests."""

from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "verl_gr").is_dir():
    _p = _p.parent
if (_p / "verl_gr").is_dir() and str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import pytest

from verl_gr.recipes.rankgrpo.alignment.convergence_gate import (
    ConvergenceStepResult,
    evaluate_convergence_gate,
    maybe_abort_on_kl_growth_failure,
    maybe_abort_on_length_blowout,
)
import verl_gr.recipes.rankgrpo.alignment.convergence_gate as cg


def _reset_watchdog_state():
    cg._TRL_KL_CACHE = None
    cg._TRL_EVAL_CACHE = None
    cg._FORK_KL_AT_GATE.clear()
    cg._ABORT_REPORT_WRITTEN = False


def test_evaluate_convergence_gate_passes_when_metrics_align(monkeypatch, tmp_path):
    fork = tmp_path / "fork"
    trl = tmp_path / "trl"
    fork.mkdir()
    trl.mkdir()

    def _fake_load(tb_dir, tag):
        if tag == "train/kl" and str(tb_dir) == str(trl):
            return {200: 0.01, 400: 0.02, 600: 0.04}
        if tag in ("actor/train/kl", "train/kl") and str(tb_dir) == str(fork):
            return {200: 0.011, 400: 0.021, 600: 0.041}
        if tag == "eval/reward_total" and str(tb_dir) == str(trl):
            return {200: 0.36, 400: 0.37, 600: 0.39}
        if tag == "eval/reward_total" and str(tb_dir) == str(fork):
            return {200: 0.355, 400: 0.375, 600: 0.385}
        return {}

    monkeypatch.setattr(
        "verl_gr.recipes.rankgrpo.alignment.convergence_gate._load_tb_scalar_series",
        _fake_load,
    )
    summary = evaluate_convergence_gate(fork_tb_dir=fork, trl_tb_dir=trl)
    assert summary.passed
    assert len(summary.steps) == 3
    assert all(row.kl_ok for row in summary.steps)
    assert all(row.eval_ok for row in summary.steps)


def test_evaluate_convergence_gate_fails_on_large_eval_gap(monkeypatch, tmp_path):
    fork = tmp_path / "fork"
    trl = tmp_path / "trl"
    fork.mkdir()
    trl.mkdir()

    def _fake_load(tb_dir, tag):
        if tag == "train/kl":
            return {600: 0.04}
        if tag == "eval/reward_total" and str(tb_dir) == str(trl):
            return {600: 0.39}
        if tag == "eval/reward_total" and str(tb_dir) == str(fork):
            return {600: 0.30}
        return {}

    monkeypatch.setattr(
        "verl_gr.recipes.rankgrpo.alignment.convergence_gate._load_tb_scalar_series",
        _fake_load,
    )
    summary = evaluate_convergence_gate(fork_tb_dir=fork, trl_tb_dir=trl, steps=(600,))
    assert not summary.passed
    row: ConvergenceStepResult = summary.steps[0]
    assert row.eval_ok is False


def test_online_watchdog_aborts_on_abs_kl_floor(monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_GR_KL_GROWTH_GATE", "1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_GR_KL_ABS_FLOORS", "200:0.001")
    monkeypatch.setenv("VERL_GR_KL_GROWTH_FLOORS", "200:0.0")  # disable ratio check
    monkeypatch.setattr(
        "verl_gr.recipes.rankgrpo.alignment.convergence_gate._load_tb_scalar_series",
        lambda *a, **k: {},
    )
    _reset_watchdog_state()
    with pytest.raises(SystemExit) as exc:
        maybe_abort_on_kl_growth_failure(200, {"actor/train/kl": 0.0002})
    assert exc.value.code == 3
    assert (tmp_path / "logs" / "rankgrpo_online_watchdog.md").is_file()


def test_online_watchdog_aborts_when_ratio_too_low(monkeypatch, tmp_path):
    trl = tmp_path / "trl"
    trl.mkdir()
    monkeypatch.setenv("VERL_GR_KL_GROWTH_GATE", "1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_GR_KL_GROWTH_FLOORS", "200:0.05")
    monkeypatch.setenv("VERL_GR_KL_ABS_FLOORS", "200:0.0")
    monkeypatch.setattr(
        "verl_gr.recipes.rankgrpo.alignment.convergence_gate._load_tb_scalar_series",
        lambda tb_dir, tag: {200: 0.01} if tag == "train/kl" else {},
    )
    _reset_watchdog_state()
    with pytest.raises(SystemExit) as exc:
        maybe_abort_on_kl_growth_failure(200, {"actor/train/kl": 0.0001}, trl_tb_dir=trl)
    assert exc.value.code == 3


def test_online_watchdog_aborts_on_eval_lag(monkeypatch, tmp_path):
    trl = tmp_path / "trl"
    trl.mkdir()
    monkeypatch.setenv("VERL_GR_KL_GROWTH_GATE", "1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_GR_KL_ABS_FLOORS", "200:0.0")
    monkeypatch.setenv("VERL_GR_KL_GROWTH_FLOORS", "200:0.0")
    monkeypatch.setenv("VERL_GR_EVAL_MAX_LAG", "200:0.05")
    monkeypatch.setattr(
        "verl_gr.recipes.rankgrpo.alignment.convergence_gate._load_tb_scalar_series",
        lambda tb_dir, tag: {200: 0.40} if tag == "eval/reward_total" else {},
    )
    _reset_watchdog_state()
    with pytest.raises(SystemExit) as exc:
        maybe_abort_on_kl_growth_failure(
            200,
            {"actor/train/kl": 0.01, "eval/reward_total": 0.30},
            trl_tb_dir=trl,
        )
    assert exc.value.code == 3


def test_online_watchdog_passes_when_ok(monkeypatch, tmp_path):
    trl = tmp_path / "trl"
    trl.mkdir()
    monkeypatch.setenv("VERL_GR_KL_GROWTH_GATE", "1")
    monkeypatch.setenv("VERL_GR_KL_GROWTH_FLOORS", "200:0.05")
    monkeypatch.setenv("VERL_GR_KL_ABS_FLOORS", "200:0.001")
    monkeypatch.setattr(
        "verl_gr.recipes.rankgrpo.alignment.convergence_gate._load_tb_scalar_series",
        lambda tb_dir, tag: {200: 0.01} if tag == "train/kl" else {},
    )
    _reset_watchdog_state()
    maybe_abort_on_kl_growth_failure(200, {"actor/train/kl": 0.002}, trl_tb_dir=trl)


def test_length_watchdog_aborts_on_low_eos_rate(monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_GR_LENGTH_GATE", "1")
    monkeypatch.setenv("VERL_GR_LENGTH_GATE_MIN_STEP", "100")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    _reset_watchdog_state()
    with pytest.raises(SystemExit) as exc:
        maybe_abort_on_length_blowout(
            200,
            {
                "train/rankgrpo/items/eos_rate": 0.0,
                "train/rankgrpo/completions/clipped_ratio": 1.0,
                "train/rankgrpo/items/overflow_token_ratio": 0.9,
            },
        )
    assert exc.value.code == 3
    assert (tmp_path / "logs" / "rankgrpo_length_watchdog.md").is_file()


def test_length_watchdog_passes_when_healthy(monkeypatch):
    monkeypatch.setenv("VERL_GR_LENGTH_GATE", "1")
    monkeypatch.setenv("VERL_GR_LENGTH_GATE_MIN_STEP", "100")
    maybe_abort_on_length_blowout(
        200,
        {
            "train/rankgrpo/items/eos_rate": 1.0,
            "train/rankgrpo/completions/clipped_ratio": 0.0,
            "train/rankgrpo/items/overflow_token_ratio": 0.0,
        },
    )


def test_length_watchdog_skips_before_min_step(monkeypatch):
    monkeypatch.setenv("VERL_GR_LENGTH_GATE", "1")
    monkeypatch.setenv("VERL_GR_LENGTH_GATE_MIN_STEP", "500")
    maybe_abort_on_length_blowout(200, {"train/rankgrpo/items/eos_rate": 0.0})
