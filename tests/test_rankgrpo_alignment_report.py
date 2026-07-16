"""Tests for RankGRPO TRL alignment report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from verl import DataProto

from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import (
    RankGRPOAlignmentAccumulator,
    alignment_report_enabled,
    calculate_rankgrpo_logprob_gate_metrics,
    compute_modular_step_latency_summary,
    compute_sidecar_probe_overhead_summary,
    evaluate_rankgrpo_alignment_gate,
    format_modular_step_latency_markdown,
    format_sidecar_probe_overhead_markdown,
    get_rankgrpo_alignment_accumulator,
    modular_step_latency_enabled,
    record_rankgrpo_alignment_metrics,
    write_rankgrpo_alignment_report,
    _metric_gate_passed,
)


def test_logprob_gate_rollout_minus_ref_labels():
    b, t = 1, 4
    rollout = torch.tensor([[-0.1, -0.2, -0.3, 0.0]])
    ref = torch.tensor([[-0.12, -0.2, -0.28, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    batch = DataProto.from_single_dict(
        {
            "old_log_probs": rollout.clone(),
            "rollout_log_probs": rollout.clone(),
            "ref_log_prob": ref,
            "item_token_mask": mask,
        }
    )
    metrics = calculate_rankgrpo_logprob_gate_metrics(batch)
    assert metrics["logprob_gate/rollout_minus_rollout/abs_mean"] == 0.0
    assert metrics["logprob_gate/rollout_minus_ref/abs_mean"] > 0.0
    assert metrics["logprob_gate/bypass_mode"] == 1.0


def test_metric_gate_pass_fraction():
    assert _metric_gate_passed(2, 3) is True
    assert _metric_gate_passed(1, 3) is False
    assert _metric_gate_passed(0, 0) is False


def test_alignment_report_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_DEBUG_STEP", "10")
    acc = get_rankgrpo_alignment_accumulator()
    acc.steps.clear()
    acc.metrics_by_step.clear()

    record_rankgrpo_alignment_metrics(
        10,
        {
            "actor/kl_loss": 0.002,
            "train/rankgrpo/reward_total": 0.2,
            "train/rankgrpo/completions/mean_length": 190.0,
            "logprob_gate/rollout_minus_rollout/abs_mean": 0.0,
            "actor/debug/logprob_diff_abs": 0.02,
            "actor/pg_clipfrac": 0.0,
            "timing_s/step": 5.0,
            "timing_s/gen": 0.5,
            "timing_s/update_actor": 1.7,
        },
    )

    out = write_rankgrpo_alignment_report(
        output_dir=tmp_path,
        trl_tb_dir="/nonexistent/trl_tb",
        experiment_name="test_run",
    )
    assert out is not None
    report_path, gate = out
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "RankGRPO TRL Alignment Report" in text
    assert "RUN_DEBUG_STEP" in text
    assert "Per-step alignment gate" in text
    assert "| step | logprob gate | KL gate | fork KL | TRL KL | step time gate | fork time | TRL time | gate |" in text
    assert "Gate Verdicts" in text
    assert "pass / (pass + fail) ≥ 67%" in text
    assert "| logprob |" in text
    assert "| **combined** |" in text
    assert "Modular step latency" in text
    assert "Sidecar probe overhead" in text

    json_path = tmp_path / "logs" / "rankgrpo_align_report_step10.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["last_step"] == 10
    assert "gate" in payload
    assert "modular_step_latency" in payload
    assert "sidecar_probe_overhead" in payload
    assert payload["modular_step_latency"]["fork_total_seconds"] == 5.0


def test_offline_tb_alignment_report_writes(tmp_path):
    from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import write_offline_tb_alignment_report

    fork_tb = (
        "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup_verlgr/verl-gr-fork-main/"
        "tensorboard_log/RankGRPO/logprob_align_v015_TP2_g0_1"
    )
    trl_tb = (
        "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/"
        "logs/debug_precision_verlgr/runs/Jul06_12-05-38_hk01dgx028"
    )
    if not Path(fork_tb).is_dir() or not Path(trl_tb).is_dir():
        return

    out = write_offline_tb_alignment_report(
        fork_tb_dir=fork_tb,
        trl_tb_dir=trl_tb,
        output_dir=tmp_path,
        experiment_name="offline_test",
        max_step=5,
        report_stem="offline_test",
    )
    assert out is not None
    assert out.exists()
    assert "Precision Alignment" in out.read_text(encoding="utf-8")


def test_alignment_report_disabled_without_env(monkeypatch):
    monkeypatch.delenv("RUN_DEBUG_STEP", raising=False)
    assert alignment_report_enabled() is False
    assert write_rankgrpo_alignment_report(output_dir="/tmp") is None


def test_per_step_gate_kl_and_timing(monkeypatch, tmp_path):
    trl_tb = (
        "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/"
        "logs/debug_precision_verlgr/runs/Jul06_12-05-38_hk01dgx028"
    )
    if not Path(trl_tb).is_dir():
        return

    acc = RankGRPOAlignmentAccumulator()
    acc.record(
        2,
        {
            "actor/kl_loss": 0.0002,
            "logprob_gate/rollout_minus_ref/abs_mean": 0.0057,
            "actor/debug/logprob_diff_abs": 0.0057,
            "timing_s/step": 5.0,
        },
    )
    sidecar = tmp_path / "rankgrpo_gate_sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "train/logprob_gate/rollout_minus_ref/abs_mean": {"2": 0.0057},
                "train/actor/debug/logprob_diff_abs": {"2": 0.0057},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VERL_GR_TRL_GATE_SIDECAR", str(sidecar))
    gate = evaluate_rankgrpo_alignment_gate(acc, trl_tb_dir=trl_tb, max_step=2)
    assert gate.steps
    row = gate.steps[0]
    assert row.step == 2
    assert row.kl_ok or row.kl_rel_err is not None
    assert row.time_ok is not None
    assert row.fork_step_time == 5.0
    assert gate.fork_step_time_avg == 5.0
    assert gate.time_gate.passed
    assert row.logprob_ok is True


def test_modular_step_latency_only_when_run_debug_step(monkeypatch):
    acc = RankGRPOAlignmentAccumulator()
    acc.record(
        5,
        {
            "timing_s/step": 5.0,
            "timing_s/gen": 0.5,
            "timing_s/update_actor": 1.7,
            "timing_s/update_weights": 1.0,
            "timing_s/ref": 0.4,
            "timing_s/adv": 0.1,
        },
    )
    monkeypatch.delenv("RUN_DEBUG_STEP", raising=False)
    assert modular_step_latency_enabled() is False
    assert compute_modular_step_latency_summary(acc) is None

    monkeypatch.setenv("RUN_DEBUG_STEP", "30")
    assert modular_step_latency_enabled() is True
    summary = compute_modular_step_latency_summary(acc, compare_steps=[5], skip_warmup_steps=0)
    assert summary is not None
    assert summary.fork_total_seconds == 5.0
    total_row = summary.rows[-1]
    assert "Total" in total_row.phase
    assert total_row.fork_seconds == 5.0
    gen_row = summary.rows[0]
    assert gen_row.phase.startswith("gen")
    assert gen_row.fork_seconds == 0.5
    assert gen_row.fork_pct is not None
    assert abs(gen_row.fork_pct - 10.0) < 0.5
    md = format_modular_step_latency_markdown(summary)
    assert any("Modular step latency" in line for line in md)
    assert any("verl-gr Time" in line for line in md)


def test_sidecar_probe_overhead_summary(monkeypatch):
    monkeypatch.setenv("RUN_DEBUG_STEP", "30")
    acc = RankGRPOAlignmentAccumulator()
    for step, step_s in ((2, 5.2), (3, 5.0), (4, 4.8)):
        acc.record(
            step,
            {
                "timing_s/step": step_s,
                "timing_s/gen": 0.5,
                "timing_s/update_actor": 1.7,
                "timing_s/update_weights": 1.0,
                "timing_s/ref": 0.4,
                "timing_s/adv": 0.1,
                "timing_rankgrpo/probe_logprob_gate": 0.02,
                "timing_rankgrpo/probe_align_accum": 0.001,
                "timing_rankgrpo/probe_tb_log": 0.15,
            },
        )
    latency = compute_modular_step_latency_summary(acc, compare_steps=[2, 3, 4], skip_warmup_steps=1)
    probe = compute_sidecar_probe_overhead_summary(
        acc,
        compare_steps=[2, 3, 4],
        latency_summary=latency,
        skip_warmup_steps=1,
    )
    assert probe is not None
    assert probe.measured_probe_total is not None
    assert abs(probe.measured_probe_total - 0.171) < 0.01
    assert probe.step_distribution["mean"] > 4.5
    md = format_sidecar_probe_overhead_markdown(probe)
    assert any("Sidecar probe overhead" in line for line in md)
    assert any("logprob gate metrics" in line for line in md)
