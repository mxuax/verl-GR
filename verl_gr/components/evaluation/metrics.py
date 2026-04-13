"""Reusable metric helpers for verl-GR Phase A."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class MetricReport:
    """Minimal metric report shared by trainer skeletons and evaluation."""

    scalars: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)


def merge_metric_reports(*reports: MetricReport) -> MetricReport:
    """Merge multiple metric reports into one."""

    scalars: dict[str, float] = {}
    metadata: dict[str, str] = {}
    for report in reports:
        scalars.update(report.scalars)
        metadata.update(report.metadata)
    return MetricReport(scalars=scalars, metadata=metadata)

