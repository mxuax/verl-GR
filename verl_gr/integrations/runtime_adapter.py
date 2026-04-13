"""Shared runtime adapter protocol for pluggable task runtimes."""

from __future__ import annotations

import subprocess
from typing import Mapping, Protocol, Sequence


class TaskRuntime(Protocol):
    """Minimal adapter contract for task runtime implementations."""

    status: str
    last_command: list[str]

    def set_run_mode(self, run_mode: str) -> None:
        """Set runtime execution mode (for example dry_run/execute)."""

    def run(
        self,
        env: Mapping[str, str],
        extra_overrides: Sequence[str] | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
        """Run one workload invocation and return optional process result."""

