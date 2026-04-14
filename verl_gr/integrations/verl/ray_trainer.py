"""Generic Ray-PPO runtime bridge backed by verl modules."""

from __future__ import annotations

import os
import subprocess
from importlib import import_module
from pathlib import Path
from typing import Mapping, Sequence


class RayPPOTrainerRuntime:
    """Stateful runtime bridge for launching verl PPO entrypoints."""

    def __init__(
        self,
        python_bin: str = "python",
        run_mode: str = "dry_run",
    ) -> None:
        self.python_bin = python_bin
        self.run_mode = run_mode

        self.status = "initialized"
        self.last_command: list[str] = []
        self.last_entrypoint: str = ""
        self.last_returncode: int | None = None

    def _import_verl_runtime(self):
        """Import verl runtime modules using canonical bridge boundaries."""

        constants_module = import_module("verl.trainer.constants_ppo")
        ray_module = import_module("ray")
        get_ppo_ray_runtime_env = getattr(constants_module, "get_ppo_ray_runtime_env")
        return ray_module, get_ppo_ray_runtime_env

    def detect_cluster(self) -> tuple[int, int]:
        """Detect active Ray cluster shape, fallback to local defaults."""

        try:
            ray, _ = self._import_verl_runtime()
            ray.init(address="auto", ignore_reinit_error=True)
            nodes = [node for node in ray.nodes() if node.get("Alive")]
            gpus = next(
                (
                    int(node.get("Resources", {}).get("GPU", 0))
                    for node in nodes
                    if node.get("Resources", {}).get("GPU", 0) > 0
                ),
                0,
            )
            return max(len(nodes), 1), max(gpus, 1)
        except Exception:
            return (1, 8)
        return (1, 8)

    def set_run_mode(self, run_mode: str) -> None:
        """Set execution mode for runtime launch."""

        self.run_mode = run_mode

    def run_entrypoint(
        self,
        entrypoint_module: str,
        overrides: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str] | None:
        """Execute or dry-run a verl-style PPO entrypoint module."""

        command = [self.python_bin, "-u", "-m", entrypoint_module, *list(overrides)]
        runtime_env = os.environ.copy()
        runtime_env.update(dict(env))
        runtime_env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        runtime_env.setdefault("WANDB_MODE", "offline")

        self.last_command = command
        self.last_entrypoint = entrypoint_module
        self.status = "running"
        if self.run_mode != "execute":
            self.status = "dry_run"
            return None

        completed = subprocess.run(
            command,
            cwd=Path(cwd) if cwd else None,
            env=runtime_env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.last_returncode = completed.returncode
        self.status = "completed"
        return completed

