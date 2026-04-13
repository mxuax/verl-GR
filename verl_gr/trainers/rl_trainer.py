"""RL trainer runtime object for verl-GR."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from verl_gr.contracts.artifact_contract import CheckpointArtifact
from verl_gr.contracts.rl_contract import RLInput, RLOutput
from verl_gr.integrations.runtime_adapter import TaskRuntime
from verl_gr.recipes.openonerec.grpo_runtime import OpenOneRecGRPORuntime


class RLTrainer:
    """Stateful RL runtime orchestrator.

    The trainer can execute RL in two ways:
    - delegate to an injected runtime bridge (integration path)
    - launch a real OpenOneRec GRPO workload via script
    """

    def __init__(
        self,
        name: str = "rl_trainer",
        metadata: Mapping[str, Any] | None = None,
        runtime_bridge: Any | None = None,
        *,
        workload: str = "bridge",
        grpo_script_path: str | Path = "scripts/run_openonerec_grpo.sh",
        run_mode: str = "dry_run",
        grpo_runtime: TaskRuntime | None = None,
    ) -> None:
        self.name = name
        self.metadata = dict(metadata or {})
        self.runtime_bridge = runtime_bridge
        self.workload = workload
        self.grpo_script_path = Path(grpo_script_path)
        self.run_mode = run_mode
        self.grpo_runtime = grpo_runtime
        self.status = "initialized"
        self.last_runtime_args: dict[str, str] = {}
        self.last_result: RLOutput | None = None
        self.run_count = 0

    def validate_input(self, rl_input: RLInput) -> None:
        """Validate the minimal RL input contract."""

        if not rl_input.tokenized_samples:
            raise ValueError("RLInput.tokenized_samples must not be empty.")

    def _build_grpo_env(self, rl_input: RLInput) -> dict[str, str]:
        """Map contract values to GRPO launcher environment variables."""

        first_metadata = rl_input.tokenized_samples[0].metadata if rl_input.tokenized_samples else {}
        env = os.environ.copy()
        env.update(
            {
                "BASE_MODEL": str(rl_input.policy_model_path),
                "REF_MODEL": str(rl_input.reference_model_path) if rl_input.reference_model_path else "",
                "ROLLOUT_N": str(first_metadata.get("rollout_n", 1)),
                "STAGE1_MAX_TOKENS": str(first_metadata.get("stage1_max_tokens", 1024)),
                "STAGE2_BEAM_SIZE": str(first_metadata.get("stage2_beam_size", 32)),
                "STAGE2_NUM_TOKENS": str(first_metadata.get("stage2_num_tokens", 3)),
                "ROLLOUT_TP_SIZE": str(first_metadata.get("rollout_tp_size", 1)),
                "TEMPERATURE": str(first_metadata.get("temperature", 1.0)),
                "USE_DYNAMIC_BSZ": str(first_metadata.get("use_dynamic_bsz", "True")),
                "MAX_TOKENS_PER_GPU": str(first_metadata.get("max_tokens_per_gpu", 40960)),
                "TRAIN_FILES": str(first_metadata.get("train_files", "[]")),
                "VAL_FILES": str(first_metadata.get("val_files", "[]")),
                "OUTPUT_DIR": str(first_metadata.get("output_dir", "outputs/openonerec")),
                "PROJECT_NAME": str(first_metadata.get("project_name", "verl-GR")),
                "EXPERIMENT_NAME": str(first_metadata.get("experiment_name", "grpo_two_stage")),
                "ENABLE_THINK": str(first_metadata.get("enable_think", "False")),
                "ENABLE_NONTHINK": str(first_metadata.get("enable_nonthink", "False")),
                "USE_FORCE_PREFIX": str(first_metadata.get("use_force_prefix", "False")),
            }
        )
        return env

    def _run_grpo_workload(self, rl_input: RLInput) -> RLOutput:
        """Launch OpenOneRec GRPO workload via RayPPO runtime or script fallback."""

        runtime_env = self._build_grpo_env(rl_input)
        self.last_runtime_args = {
            key: runtime_env[key]
            for key in ("BASE_MODEL", "ROLLOUT_N", "STAGE2_BEAM_SIZE", "OUTPUT_DIR")
        }

        runtime = self.grpo_runtime or OpenOneRecGRPORuntime()
        runtime.set_run_mode(self.run_mode)
        try:
            completed = runtime.run(runtime_env)
            if completed is None:
                return RLOutput(
                    checkpoint=CheckpointArtifact(
                        stage_name="rl",
                        checkpoint_root=Path(runtime_env["OUTPUT_DIR"]) / "ckpt",
                        metadata={
                            "mode": "dry_run",
                            "workload": "grpo",
                            "runner": "OpenOneRecGRPORuntime",
                        },
                    ),
                    metrics={"status": "initialized", "run_mode": self.run_mode},
                    traces={"command": " ".join(runtime.last_command)},
                )

            return RLOutput(
                checkpoint=CheckpointArtifact(
                    stage_name="rl",
                    checkpoint_root=Path(runtime_env["OUTPUT_DIR"]) / "ckpt",
                    metadata={
                        "mode": "execute",
                        "workload": "grpo",
                        "runner": "OpenOneRecGRPORuntime",
                    },
                ),
                metrics={"status": "completed", "returncode": completed.returncode},
                traces={
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                },
            )
        except FileNotFoundError:
            # fallback to script launcher when OpenOneRec runtime path is unavailable
            script_path = self.grpo_script_path
            if not script_path.is_absolute():
                script_path = Path(__file__).resolve().parents[2] / script_path
            if not script_path.exists():
                raise

        if self.run_mode != "execute":
            return RLOutput(
                checkpoint=CheckpointArtifact(
                    stage_name="rl",
                    checkpoint_root=Path(runtime_env["OUTPUT_DIR"]) / "ckpt",
                    metadata={
                        "mode": "dry_run",
                        "workload": "grpo",
                        "script_path": str(script_path),
                        "runner": "script_fallback",
                    },
                ),
                metrics={"status": "initialized", "run_mode": self.run_mode},
                traces={"command": f"bash {script_path}"},
            )

        completed = subprocess.run(
            ["bash", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
            env=runtime_env,
        )
        return RLOutput(
            checkpoint=CheckpointArtifact(
                stage_name="rl",
                checkpoint_root=Path(runtime_env["OUTPUT_DIR"]) / "ckpt",
                metadata={
                    "mode": "execute",
                    "workload": "grpo",
                    "script_path": str(script_path),
                        "runner": "script_fallback",
                },
            ),
            metrics={"status": "completed", "returncode": completed.returncode},
            traces={
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            },
        )

    def run(self, rl_input: RLInput) -> RLOutput:
        """Execute RL training runtime."""

        self.validate_input(rl_input)
        self.status = "running"
        self.run_count += 1

        result: RLOutput
        if self.workload == "grpo":
            result = self._run_grpo_workload(rl_input)
        elif self.runtime_bridge is not None and hasattr(self.runtime_bridge, "run"):
            result = self.runtime_bridge.run(rl_input)
        else:
            raise NotImplementedError("RLTrainer needs a runtime bridge or workload='grpo'.")

        self.last_result = result
        self.status = "completed"
        return result

