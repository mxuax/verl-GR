"""Thin RL runtime lifecycle bridge for verl-backed execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from verl_gr.contracts.artifact_contract import CheckpointArtifact
from verl_gr.contracts.rl_contract import RLInput, RLOutput
from verl_gr.integrations.verl.worker_factory import WorkerRouting


@dataclass(frozen=True)
class RLRuntimeConfig:
    """Runtime-level values mapped from stage configuration."""

    trainer_entrypoint: str = "OpenOneRec.verl_rl.recipe.onerec.main_onerec_ppo"
    ray_runtime_env: str = "ppo_default"
    checkpoint_root: Path = Path("outputs/openonerec/rl")
    dry_run: bool = True


@dataclass(frozen=True)
class RuntimeTrainerHandle:
    """Description of the trainer object created by runtime bridge."""

    trainer_entrypoint: str
    worker_routing: WorkerRouting
    runtime_args: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class VerlRLRuntime:
    """Own RL runtime lifecycle: init env, build trainer, fit, artifacts."""

    runtime_config: RLRuntimeConfig = field(default_factory=RLRuntimeConfig)

    def initialize_runtime_env(self, runtime_args: Mapping[str, Any]) -> Mapping[str, Any]:
        """Prepare runtime metadata required before trainer instantiation."""

        return {
            "ray_runtime_env": self.runtime_config.ray_runtime_env,
            "trainer_entrypoint": self.runtime_config.trainer_entrypoint,
            "runtime_args": dict(runtime_args),
        }

    def build_trainer(self, runtime_args: Mapping[str, Any], worker_routing: WorkerRouting) -> RuntimeTrainerHandle:
        """Build a lightweight trainer handle consumed by the fit loop."""

        return RuntimeTrainerHandle(
            trainer_entrypoint=self.runtime_config.trainer_entrypoint,
            worker_routing=worker_routing,
            runtime_args=dict(runtime_args),
        )

    def run_fit_loop(self, trainer: RuntimeTrainerHandle) -> Mapping[str, Any]:
        """Execute training loop. Phase B keeps this as dry-run by default."""

        if not self.runtime_config.dry_run:
            raise NotImplementedError("Concrete verl fit loop execution is deferred beyond Phase B.")
        return {
            "status": "initialized",
            "trainer_entrypoint": trainer.trainer_entrypoint,
            "worker_roles": sorted(role.value for role in trainer.worker_routing.role_worker_mapping),
        }

    def collect_artifacts(self, rl_input: RLInput, fit_metrics: Mapping[str, Any]) -> RLOutput:
        """Collect contract-compliant checkpoint artifact for RL stage."""

        checkpoint_root = Path(self.runtime_config.checkpoint_root)
        metadata = {
            "policy_model_path": str(rl_input.policy_model_path),
            "reward_schema": rl_input.reward_schema.name,
            "runtime_status": str(fit_metrics.get("status", "unknown")),
        }
        return RLOutput(
            checkpoint=CheckpointArtifact(
                stage_name="rl",
                checkpoint_root=checkpoint_root,
                metadata=metadata,
            ),
            metrics=dict(fit_metrics),
            traces={"task_config_path": str(rl_input.config_artifact.task_config_path)},
        )

    def run(self, rl_input: RLInput, runtime_args: Mapping[str, Any], worker_routing: WorkerRouting) -> RLOutput:
        """Run full runtime lifecycle and return RL output artifact."""

        self.initialize_runtime_env(runtime_args)
        trainer = self.build_trainer(runtime_args, worker_routing)
        fit_metrics = self.run_fit_loop(trainer)
        return self.collect_artifacts(rl_input, fit_metrics)

