"""OpenOneRec RL adapter owned by recipe layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verl_gr.contracts.artifact_contract import RewardOrDecodingArtifact
from verl_gr.contracts.rl_contract import RLInput, RLOutput
from verl_gr.integrations.verl.rl_runtime import RLRuntimeConfig, VerlRLRuntime
from verl_gr.integrations.verl.worker_factory import WorkerFactoryConfig, build_worker_routing


@dataclass
class OpenOneRecRLPipeline:
    """Translate RL contracts into backend bridge arguments."""

    runtime: VerlRLRuntime = field(default_factory=VerlRLRuntime)

    def build_runtime_args(self, rl_input: RLInput) -> dict[str, Any]:
        first_metadata = rl_input.tokenized_samples[0].metadata if rl_input.tokenized_samples else {}
        return {
            "policy_model_path": str(rl_input.policy_model_path),
            "reference_model_path": str(rl_input.reference_model_path) if rl_input.reference_model_path else None,
            "tokenizer_root": str(rl_input.tokenizer_artifact.tokenizer_root),
            "task_config_path": str(rl_input.config_artifact.task_config_path),
            "stage_config_path": (
                str(rl_input.config_artifact.stage_config_path)
                if rl_input.config_artifact.stage_config_path
                else None
            ),
            "rollout_n": first_metadata.get("rollout_n", 1),
            "stage1_max_tokens": first_metadata.get("stage1_max_tokens", 1024),
            "stage2_beam_size": first_metadata.get("stage2_beam_size", 32),
            "stage2_num_tokens": first_metadata.get("stage2_num_tokens", 16),
            "reward_components": [component.name for component in rl_input.reward_schema.components],
            "grpo_grouping_key": first_metadata.get("uid_group_key", "uid"),
        }

    def _build_worker_config(self, rl_input: RLInput) -> WorkerFactoryConfig:
        first_metadata = rl_input.tokenized_samples[0].metadata if rl_input.tokenized_samples else {}
        rollout_name = "two_stage" if rl_input.reward_schema.constrained_decoding_aware else "vllm"
        use_reference_policy = (
            rl_input.reward_schema.normalization in {"kl", "adaptive_kl"}
            or bool(first_metadata.get("use_kl_loss", False))
        )
        return WorkerFactoryConfig(
            actor_strategy=str(first_metadata.get("actor_strategy", "fsdp")),
            critic_enabled=bool(first_metadata.get("critic_enabled", True)),
            reward_model_enabled=bool(first_metadata.get("reward_model_enabled", False)),
            use_reference_policy=use_reference_policy,
            rollout_name=rollout_name,
            rollout_mode=str(first_metadata.get("rollout_mode", "sync")),
            resource_pool_id=str(first_metadata.get("resource_pool_id", "global_pool")),
        )

    def build_aux_artifact(self, rl_input: RLInput) -> RewardOrDecodingArtifact | None:
        decoding_path = None
        if rl_input.reward_schema.constrained_decoding_aware:
            decoding_path = Path("outputs/openonerec/decoding_policy.json")
        if rl_input.reward_or_decoding_artifact and rl_input.reward_or_decoding_artifact.reward_schema_path:
            return RewardOrDecodingArtifact(
                reward_schema_path=rl_input.reward_or_decoding_artifact.reward_schema_path,
                decoding_policy_path=decoding_path,
            )
        if decoding_path:
            return RewardOrDecodingArtifact(decoding_policy_path=decoding_path)
        return rl_input.reward_or_decoding_artifact

    def run(self, rl_input: RLInput) -> RLOutput:
        runtime_args = self.build_runtime_args(rl_input)
        worker_routing = build_worker_routing(self._build_worker_config(rl_input))

        checkpoint_root = Path(runtime_args.get("task_config_path", "outputs/openonerec")) / "rl_checkpoints"
        self.runtime.runtime_config = RLRuntimeConfig(
            trainer_entrypoint="verl_gr.recipes.openonerec.main_onerec_ppo",
            ray_runtime_env="ppo_default",
            checkpoint_root=checkpoint_root,
            dry_run=True,
        )
        output = self.runtime.run(rl_input, runtime_args=runtime_args, worker_routing=worker_routing)
        aux_artifact = self.build_aux_artifact(rl_input)
        if aux_artifact and aux_artifact.decoding_policy_path:
            output.traces["decoding_policy_path"] = str(aux_artifact.decoding_policy_path)
        return output

