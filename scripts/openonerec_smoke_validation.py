"""Minimal smoke validation for Phase B OpenOneRec wiring."""

from __future__ import annotations

from pathlib import Path

from verl_gr.contracts.artifact_contract import (
    ArtifactBundle,
    RewardOrDecodingArtifact,
    StageConfigArtifact,
    TokenizerArtifact,
)
from verl_gr.contracts.objective_schema import ObjectiveKind, RLRewardSchema, RewardComponent
from verl_gr.contracts.sample_schema import RepresentationType
from verl_gr.contracts.tokenizer_contract import TokenizedSample
from verl_gr.recipes.recipe_registry import build_default_registry
from verl_gr.recipes.openonerec.rl_pipeline import OpenOneRecRLPipeline
from verl_gr.trainers.rl_trainer import RLTrainer
from verl_gr.contracts.rl_contract import RLInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs/verl_gr/openonerec/base.yaml"
PATHS_CONFIG = PROJECT_ROOT / "configs/verl_gr/openonerec/paths.yaml"
MAPPING_DOC = PROJECT_ROOT / "docs/verl_gr/openonerec_mapping.md"
PARITY_DOC = PROJECT_ROOT / "docs/verl_gr/openonerec_parity_plan.md"


def _assert_contains(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        raise AssertionError(f"Expected '{needle}' in {path}")


def validate_recipe_registration() -> None:
    registry = build_default_registry()
    spec = registry.get("openonerec")
    assert spec.composition.task_type.value == "openonerec"
    assert spec.composition.representation_type.value == "sid"
    assert tuple(stage.value for stage in spec.composition.stages) == (
        "tokenizer",
        "sft",
        "distill",
        "rl",
        "eval",
    )


def validate_config_and_paths() -> None:
    assert BASE_CONFIG.exists(), f"Missing {BASE_CONFIG}"
    assert PATHS_CONFIG.exists(), f"Missing {PATHS_CONFIG}"
    _assert_contains(BASE_CONFIG, "rollout:")
    _assert_contains(BASE_CONFIG, "stage2_beam_size:")
    _assert_contains(PATHS_CONFIG, "stage_config_artifact:")
    _assert_contains(PATHS_CONFIG, "task_config_path:")


def validate_trainer_init_and_artifacts() -> None:
    tokenizer_artifact = TokenizerArtifact(
        tokenizer_root=PROJECT_ROOT / "artifacts/openonerec/tokenizer",
        representation_type=RepresentationType.SID,
        special_token_file=PROJECT_ROOT / "artifacts/openonerec/tokenizer/special_tokens.json",
        representation_schema_file=PROJECT_ROOT / "artifacts/openonerec/tokenizer/sid_schema.json",
    )
    config_artifact = StageConfigArtifact(
        task_config_path=BASE_CONFIG,
        stage_config_path=PATHS_CONFIG,
        objective_config_path=BASE_CONFIG,
    )
    tokenized = TokenizedSample(
        sample_id="sample-1",
        representation_type=RepresentationType.SID,
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        labels=(1, 2, 3),
        metadata={"stage2_beam_size": 8, "stage2_num_tokens": 12, "rollout_n": 2, "uid_group_key": "uid"},
    )
    reward_schema = RLRewardSchema(
        kind=ObjectiveKind.RL,
        name="openonerec_reward",
        components=(RewardComponent(name="ctr", weight=1.0),),
        normalization="adaptive_kl",
        constrained_decoding_aware=True,
    )
    rl_input = RLInput(
        tokenized_samples=(tokenized,),
        policy_model_path=PROJECT_ROOT / "models/openonerec/policy",
        tokenizer_artifact=tokenizer_artifact,
        config_artifact=config_artifact,
        reward_schema=reward_schema,
        reference_model_path=PROJECT_ROOT / "models/openonerec/reference",
        reward_or_decoding_artifact=RewardOrDecodingArtifact(
            reward_schema_path=PROJECT_ROOT / "artifacts/openonerec/reward_schema.json"
        ),
    )

    trainer = RLTrainer(runtime_bridge=OpenOneRecRLPipeline())
    output = trainer.run(rl_input)
    assert trainer.status == "completed"
    assert trainer.run_count == 1
    assert trainer.last_result is not None
    assert output.checkpoint.stage_name == "rl"
    assert output.metrics.get("status") == "initialized"

    bundle = ArtifactBundle(
        tokenizer=tokenizer_artifact,
        config=config_artifact,
        checkpoints=(output.checkpoint,),
        reward_or_decoding=RewardOrDecodingArtifact(
            reward_schema_path=PROJECT_ROOT / "artifacts/openonerec/reward_schema.json",
            decoding_policy_path=PROJECT_ROOT / "artifacts/openonerec/decoding_policy.json",
        ),
    )
    assert bundle.config.task_config_path == BASE_CONFIG
    assert bundle.checkpoints[0].checkpoint_root.name == "rl_checkpoints"


def validate_docs() -> None:
    assert MAPPING_DOC.exists(), f"Missing {MAPPING_DOC}"
    assert PARITY_DOC.exists(), f"Missing {PARITY_DOC}"
    _assert_contains(MAPPING_DOC, "Artifact Handoff Matrix")
    _assert_contains(PARITY_DOC, "Legacy to New Flow Checklist")


def main() -> None:
    validate_recipe_registration()
    validate_config_and_paths()
    validate_trainer_init_and_artifacts()
    validate_docs()
    print("OpenOneRec Phase B smoke validation passed.")


if __name__ == "__main__":
    main()

