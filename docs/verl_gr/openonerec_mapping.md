# OpenOneRec Mapping

This document defines the explicit Phase B mapping from OpenOneRec legacy runtime
entrypoints to the `verl_gr` recipe + integration bridge layout.

## Entrypoint Mapping

- old `recipe.onerec.main_onerec_ppo` -> new `verl_gr.recipes.openonerec_recipe.OpenOneRecRecipe`
- old `recipe.onerec.onerec_ray_trainer` -> new `verl_gr.integrations.verl.rl_runtime.VerlRLRuntime`
- old `recipe.onerec.onerec_fsdp_workers` -> new `verl_gr.integrations.verl.worker_factory.build_worker_routing`
- old `recipe.onerec.onerec_vllm_rollout` -> new `verl_gr.recipes.openonerec.rl_pipeline.OpenOneRecRLPipeline`

## Stage Ownership (Boundary-Preserving)

- `recipe`: stage composition + adapter selection only
- `recipes/openonerec`: task-specific contract translation + OpenOneRec adapter entrypoint metadata
- `integrations/verl`: runtime lifecycle + role-to-worker routing
- `trainers/rl_trainer`: invokes runtime bridge, does not embed framework internals

## Artifact Handoff Matrix

| Artifact | Producer | Consumer | Contract Type | Notes |
| --- | --- | --- | --- | --- |
| Tokenizer root/schema | tokenizer stage | SFT/Distill/RL adapters | `TokenizerArtifact` | `representation_type` must remain `sid` for OpenOneRec |
| Task + stage config paths | config layer | all pipelines | `StageConfigArtifact` | `task_config_path` points to `base.yaml`, stage path to `paths.yaml` |
| SFT checkpoint | SFT adapter | distill/eval | `CheckpointArtifact` | stage name `sft` |
| Distill checkpoint | distill adapter | RL/eval | `CheckpointArtifact` | stage name `distill` |
| RL checkpoint | RL runtime bridge | eval/export | `CheckpointArtifact` | stage name `rl` |
| Reward schema / decoding metadata | RL config or runtime | RL runtime + eval | `RewardOrDecodingArtifact` | preserve constrained decoding + beam metadata path |

## Behavior-Critical RL Settings Preserved

- two-stage rollout routing (`rollout.name == two_stage`) selects OneRec custom actor worker mapping
- beam parameters (`stage2_beam_size`, `stage2_num_tokens`) are carried through runtime args
- GRPO grouping key defaults to `uid` but is configurable in tokenized metadata
- KL-aware reference policy routing is enabled when normalization/metadata requests KL coupling

