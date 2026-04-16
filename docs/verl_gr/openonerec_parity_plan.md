# OpenOneRec Parity Checklist (Phase B)

This checklist tracks old-vs-new behavior parity for the Phase B integration layer.

## Legacy to New Flow Checklist

- [x] Legacy `main_onerec_ppo` is launched via `verl_gr.trainers.main_ppo`
- [x] Legacy `onerec_ray_trainer` lifecycle is mapped into `verl_gr.trainers.rl_trainer`
- [x] Legacy role-worker mapping is represented in OpenOneRec worker modules under `recipes/openonerec`
- [x] Two-stage rollout route resolves to the OneRec custom FSDP worker mapping
- [x] RL adapter preserves beam/two-stage metadata in runtime arguments

## Contract Alignment Checklist

- [x] `TaskComposition` uses `TaskType.OPENONEREC` + `RepresentationType.SID`
- [x] Stage order includes tokenizer before RL/Eval
- [x] Config files produce resolvable `StageConfigArtifact` paths
- [x] RL flow emits a contract-compliant `RLOutput` with `CheckpointArtifact`
- [x] Reward/decoding auxiliary paths map to `RewardOrDecodingArtifact`

## Runtime/Worker Assumptions

- The Phase B runtime directly uses upstream `verl` trainer modules via local wrappers.
- Cluster init and `fit()` execution follow the launcher/main trainer path.
- Worker route strings intentionally mirror OpenOneRec legacy names for traceability.

## Validation Gate (Smoke)

- runtime entrypoint module resolves and can be executed through `main_ppo`
- RL trainer extensions load successfully with expected OpenOneRec hooks
- config files are present with required keys for paths/stage config linkage
- artifact handoff matrix can be instantiated into an `ArtifactBundle`

