# OpenOneRec Parity Checklist (Phase B)

This checklist tracks old-vs-new behavior parity for the cleaned-up Phase B runtime.

## Legacy to New Flow Checklist

- [x] Legacy `main_onerec_ppo` is launched via `verl_gr.trainers.main_ppo`
- [x] Legacy `onerec_ray_trainer` lifecycle is mapped into `verl_gr.trainers.rl_trainer`
- [x] Legacy role-worker mapping is represented in OpenOneRec worker modules under `recipes/openonerec`
- [x] Two-stage rollout route resolves to the OneRec custom FSDP worker mapping
- [x] Two-stage rollout implementation lives under `workers/rollout` instead of the removed `components` layer

## Runtime/Worker Assumptions

- The Phase B runtime directly uses upstream `verl` trainer modules via local wrappers.
- Cluster init and `fit()` execution follow the launcher/main trainer path.
- OpenOneRec-specific preparation stays under `recipes/openonerec`.
- Worker route strings intentionally mirror OpenOneRec legacy names for traceability.

## Validation Gate (Smoke)

- runtime entrypoint module resolves and can be executed through `main_ppo`
- RL trainer extensions load successfully with expected OpenOneRec hooks
- config files are present with the OpenOneRec launcher defaults
- removed bridge/component/contract directories are no longer referenced by the runtime path

