# OpenOneRec Mapping

This document records the explicit mapping from OpenOneRec legacy runtime
entrypoints to the current `verl_gr` runtime layout after the cleanup refactor.

## Entrypoint Mapping

- old `recipe.onerec.main_onerec_ppo` -> new `verl_gr.trainers.main_ppo`
- old `recipe.onerec.onerec_ray_trainer` -> new `verl_gr.trainers.rl_trainer.RLTrainer`
- old `recipe.onerec.onerec_fsdp_workers` -> new `verl_gr.recipes.openonerec.onerec_fsdp_workers.OneRecActorRolloutRefWorker`
- old `recipe.onerec.onerec_vllm_rollout` -> new `verl_gr.workers.rollout.two_stage_vllm_rollout`

## Ownership

- `recipes/openonerec`: task-specific preparation, dataset/reward logic, and custom workers
- `trainers`: thin wrappers around upstream `verl` trainer code
- `workers/rollout`: rollout extensions that are still reusable at the worker layer
- `third_party`: light helpers for non-`verl` dependencies such as `vllm`

## Behavior-Critical RL Settings Preserved

- two-stage rollout routing (`rollout.name == two_stage`) selects OneRec custom actor worker mapping
- beam parameters (`stage2_beam_size`, `stage2_num_tokens`) are still consumed by rollout worker code
- validation still expands and scores beam candidates in OpenOneRec-specific trainer helpers
- runtime code now imports upstream `verl` directly instead of routing through local bridge layers

