# OpenOneRec Mapping

This document records the explicit mapping from OpenOneRec legacy runtime
entrypoints to the current `verl_gr` runtime layout after the cleanup refactor.

## Entrypoint Mapping

- old `recipe.onerec.main_onerec_ppo` -> new `verl_gr.trainers.main_ppo`
- old `recipe.onerec.onerec_ray_trainer` -> new `verl_gr.trainers.rl_trainer.RLTrainer`
- old `recipe.onerec.onerec_fsdp_workers` -> new `verl_gr.recipes.openonerec.onerec_fsdp_workers.OneRecActorRolloutRefWorker`
- old `recipe.onerec.onerec_vllm_rollout` -> new `verl_gr.workers.rollout.two_stage_vllm_async`

## Async Two-Stage Path

- `rollout.name == two_stage` registers `verl_gr.workers.rollout.two_stage_vllm_async.TwoStagevLLMReplica`
- async request grouping is handled by `verl_gr.recipes.openonerec.two_stage_agent_loop`
- `onerec_recipe` wires the custom `AgentLoopManager` through config instead of patching upstream `verl`

### Where beam search actually runs now

Current implementation is **not** the old trainer-side Python beam orchestration.

- Agent loop worker only prepares per-sample metadata
  (`stage1_sample_idx`, `beam_index`, `beam_group_id`) and dispatches requests.
- Two-stage generation and beam expansion run inside
  `workers/rollout/two_stage_vllm_async.py::TwoStagevLLMHttpServer`:
  - stage-1 sampling,
  - stage-2 constrained beam search,
  - per-group cache reuse across beam indices,
  - inflight request throttling via semaphore.
- Shared beam kernel is `workers/rollout/beam_backend.py::run_async_beam_search`.

## Ownership

- `recipes/openonerec`: task-specific preparation, dataset/reward logic, and custom workers
- `trainers`: thin wrappers around upstream `verl` trainer code
- `workers/rollout`: rollout extensions that are still reusable at the worker layer
- `third_party`: light helpers for non-`verl` dependencies such as `vllm`

## Behavior-Critical RL Settings Preserved

- two-stage rollout routing (`rollout.name == two_stage`) selects OneRec runtime wiring inside `verl-GR`
- beam parameters (`stage2_beam_size`, `stage2_num_tokens`) are still consumed by rollout worker code
- validation still expands and scores beam candidates in OpenOneRec-specific trainer helpers
- runtime code now imports upstream `verl` directly instead of routing through local bridge layers
- upstream `verl` source remains unchanged; all OpenOneRec-specific behavior is injected from `verl-GR`

## Engine-Side Change Summary

Compared with the earlier low-efficiency approach (Python-side beam logic in
high-level runtime flow), current code moves performance-critical decoding to
the rollout server layer:

- beam cache lifetime is managed in `TwoStagevLLMHttpServer`,
- concurrent engine calls are bounded with `_two_stage_engine_request_semaphore`,
- abort/cleanup path clears pending build tasks and cache entries,
- generation result selection (`_beam_index`) is served from cached beam groups.

## Mapping Conclusion

| Scope | Current status |
|---|---|
| Effective launch hyperparameters | Matched for the intended 4-GPU run shape and current feature set. |
| Checkpoint selection by validation score | Preserved in both launchers. |
| Validation/logging/runtime feature knobs | Now mirrored into the OpenOneRec launcher instead of removed from verl-GR. |
| Shell validity | `bash -n` passes for `run_openonerec_grpo.sh`, `run_matchup.sh`, and `openonerec_fredfork/verl_rl/recipe/onerec/run_grpo.sh`. |
| Structural differences | Still intentionally different: `run_openonerec_grpo.sh` uses `verl_gr.trainers.main_ppo` and the verl-GR recipe path, while OpenOneRec uses `recipe.onerec.main_onerec_ppo` and its local recipe path. |

Final corrected conclusion: the hyperparameter and useful validation/checkpoint feature settings are now matched by modifying the OpenOneRec launcher, while preserving the verl-GR checkpoint-saving behavior.
