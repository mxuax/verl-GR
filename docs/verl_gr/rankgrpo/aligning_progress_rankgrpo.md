# Aligning verl-gr RankGRPO with TRL — Progress by Target Item

Goal: make verl-gr RankGRPO match or exceed TRL's convergence rate and compute
efficiency.

Reference target analysis:
`docs/verl_gr/rankgrpo/aligning_target_rankgrpo.md`.

This document is organized in the same order as the target analysis. It tracks
what has been aligned, what remains different, and what still needs evidence
from fresh runs.

Status legend:

- **Done**: code/config default has been changed and the target behavior is now
  represented by the default launch path.
- **Partial**: a likely cause has been addressed, but the run evidence is not
  complete yet.
- **Pending**: known target item is not implemented or not yet tested.

## Current Status Snapshot

| Target item | Status | Current state | Remaining work |
|---|---|---|---|
| 1.1 Effective batch size | **Done** | good `fp32opt` run uses 6 unique prompts, 8 rollouts, 48 generated sequences, 6 accumulation micro-batches, 1 optimizer step | Continue using this as baseline |
| 1.2 PPO clip ratio | **Done** | verl-gr defaults to `[0.94, 1.08]`, matching TRL | Monitor `actor/pg_clipfrac` |
| 1.3 PPO epochs / sequence reuse | **Done** | good `fp32opt` run shows `ppo_epochs=1`, matching TRL `mu=1` | Keep no override in future runs |
| 1.4 Other aligned hparams | **Done** | LR, KL coefficient, Adam betas, shuffle behavior, rollout count, seed, and actor dtype defaults are aligned in `fp32opt` | Keep these fixed for follow-ups |
| 1.5 Distributed backend | **Partial** | TRL uses DeepSpeed ZeRO-3 + colocated vLLM TP=2; verl-gr now defaults rollout TP=2 but still uses FSDP2 + Ray hybrid engine. vLLM custom all-reduce is disabled after a TP=2 startup failure, so TP topology remains aligned while the collective implementation falls back to NCCL | Backend/runtime remains different |
| 2. Compute performance | **Pending** | Batch work per optimizer step is aligned; structural overhead remains | Measure wall-clock and phase timing |
| 3. Convergence analysis | **Partial** | `fp32opt` fixed the flat-KL failure, but newer run `newmatchg2_3_trlmatch_fp32opt` shows max-length generation collapse and KL spikes | Align generation termination/length behavior with TRL before further reward conclusions |
| 5. Recommended fixes | **Partial** | confirmed-good defaults are in place through `fp32opt`; old-logprob=current, old-log-prob recompute bypass, rollout TP=2, safe checkpoint pruning, and TRL-like save/eval cadence are default-aligned before the next run | Backend/Ray differences remain |
| 6. Verification plan | **Pending** | checklist exists below | Run and record evidence |

## 1. Hyperparameter Analysis

### 1.1 Effective Batch Size: Done

Accomplished:

- Corrected the interpretation of TRL's `generation_batch_size`.
- Moved TRL-alignment defaults into `scripts/run_rankgrpo.sh`, so
  `.match_rankgrpo.sh` only keeps endpoint-specific GPU/Ray/output setup.
- Added `data.gen_batch_size=6` as unique prompts per optimizer step.
- Set actor `global_batch_size = mini_batch_size = gen_batch_size × rollout.n`
  in `verl_080_dev/verl/trainer/ppo/ray_trainer.py`, so one actor update uses
  one mini-batch and lets the engine accumulate gradients across micro-batches.
- Disabled dynamic actor micro-batching for the aligned run and set
  `ppo_micro_batch_size_per_gpu=4`, giving 6 micro-batches per optimizer step on
  2 GPUs.

Current verl-gr defaults:

```text
TRAIN_BATCH_SIZE              = 1   unique prompt per micro-batch
GRADIENT_ACCUMULATION_STEPS   = 6
GEN_BATCH_SIZE                = 1 × 6 = 6 unique prompts per optimizer step
ROLLOUT_N                     = 8
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU = 4 sequences/GPU

Total generated sequences per optimizer step = 6 × 8 = 48
```

#### TRL Batch Logic

TRL has three related but different quantities:

- `num_generations`: GRPO group size `G`. Each unique prompt is sampled `G`
  times so rewards can be normalized within the same-prompt group.
- `per_device_train_batch_size × num_processes`: generated sequence slots
  processed by one global forward/backward micro-step.
- `gradient_accumulation_steps`: number of micro-steps accumulated before
  `optimizer.step()`.

Therefore TRL's `generation_batch_size` is **not** the number of unique prompts.
It is the number of generated sequence slots consumed before one optimizer step:

```text
generation_batch_size
  = per_device_train_batch_size × num_processes × gradient_accumulation_steps
  = 4 × 2 × 6
  = 48 generated sequence slots
```

Because `num_generations=8`, those 48 slots correspond to:

```text
unique prompts per optimizer step
  = generation_batch_size / num_generations
  = 48 / 8
  = 6 unique prompts
```

In the current 2-GPU reference run, `gradient_accumulation_steps=6` happens to
equal the number of unique prompts per optimizer step. That equality is
incidental:

```text
per_device_train_batch_size × num_processes
  = 4 × 2
  = 8 generated sequence slots per micro-step
  = num_generations
```

So each micro-step contains exactly one prompt group:

```text
micro-step 1: prompt A × 8 generations
micro-step 2: prompt B × 8 generations
...
micro-step 6: prompt F × 8 generations
optimizer.step()
```

If `per_device_train_batch_size` changed to 8 while keeping
`num_processes=2`, `gradient_accumulation_steps=6`, and `num_generations=8`,
then:

```text
generation_batch_size = 8 × 2 × 6 = 96 slots
unique prompts        = 96 / 8 = 12 prompts
```

`gradient_accumulation_steps` would still be 6, but each micro-step would
contain 16 generated sequence slots = 2 unique prompts × 8 generations. This is
why `gradient_accumulation_steps` cannot be treated as the unique-prompt batch
size in general.

#### verl-gr Mapping

```text
verl-gr:
  DataLoader yields gen_batch_size=6 unique prompts
  Generate 6 × 8 = 48 completions
  _update_actor:
    global_batch_size = mini_batch_size = 48
    → train_mini_batch creates 1 mini-batch of 48 seq (24 seq/GPU)
  Engine.train_batch:
    Splits 24 seq/GPU into micro-batches of 4 seq/GPU
    → 6 micro-batches per optimizer step when ppo_epochs=1
    → 1 optimizer.step()
```

Progress-bar denominator implication:

```text
raw dataset prompts // 6 unique prompts per optimizer step
```

The latest startup log reports `Total training steps: 63804` after prompt-length
filtering, which is the denominator to expect for the current aligned launch.

The previous `gen_batch_size=48` verl-gr setting meant 48 unique prompts × 8
rollouts = 384 generated sequences per update, which was not equivalent to TRL.

Remaining verification:

- Confirm `data.gen_batch_size=6` in the Hydra dump.
- Confirm actor `global_batch_size=48` and `mini_batch_size=48` in debug logs.
- Confirm one optimizer update per 6 unique prompts.

### 1.2 PPO Clip Ratio: Done

Accomplished:

- `scripts/run_rankgrpo.sh` now defaults:

```bash
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.06}"
PPO_CLIP_RATIO_HIGH="${PPO_CLIP_RATIO_HIGH:-0.08}"
```

- These values are passed to:

```text
actor_rollout_ref.actor.clip_ratio=0.06
actor_rollout_ref.actor.clip_ratio_low=0.06
actor_rollout_ref.actor.clip_ratio_high=0.08
```

This matches TRL's effective clip range `[0.94, 1.08]`.

Remaining verification:

- Monitor `actor/pg_clipfrac`; it should not immediately saturate under the
  aligned default.

### 1.3 PPO Epochs / Sequence Reuse: Done

Accomplished:

- `scripts/run_rankgrpo.sh` now defaults `PPO_EPOCHS=1`.
- This matches TRL's `--mu 1` / `num_iterations=1`.
- Each generated sequence is used for one forward/backward pass before the next
  optimizer step.

Remaining verification:

- Confirm launched jobs do not override `PPO_EPOCHS`.
- Compare `actor/pg_loss`, KL, and reward with one pass per generated batch.

### 1.4 Other Aligned Hyperparameters: Done

Current defaults aligned with TRL:

| Parameter | TRL | verl-gr |
|---|---|---|
| Learning rate | `1e-6` | `LEARNING_RATE=1e-6` |
| KL coefficient | `1e-3` | `KL_COEF=1e-3` |
| Adam beta1/beta2 | `0.9 / 0.99` | `0.9 / 0.99` |
| Rollouts per prompt | `num_generations=8` | `ROLLOUT_N=8` |
| Prompt/completion length | `2048 / 1024` | `2048 / 1024` |
| Save/eval frequency | `200 / 200` | `SAVE_FREQ=200`, `TEST_FREQ=SAVE_FREQ` |
| Top-k checkpoint pruning | keep best 3 by eval metric | keep best 3 by `eval/reward_total` and delete all non-kept `global_step_*` dirs after ranking |
| Seed | `3407` | `SEED=3407` |
| Train shuffle | enabled | `DATA_SHUFFLE=True` |
| Validation shuffle | disabled | `VALIDATION_SHUFFLE=False` |
| Eval before train | not explicitly requested | `VAL_BEFORE_TRAIN=False` |
| Actor train dtype | fp32 master/mixed bf16 | `ACTOR_MODEL_DTYPE=fp32` + FSDP mixed precision |

Accomplished:

- Validation now follows TRL's `--no-val_shuffle` behavior.
- The trainable actor now defaults to fp32 loading to avoid quantizing
  `lr=1e-6` AdamW updates into bf16 parameters.
- Save/eval cadence now defaults to TRL's `200 / 200`, and top-k pruning removes
  all checkpoint directories outside the kept best-3 set after each successful
  validation ranking update.
- Eval-before-train is disabled by default to avoid an extra initial validation
  pass not requested by the TRL launch script.

Confirmed in the good `fp32opt` run:

- Log shows `Train/validation shuffle: True/False`.
- Hydra dump shows actor `model_dtype: 'fp32'`.
- Hydra dump shows `ppo_epochs: 1`.
- Hydra dump shows `validation_shuffle: False`.

### 1.5 Distributed Backend Differences: Partial / Known Difference

Not aligned yet:

| Aspect | TRL | verl-gr current default |
|---|---|---|
| Training backend | DeepSpeed ZeRO-3 | FSDP2 |
| Runtime topology | Accelerate trainer process | Ray hybrid engine |
| vLLM integration | colocated | Ray rollout workers |
| vLLM tensor parallelism | TP=2 | TP=2 by default |
| vLLM all-reduce implementation | vLLM default | custom all-reduce disabled; NCCL fallback after `custom_all_reduce.cuh:455 invalid argument` on this stack |

Current decision:

- The confirmed `fp32opt` run remains the historical dtype/batch/clip baseline.
- The current aligned launch keeps rollout TP=2 to match TRL's colocated vLLM
  tensor-parallel topology, but disables vLLM custom all-reduce for stability on
  this machine.

Remaining verification:

- Compare the next successful TP=2 run against the confirmed TP=1/DP=2 `fp32opt` run under
  the same batch/clip/epoch/dtype settings.
- Compare checkpoint parameter drift and optimizer-state dtypes between TRL and
  verl-gr.

## 2. Compute Performance Analysis

### 2.1 Per-Optimizer-Step Work Breakdown: Partial

Accomplished:

- The amount of training data per optimizer step is now aligned:

```text
6 unique prompts × 8 rollouts = 48 generated sequences
```

Still different:

- TRL computes generation, old policy log-probs, and reference log-probs in a
  more colocated/inlined path.
- verl-gr still performs separate Ray phases for generation, old log-prob,
  reference log-prob, and actor update.

Current verl-gr step shape:

```text
1. vLLM rollout: 6 prompts × 8 = 48 completions
2. old_log_prob: bypassed; required field is filled from rollout_log_probs
3. ref_log_prob: separate ref forward over 48 sequences
4. actor update: 6 fixed micro-batches, 1 optimizer step
```

For the TRL-matched loss path, `algorithm.rank_grpo.old_log_prob_mode=current`
uses the current forward pass detached from gradient (`log_prob.detach()`) as
the PPO anchor, matching TRL's aligned-generation behavior when
`old_per_token_logps` is absent. The current default also sets
`algorithm.rollout_correction.bypass_mode=true`, which skips the separate
old-log-prob actor forward by copying vLLM
`rollout_log_probs` into the required `old_log_probs` batch field. In Rank-GRPO
`trl_match` mode the loss still uses `log_prob.detach()` as the PPO anchor, so
the copied `old_log_probs` field is only a compatibility/input-plumbing value.

### 2.2 Why verl-gr Is Slower: Pending

Known remaining speed differences:

- The old-log-prob actor recompute is bypassed by default; verify timing logs no
  longer include a separate `old_log_prob` phase.
- Separate `ref_log_prob` forward pass.
- Ray RPC/DataProto boundaries between phases.
- Historical confirmed good run used vLLM TP=1/DP=2; the current aligned default
  uses TP=2 to match TRL's colocated TP=2 group.
- vLLM custom all-reduce is disabled for stability, so TP=2 topology is aligned
  while the collective implementation differs from vLLM's default fast path.
- FSDP2/Ray memory layout differs from DeepSpeed ZeRO-3.

No fix is marked done for these structural speed items yet. The batch fix makes
the comparison fair, but it does not remove the extra forward/RPC overhead.

### 2.3 Timing Budget: Pending

Remaining evidence to collect:

- Wall-clock time per 100 optimizer steps.
- Rollout generation tokens/sec.
- old/ref log-prob phase time.
- Actor update phase time.
- End-to-end comparison against the TRL run after batching and dtype alignment.

## 3. Training Convergence Analysis

### 3.1 Advantage Noise / Batch-Size Mismatch: Done

Accomplished:

- Both systems now use 6 prompt groups per optimizer step.
- Both systems use 8 rollouts per prompt.
- The previous "TRL has 48 unique prompts while verl-gr has 6" concern is
  resolved; it came from treating TRL generated slots as unique prompts.

Remaining verification:

- Compare reward variance and eval reward slope after the fresh aligned run.

### 3.2 Clip Range: Done

Accomplished:

- Both systems now use the same item-level trust region, `[0.94, 1.08]`.

Remaining verification:

- Compare `actor/pg_clipfrac` and KL against TRL.

### 3.3 PPO Epochs: Done

Accomplished:

- verl-gr uses `PPO_EPOCHS=1`, matching TRL `mu=1`.
- Historical concerns about 12 repeated PPO epochs no longer apply to the
  default aligned launch.

Remaining verification:

- Confirm no launch-time override.

### 3.4 Sample Diversity and Generalization: Done

Accomplished:

- Both systems now consume 6 new unique prompts per optimizer step.
- Both systems therefore cover a dataset of `N` prompts in about `N / 6`
  optimizer steps.

Remaining verification:

- Confirm future runs keep the same filtered dataloader denominator unless the
  dataset or prompt-length filtering changes.

### 3.5 KL Divergence Dynamics: Partial

Accomplished:

- Batch size, clip range, PPO epochs, shuffle behavior, and actor train dtype
  have been aligned.
- `ACTOR_MODEL_DTYPE=fp32` is now the default, addressing the observed flat
  `actor/kl_loss` failure mode caused by bf16 actor parameters and bf16 AdamW
  moments at `lr=1e-6`.
- The good run `g2_3_trlmatch_ppoegradaccu6_trainshuffleOn_fp32opt` confirms
  KL is no longer flat: `actor/kl_loss` went from `0.000167` at step 10 to
  `0.020529` at step 360. The TRL reference was `0.000064` at step 10 and
  `0.021844` at step 360.
- The newer run `newmatchg2_3_trlmatch_fp32opt` confirms the actor learning-rate
  path itself is not dynamically drifting:

```text
actor/lr      min=max=9.999999974752427e-07
actor/base_lr min=max=9.999999974752427e-07
```

Backend code check:

- `scripts/run_rankgrpo.sh` passes
  `actor_rollout_ref.actor.optim.lr_scheduler_type=constant`.
- verl FSDP builds `get_constant_schedule_with_warmup(...)`.
- With `lr_warmup_steps=0`, the constant schedule returns multiplier `1.0` at
  every step.

Still open:

- The latest failure is no longer "flat KL"; it is a generation behavior
  divergence. In `newmatchg2_3_trlmatch_fp32opt`, KL spikes coincide with
  completions hitting `max_response_length=1024`.
- TRL's comparable trace keeps completions near 187 tokens and almost never
  clips, so `eval/reward_total` is not aligned while verl-gr is producing
  max-length / non-terminating generations.
- Remaining backend differences still include generation termination semantics,
  old/ref log-prob plumbing, vLLM topology, FSDP2 vs ZeRO-3, and Ray boundaries.

Next checks:

- Keep `fp32opt` as the baseline for dtype/batch/clip/epoch alignment, but do
  not mark convergence fully aligned until generation length behavior matches
  TRL.
- Prioritize validating generation termination / length behavior under the
  current aligned defaults before changing other backend/runtime knobs.
- Compare reward, clipped-completion ratio, detected item count, overflow ratio,
  and checkpoint drift at the same steps.

### 3.6 Generation Termination / Length Collapse: Next Alignment Target

#### Problem Evidence

Latest failing verl-gr trace:

```text
Trace:
tensorboard_log/RankGRPO/newmatchg2_3_trlmatch_fp32opt

actor/lr:
  constant 1e-6 for all logged steps

Before the failure:
  step 2800 response_length/mean       = 201.75
  step 2800 response_length/clip_ratio = 0.0
  step 2800 actor/kl_loss              = 0.2168

At first collapse:
  step 3000 response_length/mean       = 1016.40
  step 3000 response_length/clip_ratio = 0.9583
  step 3000 actor/kl_loss              = 0.8666
  step 3000 eval/reward_total          = 0.4520

At major KL/reward failure:
  step 5000 response_length/mean       = 1012.58
  step 5000 response_length/clip_ratio = 0.9792
  step 5000 actor/kl_loss              = 3.4241
  step 5000 eval/reward_total          = 0.3363

Peak KL window:
  step 5020 actor/kl_loss              = 3.6645
  step 5020 response_length/clip_ratio = 1.0
  step 5430 actor/kl_loss              = 3.8333
```

Comparable TRL trace:

```text
Trace:
Rank-GRPO/results/grpo/new2/runs/May28_09-35-22_hk01dgx028

train/learning_rate:
  constant 1e-6

train/completions/mean_length:
  min 183.40, max 194.08, final about 188

train/completions/clipped_ratio:
  min 0.0, max 0.00417, final 0.0

eval/reward_total:
  range 0.4685 to 0.4891
```

Working hypothesis:

```text
The latest verl-gr run fails because generated completions stop terminating
normally and saturate the 1024-token response budget. This inflates KL, makes
loss/eval behavior non-comparable to TRL, and prevents reward convergence.
Learning-rate drift is not the root cause for this trace.
```

#### Code-Level Alignment Targets

The next implementation must align these code/design surfaces, in this order:

1. **Training length metrics parity: Implemented**

   Add TRL-equivalent training metrics in verl-gr so every run can detect this
   failure without manual scalar scripts.

   Files:

   - `verl_gr/recipes/rankgrpo/rankgrpo_algorithm.py`
   - `verl_gr/trainers/rl_trainer.py`
   - `tests/test_rankgrpo_loss_modes.py`

   Required metrics:

   ```text
   train/rankgrpo/completions/mean_length
   train/rankgrpo/completions/min_length
   train/rankgrpo/completions/max_length
   train/rankgrpo/completions/clipped_ratio
   train/rankgrpo/completions/mean_terminated_length
   train/rankgrpo/completions/min_terminated_length
   train/rankgrpo/completions/max_terminated_length
   train/rankgrpo/items/detected_mean
   train/rankgrpo/items/detected_max
   train/rankgrpo/items/overflow_token_ratio
   train/rankgrpo/items/eos_rate
   ```

   Semantics must match TRL where names overlap:

   - `clipped_ratio = fraction of completions without EOS before max length`.
   - `terminated_length` statistics only include completions with EOS; if no
     completion has EOS, log `0.0` for terminated length statistics.
   - `items/detected` must use the same newline-token segmentation semantics as
     TRL's `_segment_items_from_tokens(...)`.

2. **Generated sample dump around failure windows**

   Make it easy to inspect what the model is actually generating at the collapse
   steps.

   Files:

   - `verl_gr/recipes/rankgrpo/rankgrpo_trainer.py`
   - Existing dump/log-generation helpers if available in `verl_gr/trainers/rl_trainer.py`

   Required behavior:

   ```text
   When VERL_GR_DEBUG=1 or an explicit dump flag is set:
     dump prompts/completions/reward_model/rank_rewards/items_detected/eos_found
     at selected train steps, especially 2800, 3000, 5000, 5400
   ```

   The dump should answer:

   - Are completions repeating text, movie IDs, separators, or prompt fragments?
   - Are newline separators still present?
   - Is EOS missing, malformed, or masked out?
   - Are more than 20 recommendation items emitted?

3. **Sampling parameter parity: Implemented for Rank-GRPO fast path**

   Confirm the actual vLLM requests match TRL, not only the Hydra config.

   TRL source target:

   - `Rank-GRPO/libs/trl/rank_grpo_trainer.py`

   TRL vLLM generation uses:

   ```text
   SamplingParams(
     n=1,
     repetition_penalty=1.0,
     temperature=1.0,
     top_p=1.0,
     top_k=-1,
     max_tokens=1024,
   )
   ```

   verl-gr source targets:

   - `verl_gr/recipes/rankgrpo/rankgrpo_agent_loop.py`
   - `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
   - `configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml`

   Required checks:

   ```text
   config.temperature = 1.0
   config.top_p = 1.0
   config.top_k = -1
   request min_p = 0.0
   request n = 1
   repetition_penalty = 1.0
   max_tokens resolves to response_length=1024
   vLLM EOS behavior is not changed by ignore_eos or stop settings
   generated token ids include tokenizer.eos_token_id when the model terminates
   ```

   Implemented in `verl_gr/recipes/rankgrpo/rankgrpo_agent_loop.py`:

   ```text
   n=1
   repetition_penalty=1.0
   temperature=1.0
   top_p=1.0
   top_k=-1
   min_p=0.0
   max_tokens=response_length
   ```

   Compatibility note:

   - `min_p` is intentionally not passed as `actor_rollout_ref.rollout.min_p` in
     Hydra, because this verl version's `RolloutConfig` rejects that field.
   - The Rank-GRPO agent loop still injects `min_p=0.0` into actual vLLM
     sampling params, preserving TRL-aligned sampling behavior without breaking
     rollout config instantiation.
   - `actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce`
     defaults to `true` after TP=2 vLLM startup failed with
     `custom_all_reduce.cuh:455 invalid argument`. This keeps TP=2 aligned with
     TRL while using NCCL fallback collectives.

4. **EOS and masking parity: Implemented in agent loop**

   TRL masks after first EOS:

   ```text
   is_eos = completion_ids == eos_token_id
   eos_idx = first EOS position or sequence length
   completion_mask = sequence_indices <= eos_idx
   ```

   verl-gr now applies the same mask when building rollout `response_mask` in
   `verl_gr/recipes/rankgrpo/rankgrpo_agent_loop.py` via
   `build_trl_completion_mask()` (`verl.utils.torch_functional.get_response_mask`).

   Remaining checks on the next run:

   - `train/rankgrpo/completions/clipped_ratio` stays near TRL (~0).
   - If vLLM strips EOS from returned token IDs, treat as non-terminated and
     investigate tokenizer/vLLM return semantics explicitly.

5. **Length-shaping parity**

   TRL applies length shaping after item segmentation:

   ```text
   overflow tokens after item 20 receive extra_token_penalty = -0.1
   exact 20-item EOS receives end_of_list_reward = 0.1
   early EOS before 20 items receives early_stop_penalty = -0.1
   ```

   verl-gr target:

   - Confirm `rank_token_mask`, `item_token_mask`, `overflow_token_mask`,
     `exact_len`, and `early_stop` match TRL on the same synthetic completions.
   - Do not add a separate no-EOS penalty by default because TRL does not have a
     `non_terminated_penalty` hyperparameter. Keep the active reward semantics
     aligned with TRL while fixing sampling/EOS behavior at the source.
   - Add tests with:
     - exactly 20 newline-separated items followed by EOS,
     - fewer than 20 items followed by EOS,
     - more than 20 items with overflow tokens,
     - no EOS and max-length padding/clipping.

#### Execution Plan for Next Run

The observability and sampling/old-log-prob alignment code has been implemented.
The next step is a fresh run that proves startup succeeds and generated lengths
stay near TRL.

1. Launch with the current defaults from `.match_rankgrpo.sh` /
   `scripts/run_rankgrpo.sh`.
2. Confirm the startup log shows TP=2, `disable_custom_all_reduce=True`,
   `VAL_BEFORE_TRAIN=False`, and total training steps around the filtered
   dataloader count (`63804` in the latest startup log).
3. Confirm `train/rankgrpo/completions/clipped_ratio` and
   `train/rankgrpo/completions/mean_length` are logged.
4. Stop early if clipped ratio moves toward the previous failure mode
   (`~1.0`) instead of TRL's near-zero clipped ratio.
5. If convergence still differs after length behavior is aligned, test
   remaining backend differences separately: ref-log-prob phase cost, FSDP2/Ray
   vs ZeRO-3, and Ray RPC overhead.

## 4. Root Causes Summary

### 4.1 Primary Convergence Causes

| Cause | Status | What changed | Remaining work |
|---|---|---|---|
| Unique prompts per step | **Done** | `GEN_BATCH_SIZE=6`; actor global/mini batch = 48 seq; latest startup log reports `Total training steps: 63804` after filtering | Keep defaults |
| Clip epsilon | **Done** | `0.06 / 0.08` defaults | Monitor clipfrac |
| PPO epochs | **Done** | `PPO_EPOCHS=1` default | Confirm no override |
| Actor update precision | **Done for KL** | `ACTOR_MODEL_DTYPE=fp32` default; `fp32opt` run shows KL growth comparable to TRL | Continue reward/drift checks |
| Generation termination / length collapse | **Partial / EOS mask in agent loop** | fast-path sampling aligned; agent loop applies TRL `completion_mask` via `build_trl_completion_mask` (`rankgrpo_agent_loop.py`); prior `newmatchg2_3` late max-length collapse predates this fix | Next run: verify `train/rankgrpo/completions/clipped_ratio` stays near TRL (~0). If clip ratio still climbs late in training, the remaining gap is likely generation not stopping (vLLM/model/stack), not mask plumbing |

### 4.2 Secondary Speed Causes

| Cause | Status | Current state |
|---|---|---|
| Separate forward passes | **Pending** | loss anchors to current detached log-probs, but backend old/ref log-prob work may still exist |
| vLLM TP topology | **Partial** | default uses TP=2; latest attempt reached vLLM startup but needed custom all-reduce disabled |
| Distributed backend | **Pending** | FSDP2/Ray remains different from ZeRO-3 |
| Ray RPC overhead | **Pending** | not optimized yet |

## 5. Recommended Fixes from Target Doc

### Already Applied

- Correct unique-prompt batch size: `GEN_BATCH_SIZE=6`.
- Fixed micro-batching for gradient accumulation:
  `ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=4`.
- Single actor mini-batch per update:
  `global_batch_size = mini_batch_size = gen_batch_size × rollout.n`.
- Tight TRL clip range: `0.06 / 0.08`.
- Single pass per generated sequence: `PPO_EPOCHS=1`.
- Train shuffle enabled and validation shuffle disabled.
- fp32 actor load for trainable parameters: `ACTOR_MODEL_DTYPE=fp32`.
- Save/eval frequency aligned to `200 / 200`.
- Top-k checkpoint pruning now deletes all `global_step_*` dirs outside the
  kept best-3 set, including previously unranked save-only checkpoints.
- Eval-before-train disabled by default: `VAL_BEFORE_TRAIN=False`.
- `old_log_prob_mode=current` wired in `verl_gr/recipes/rankgrpo/rankgrpo_loss.py`;
  in `trl_match` mode it resolves `old_log_probs` to `log_prob.detach()`,
  matching TRL's aligned-generation fallback when `old_per_token_logps` is
  absent.
- `RANKGRPO_BYPASS_OLD_LOG_PROB=True` by default, so verl-gr no longer runs the
  separate actor old-log-prob recompute before the Rank-GRPO update.
- `ROLLOUT_TENSOR_PARALLEL_SIZE=2` is now the script default for 2-GPU runs.
- vLLM custom all-reduce is disabled by default so TP=2 startup falls back to
  NCCL instead of failing on `custom_all_reduce.cuh:455 invalid argument`.
- `min_p=0.0` remains in the Rank-GRPO agent-loop sampling params, but is no
  longer passed as a Hydra `RolloutConfig` field.

### Not Applied Yet

- None from the current high-priority TRL-alignment list. Remaining differences
  are backend/runtime differences rather than simple launch defaults.

## 6. Verification Plan

Fresh aligned run checklist:

- [x] Hydra/logs show `data.gen_batch_size=6`.
- [x] Hydra dump shows `actor_rollout_ref.actor.ppo_epochs=1`.
- [x] Hydra dump shows clip low/high `0.06 / 0.08`.
- [x] Hydra dump shows `actor_rollout_ref.actor.fsdp_config.model_dtype=fp32`.
- [x] Logs show total actor batch of 48 generated sequences per optimizer step.
- [x] Logs show 6 fixed micro-batches of 4 seq/GPU per optimizer step.
- [x] Save/test frequency defaults are aligned to `200 / 200`.
- [x] Top-k pruning deletes unkept checkpoint dirs after successful validation
  ranking.
- [x] Latest startup log reports `Total training steps: 63804` with
  `GEN_BATCH_SIZE=6` after dataset filtering.
- [x] TP=2 is the default rollout topology.
- [x] vLLM custom all-reduce is disabled to keep TP=2 stable on this stack.
- [x] `min_p` is not passed through Hydra `RolloutConfig`, avoiding the previous
  `unexpected keyword argument 'min_p'` startup failure.
- [x] Old-log-prob actor recompute is bypassed by default while Rank-GRPO loss
  keeps TRL's `log_prob.detach()` anchor.
- [x] `actor/kl_loss` grows comparably to TRL `train/kl` in the good `fp32opt`
  run.
- [ ] `train/rankgrpo/completions/clipped_ratio` exists and stays near TRL's
  `train/completions/clipped_ratio`.
- [ ] `train/rankgrpo/completions/mean_length` exists and stays near TRL's
  `train/completions/mean_length` under the same checkpoint segment.
- [ ] `train/rankgrpo/items/detected_mean`, `items/overflow_token_ratio`, and
  `items/eos_rate` exist and identify whether max-length outputs are caused by
  missing EOS, excessive item overflow, or malformed separators.
- [ ] Debug sample dumps around steps 2800/3000/5000/5400 show actual
  completions, detected items, EOS status, and rank rewards.
- [ ] Checkpoint parameter drift is no longer bf16-quantized away.
- [ ] `eval/reward_total` slope is compared against the TRL baseline.
- [ ] Wall-clock time per 100 optimizer steps is recorded.
- [ ] If convergence still differs beyond KL, revisit old-log-prob plumbing only
  if logs show bypass/current anchoring is not being applied as expected.
- [ ] If speed or generation behavior still differs under TP=2, compare TP=2
  against the historical TP=1/DP=2 `fp32opt` baseline.

## Implementation Inventory

| File | Current role |
|---|---|
| `scripts/run_rankgrpo.sh` | Owns alignment defaults and passes Hydra overrides |
| `scripts/.match_rankgrpo.sh` | Endpoint-specific GPU/Ray/output/resume wrapper; cleanup is scoped to this run's GCS port and Ray temp dir, not arbitrary worker-port listeners |
| `configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml` | RankGRPO base config; actor dtype default is fp32, TP=2 is default, and vLLM custom all-reduce is disabled |
| `verl_080_dev/verl/trainer/ppo/ray_trainer.py` | Actor update uses `gen_batch_size × rollout.n` and one full mini-batch |
| `verl_gr/recipes/rankgrpo/rankgrpo_loss.py` | TRL-matched RankGRPO PPO loss path |
| `verl_gr/recipes/rankgrpo/rankgrpo_algorithm.py` | Per-prompt-group RankGRPO advantage computation |
