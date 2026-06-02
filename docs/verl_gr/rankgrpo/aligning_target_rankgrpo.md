# Aligning verl-gr RankGRPO with TRL: Root Cause Analysis

## Overview

This document provides a detailed comparison between two implementations of RankGRPO training for Qwen2.5-0.5B-Instruct:

| | **TRL (Reference)** | **verl-gr (Target)** |
|---|---|---|
| **Entry script** | `Rank-GRPO/scripts/run_rl.sh` | `verl-gr-fork-workingbranch/scripts/.match_rankgrpo.sh` |
| **Trainer** | `trl/trainer/rank_grpo_trainer.py` (installed in `rank-grpo` conda env) | `verl_gr/recipes/rankgrpo/rankgrpo_trainer.py` + verl core `verl_080_dev/verl/trainer/ppo/ray_trainer.py` |
| **Loss function** | `_compute_loss` in `rank_grpo_trainer.py` | `rankgrpo_ppo_loss` in `verl_gr/recipes/rankgrpo/rankgrpo_loss.py` |
| **Advantage computation** | `_generate_and_score_completions` in `rank_grpo_trainer.py` | `compute_rank_grpo_advantage` in `verl_gr/recipes/rankgrpo/rankgrpo_algorithm.py` |
| **Conda environment** | `rank-grpo` | `verl_080_fromscratch` + `verl_080_vllm_015` |
| **Model** | Qwen2.5-0.5B-Instruct | Qwen2.5-0.5B-Instruct |

The verl-gr implementation is **slower per unit of training progress** and **converges more slowly** (lower validation reward per step) compared to the TRL reference. This document analyzes the root causes across three dimensions: hyperparameter alignment, compute performance, and training convergence dynamics.

---

## 1. Hyperparameter Analysis

### 1.1 Effective Batch Size (Unique Prompts per Optimizer Step)

**Status: finished / aligned as of 2026-05-27.** Earlier revisions of this
document treated TRL's `generation_batch_size=48` as 48 unique prompts. That was
incorrect for the current TRL RankGRPO code path. TRL counts repeated generation
slots here; with `num_generations=8`, 48 slots correspond to 6 unique prompts.

**TRL:**

```
per_device_train_batch_size      = 4   (prompts per GPU per micro-batch)
num_processes                    = 2   (GPUs)
gradient_accumulation_steps      = 6
num_generations (rollouts/prompt)= 8

generation_batch_size             = 4 × 2 × 6 = 48 repeated generation slots
Unique prompts per optimizer step = 48 / 8 = 6
Total sequences per optimizer step = 6 × 8 = 48
Sequences per micro-batch          = 48 / 6 = 8 global = 4/GPU
```

The dataloader is controlled by `RepeatSampler` at [rank_grpo_trainer.py:1056-1090]:

```python
RepeatSampler(
    data_source=dataset,
    mini_repeat_count=self.num_generations,       # 8
    batch_size=self.args.generation_batch_size // self.num_generations,  # 48//8 = 6
    repeat_count=self.num_iterations * self.args.steps_per_generation,   # 1 × 6 = 6
)
```

Where `steps_per_generation` defaults to `gradient_accumulation_steps=6` (see `grpo_config.py:590-591`), and `generation_batch_size = per_device_train_batch_size × num_processes × steps_per_generation = 4 × 2 × 6 = 48`.

The generation batch contains 6 unique prompts × 8 rollouts = 48 generated
sequences. It is split into 6 micro-batches of 8 sequences each. After 6
gradient-accumulation micro-batches, one optimizer update is applied.

**verl-gr current status (2026-05-27): aligned for 1.1**

`scripts/run_rankgrpo.sh` now owns the TRL-alignment defaults so every wrapper
gets the same behavior:

```
TRAIN_BATCH_SIZE              = 1   (unique prompts per micro-batch)
GRADIENT_ACCUMULATION_STEPS   = 6
GEN_BATCH_SIZE                = 1 × 6 = 6
ROLLOUT_N (n)                 = 8
ppo_mini_batch_size           = 1
ppo_micro_batch_size_per_gpu  = 4
use_dynamic_bsz               = False when GRADIENT_ACCUMULATION_STEPS > 1

Unique prompts per optimizer step = 6
Total sequences per optimizer step = 6 × 8 = 48
Sequences per micro-batch          = 8 total = 4/GPU
```

`scripts/run_rankgrpo.sh` computes:

```bash
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=6
GEN_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))  # 6
```

and passes `++data.gen_batch_size="${GEN_BATCH_SIZE}"` to Hydra. When
`GRADIENT_ACCUMULATION_STEPS > 1`, it forces fixed micro-batching:

```bash
USE_DYNAMIC_BSZ=False
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=$((TRAIN_BATCH_SIZE * ROLLOUT_N / N_GPUS))  # 4
```

Inside `_update_actor`, the current verl patch uses `gen_batch_size × rollout.n` as the global actor batch and sets the mini-batch equal to that full global batch:

```python
gen_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
global_batch_size = gen_batch_size * self.config.actor_rollout_ref.rollout.n  # 6 × 8 = 48
tu.assign_non_tensor(
    batch_td,
    global_batch_size=global_batch_size,
    mini_batch_size=global_batch_size,
    ...
)
```

This creates one actor mini-batch containing all 48 sequences. The FSDP engine
then splits that mini-batch into fixed micro-batches of 4 sequences/GPU,
producing exactly 6 gradient-accumulation micro-batches before
`optimizer.step()`.

**Conclusion for 1.1:** verl-gr is now aligned with TRL on batching and
optimizer-update behavior: both use 6 unique prompts, 8 rollouts per prompt, 48
generated sequences, 6 accumulation micro-batches, and 1 optimizer step.

#### Why This Matters for GRPO

GRPO normalizes advantages per-prompt-group. In `rankgrpo_algorithm.py:138-145`:

```python
for indices in uid_to_indices.values():  # each uid = one prompt
    group_rewards = rank_rewards.index_select(0, idx_tensor)
    centered = group_rewards - group_rewards.mean(dim=0)
    if normalize_by_std:
        centered = centered / (std + 1e-4)
```

- **TRL**: 6 prompt groups × 8 rollouts each → group mean/std estimated from 8 samples.
- **verl-gr current default**: 6 prompt groups × 8 rollouts each → same group count and rollout count as TRL.

The previous "48 vs 6 prompt groups" concern is resolved because it came from
misinterpreting TRL's `generation_batch_size` as unique prompts.

### 1.2 PPO Clip Ratio — Resolved

This was a hyperparameter misalignment in earlier verl-gr runs. It is now
aligned in the default `run_rankgrpo.sh` launch path.

| Parameter | TRL | verl-gr | Ratio |
|---|---|---|---|
| `epsilon` (clip low) | **0.06** | **0.06** | 1× |
| `epsilon_high` (clip high) | **0.08** | **0.08** | 1× |
| Effective clip range | **[0.94, 1.08]** | **[0.94, 1.08]** | — |

**TRL** — defined in `train_rank_grpo.py:308-309`:

```python
epsilon=0.06,
epsilon_high=0.08,
```

Used in `_compute_loss` at [rank_grpo_trainer.py:2103-2104]:

```python
coef_1 = torch.exp(log_importance_weights)
coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
# clamp to [0.94, 1.08]
```

**verl-gr** — aligned in `run_rankgrpo.sh` as of 2026-05-27:

```bash
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.06}"
PPO_CLIP_RATIO_HIGH="${PPO_CLIP_RATIO_HIGH:-0.08}"
```

Passed via CLI from `run_rankgrpo.sh`:

```bash
actor_rollout_ref.actor.clip_ratio="${PPO_CLIP_RATIO}"           # 0.06
actor_rollout_ref.actor.clip_ratio_low="${PPO_CLIP_RATIO}"       # 0.06
actor_rollout_ref.actor.clip_ratio_high="${PPO_CLIP_RATIO_HIGH}" # 0.08
```

Used in `rankgrpo_loss.py:93-102` (trl_match path):

```python
coef_1 = torch.exp(log_importance_weights)
coef_2 = torch.clamp(coef_1, 1 - clip_ratio_low, 1 + clip_ratio_high)
# clamp to [0.94, 1.08]
```

#### Impact

The clip ratio defines a trust region around the frozen rollout policy π_old.
This mismatch is now resolved for the default launch path: both TRL and verl-gr
use `[0.94, 1.08]`.

- **TRL's tight clip**: Each optimizer step makes small, conservative policy changes. Even with imperfect advantage estimates, the damage per step is bounded. Trade-off: requires more steps to move the policy a given distance.
- **verl's current default**: Uses the same tight clip range as TRL.

The previous wide-clip explanation applies only to historical runs launched
before this default was corrected.

### 1.3 PPO Epochs (Sequence Reuse Strategy)

| Parameter | TRL | verl-gr |
|---|---|---|
| Reuse strategy | `mu=1` (`num_iterations`) | `ppo_epochs=1` |
| Passes per generated sequence | 1 forward+backward | 1 forward+backward |
| Generation cadence | Once per `steps_per_generation=6` micro-batches | Once per global step |

**TRL** — `mu=1` at `run_rl.sh:26`:

```
--mu 1
```

means `num_iterations=1`. Every generated sequence is used for exactly one forward+backward pass. New completions are generated after `steps_per_generation × num_iterations = 6 × 1 = 6` micro-batches (one full optimizer step).

**verl-gr** — `ppo_epochs=1` via `run_rankgrpo.sh`:

```bash
PPO_EPOCHS="${PPO_EPOCHS:-1}"
actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}"
```

The same 48 completions are used for one PPO epoch. This is controlled at [ray_trainer.py:1208-1219]:

```python
ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs  # 1
# ...
tu.assign_non_tensor(
    batch_td,
    global_batch_size=ppo_mini_batch_size,
    mini_batch_size=ppo_mini_batch_size,
    epochs=ppo_epochs,  # 1
    # ...
)
```

#### Status

The earlier `ppo_epochs=12` concern is resolved for the current aligned default.
verl-gr now uses one pass per generated sequence, matching TRL's `mu=1`
optimizer-update behavior. Historical runs with `ppo_epochs=12` can still show
the gradient-starvation/wasted-compute pattern described below, but that is no
longer the default alignment target.

### 1.4 Aligned Hyperparameters

The following are correctly aligned between both implementations:

| Parameter | TRL | verl-gr | Source (TRL / verl) |
|---|---|---|---|
| Learning rate | 1e-6 | 1e-6 | `run_rl.sh:27` / `run_rankgrpo.sh:93` |
| KL coefficient | 1e-3 | 1e-3 | `run_rl.sh:28` / `run_rankgrpo.sh:82` |
| KL loss type | k3 estimator | `low_var_kl` (same) | implicit / `run_rankgrpo.sh:85` |
| Adam β₁ | 0.9 | 0.9 | `run_rl.sh:29` / `run_rankgrpo.sh:95` |
| Adam β₂ | 0.99 | 0.99 | `run_rl.sh:30` / `run_rankgrpo.sh:96` |
| Weight decay | 0.0 | 0.0 | default / `run_rankgrpo.sh:97` |
| LR schedule | constant | constant | `train_rank_grpo.py:296` / `run_rankgrpo.sh:93` |
| Loss aggregation | seq-mean-token-mean | `seq-mean-token-mean` | default / `run_rankgrpo.sh:88` |
| Max prompt length | 2048 | 2048 | `run_rl.sh:47` / config yaml |
| Max completion length | 1024 | 1024 | `run_rl.sh:48` / config yaml |
| Rollouts per prompt (n) | 8 | 8 | `run_rl.sh:49` / `run_rankgrpo.sh:45` |
| rec_num | 20 | 20 | `train_rank_grpo.py:275` / `run_rankgrpo.sh:46` |
| Importance sampling | `item` | `item` | `train_rank_grpo.py:304` / config yaml |
| Reward function | `exp_inf` | same logic in `rankgrpo_reward.py` | `run_rl.sh:25` / recipe |
| Seed | 3407 | 3407 | `run_rl.sh:50` / `run_rankgrpo.sh:107` |
| Gradient checkpointing | enabled | enabled | `run_rl.sh:43` / config yaml |
| Remove padding | implicit (HF) | enabled | — / `run_rankgrpo.sh:76` |
| vLLM GPU memory util | 0.25 | 0.25 | `run_rl.sh:45` / config yaml |
| Entropy coefficient | 0.0 | 0.0 | TRL default / config yaml:60 |
| Train dataset shuffle | True | True | `GRPOConfig.shuffle_dataset` default / `run_rankgrpo.sh` |
| Validation dataset shuffle | False | False | `run_rl.sh --no-val_shuffle` / `VALIDATION_SHUFFLE=False` |
| Actor training parameter dtype | fp32 master/mixed bf16 | fp32 actor load + FSDP mixed bf16 | DeepSpeed bf16 / `ACTOR_MODEL_DTYPE=fp32` |

### 1.5 Distributed Backend Differences

| Aspect | TRL | verl-gr |
|---|---|---|
| Strategy | **DeepSpeed ZeRO-3** | **FSDP2** |
| Accelerate config | `configs/qwen25_0.5b_grpo.yaml` | N/A (Ray-based) |
| vLLM integration | **Colocated** (same process) | **Hybrid engine** (separate Ray actors) |
| vLLM TP size | **2** (both GPUs in one TP group) | **1** in the confirmed good `fp32opt` run (DP=2 rollout workers) |

**TRL** (`qwen25_0.5b_grpo.yaml`):

```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 3
  gradient_accumulation_steps: 6
mixed_precision: bf16
```

DeepSpeed ZeRO-3 partitions optimizer states, gradients, and parameters across GPUs — more memory-efficient than FSDP2 for small models. It allows larger micro-batch sizes or leaves headroom for vLLM.

Colocated vLLM (`vllm_mode=colocate`): vLLM runs in the same process as the training model, sharing GPU memory directly. Weight updates are loaded via `_move_model_to_vllm()` at [rank_grpo_trainer.py:1558-1560]:

```python
if self.state.global_step != self._last_loaded_step:
    self._move_model_to_vllm()
    self._last_loaded_step = self.state.global_step
```

With TP=2, both GPUs form one vLLM instance, generating completions cooperatively. All prompts across GPUs are gathered, generated jointly, then scattered back.

**verl-gr**: Uses FSDP2 (PyTorch native) via the `fsdp2` strategy flag. The hybrid engine runs vLLM, actor, and reference policy as separate Ray actors communicating via RPC. The confirmed good `fp32opt` run kept `ROLLOUT_TENSOR_PARALLEL_SIZE=1`, so rollout topology remains a known backend difference from TRL.

---

## 2. Compute Performance Analysis

### 2.1 Per-Optimizer-Step Work Breakdown

**TRL** — one optimizer update (6 micro-batches, 48 seq total):

```
Phase 1: Generation (once per 6 micro-batches)
├── vLLM.generate on 6 unique prompts × 8 = 48 completions (TP=2, colocated)
├── _get_per_token_logps_and_entropies on policy model (forward, 48 seq)
│   └── old_per_token_logps computed inline, no separate pass
└── _get_per_token_logps_and_entropies on ref model (forward, 48 seq)
    └── ref_per_token_logps computed inline

Phase 2: Training (6× micro-batches, 8 seq each)
├── _compute_loss (forward + backward on model, 8 seq)
│   ├── Per-token log probs on current model (mini-batch of 8 seq)
│   ├── Item-level importance weights
│   ├── Clipped PG loss with coef_1/coef_2
│   ├── KL divergence (from stored ref_per_token_logps)
│   └── Backward pass
└── Optimizer step (after 6 accumulations)

Total per optimizer step:
  1 generate (48 seq)
  + 1 policy log_prob forward (48 seq, during generation)
  + 1 ref log_prob forward (48 seq, during generation)
  + 6 train forward+backward (8 seq each)
```

Key efficiency: old_log_probs and ref_log_probs are computed **during generation** in a single consolidated pass — no separate forward passes needed.

**verl-gr** — one optimizer update (1 step, 48 seq):

```
Phase 1: Generation (every step)
└── async_rollout_manager.generate_sequences on 6 prompts × 8 = 48 completions
    └── vLLM generate (TP=1, DP=2, hybrid engine via Ray RPC)

Phase 2: old_log_prob (separate forward pass, every step)
└── _compute_old_log_prob → actor_rollout_wg.compute_log_prob (Ray RPC)
    └── Full forward pass through actor model on 48 seq

Phase 3: ref_log_prob (separate forward pass, every step)
└── _compute_ref_log_prob → ref_policy_wg.compute_ref_log_prob (Ray RPC)
    └── Full forward pass through ref model on 48 seq

Phase 4: Training (1 epoch, 48 seq)
├── _update_actor → actor_rollout_wg.update_actor (Ray RPC)
│   └── train_mini_batch × 1 epoch on 48 seq
│       ├── Forward + backward through actor
│       ├── rankgrpo_ppo_loss (trl_match path)
│       │   ├── _compute_item_mean_log_ratio (scatter/gather on GPU)
│       │   ├── kl_penalty(k3 estimator)
│       │   └── _trl_clipped_pg_loss
│       └── Optimizer step (no accumulation, single mini-batch)

Total per optimizer step:
  1 small generate (48 seq)
  + 1 old_log_prob forward (48 seq, separate RPC call)
  + 1 ref_log_prob forward (48 seq, separate RPC call)
  + 6 accumulated train micro-batches (4 seq/GPU each)
```

### 2.2 Why verl-gr Is Slower Per Unit of Training Progress

#### 2.2.1 Forward Pass Overhead

verl-gr performs **3 separate forward passes** through the model per step:

1. **vLLM generation** (forward pass through vLLM for sampling)
2. **old_log_prob computation** (forward pass through actor, `_compute_old_log_prob` at [ray_trainer.py:1163-1189])
3. **ref_log_prob computation** (forward pass through ref policy, `_compute_ref_log_prob` at [ray_trainer.py:1139-1161])

Each of these involves:
- Ray RPC serialization/deserialization of the batch (`DataProto.to_tensordict()` → `left_right_2_no_padding()` → RPC → compute → `no_padding_2_padding()` → `DataProto.from_tensordict()`)
- CUDA kernel launch overhead
- Memory allocation and deallocation for intermediate tensors

In contrast, TRL computes old_log_probs and ref_log_probs **inline during generation** (lines 1754-1804 of `rank_grpo_trainer.py`), amortizing the forward pass cost.

#### 2.2.2 ppo_epochs alignment status

The previous `ppo_epochs=12` compute-waste issue is resolved in the current
default: both TRL and verl-gr use one pass over the generated sequences per
optimizer update. If a historical run overrides `PPO_EPOCHS=12`, the old
analysis still applies: repeated passes can hit the clip boundary and produce
diminishing or zero gradient.

At the boundary:

```python
# rankgrpo_loss.py:96-98
pg1 = coef_1 * advantages      # unclipped
pg2 = coef_2 * advantages      # clipped (historical run: [0.8, 1.2])
per_token_loss = -torch.min(pg1, pg2)
# When coef_1 is outside [1-ε, 1+ε]: pg2 is always chosen, gradient = 0
```

This means extra PPO epochs can produce **diminishing or zero gradient**,
consuming GPU compute without contributing to learning. The current aligned
default avoids that by using `PPO_EPOCHS=1`.

#### 2.2.3 Ray RPC Overhead

verl-gr's hybrid engine architecture introduces RPC overhead at every boundary:

```
Driver (CPU)    ←──RPC──→    Actor Worker (GPU)    ←──RPC──→    vLLM Worker (GPU)
    │                              │                                  │
    ├─ compute_advantage           ├─ compute_log_prob                ├─ generate_sequences
    ├─ _update_actor               ├─ train_mini_batch                ├─ sleep/release
    └─ _validate                   └─ ...                            └─ ...
```

Each arrow crossing involves:
- Tensor serialization (via `DataProto`)
- Network transfer (even localhost has overhead)
- Ray task scheduling (queuing, dispatch)

TRL avoids this entirely: the trainer, model, and vLLM all run in the same process with direct memory access.

#### 2.2.4 Weight Synchronization Frequency

**TRL**: Weights are synced to vLLM only when `global_step` changes (i.e., after a full optimizer step). With `steps_per_generation=6`, this means one sync per 6 micro-batches.

```python
# rank_grpo_trainer.py:1558-1560
if self.state.global_step != self._last_loaded_step:
    self._move_model_to_vllm()
    self._last_loaded_step = self.state.global_step
```

**verl-gr**: The hybrid engine syncs weights every global step via `checkpoint_manager.update_weights(self.global_steps)`. Each sync involves loading the updated FSDP2 parameters into the vLLM model.

#### 2.2.5 vLLM Throughput

**TRL**: TP=2 means both GPUs form one vLLM instance. With `vllm_mode=colocate`, all prompts across GPUs are gathered, processed jointly, then scattered. This is efficient for generation throughput.

**verl-gr**: the confirmed good `fp32opt` run used TP=1/DP=2, so each GPU ran an independent vLLM rollout worker. This remains less aligned with TRL's TP=2 colocated generation topology and should be tested separately.

#### 2.2.6 Memory Layout

- **TRL (DeepSpeed ZeRO-3)**: Partitions parameters, gradients, and optimizer states. More GPU memory available for activations and vLLM KV cache.
- **verl-gr (FSDP2)**: FSDP2 shards parameters but the sharding granularity is coarser than ZeRO-3. Some memory pressure may exist with the hybrid engine's multiple resident models.

### 2.3 Estimated Timing Budget (2× A10 or similar GPUs, Qwen2.5-0.5B)

| Phase | TRL (per opt step) | verl-gr (per opt step) |
|---|---|---|
| vLLM generation | 48 seq, TP=2 | 48 seq, TP=1×2 in the good `fp32opt` run |
| Policy log_prob forward | 48 seq, inline | 48 seq + RPC overhead |
| Ref log_prob forward | 48 seq, inline | 48 seq + RPC overhead |
| Training F+B | 6×8 seq micro-batches | 6×4 seq/GPU micro-batches |
| **Unique prompts processed** | **6** | **6** |
| **Active structural concern** | colocated/inlined work | Ray RPC + separate log-prob passes |

> Note: Exact timings depend on GPU model, CUDA version, vLLM version, and sequence lengths. After batching alignment, this table should be treated as structural rather than a numeric timing estimate until rerun logs are collected.

---

## 3. Training Convergence Analysis

### 3.1 Advantage Noise: Resolved Batch-Size Mismatch

GRPO advantage normalization is per-prompt-group. The key code path in verl-gr at [rankgrpo_algorithm.py:130-145]:

```python
uid_to_indices: dict[Any, list[int]] = defaultdict(list)
for idx, uid in enumerate(uids):
    uid_to_indices[uid].append(idx)

rank_advantages = torch.zeros_like(rank_rewards)
for indices in uid_to_indices.values():
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=responses.device)
    group_rewards = rank_rewards.index_select(0, idx_tensor)
    centered = group_rewards - group_rewards.mean(dim=0, keepdim=True)
    if normalize_by_std:
        std = group_rewards.std(dim=0, unbiased=False, keepdim=True)
        centered = centered / (std + 1e-4)
    rank_advantages.index_copy_(0, idx_tensor, centered)
```

And the equivalent in TRL at [rank_grpo_trainer.py:1831-1843]:

```python
G = self.num_generations  # 8
Bglob = rewards_items.size(0) // G  # 6
group_means_items = rewards_items.view(Bglob, G, rec_num).mean(dim=1)
group_stds_items  = rewards_items.view(Bglob, G, rec_num).std(dim=1)
mean_rep = group_means_items.repeat_interleave(G, dim=0)
std_rep  = group_stds_items.repeat_interleave(G, dim=0)
advantages_items = rewards_items - mean_rep
if self.scale_rewards:
    advantages_items = advantages_items / (std_rep + 1e-4)
```

**Current statistical implications:**

- **Within-group variance**: Both systems have 8 rollouts per prompt, so the within-group mean/std estimation quality is identical.
- **Between-group variance**: Both systems now compute one optimizer update from 6 independent prompt groups.
- **Standard error of group mean**: The previous √(48/6) ≈ 2.8× gap was based on the incorrect assumption that TRL had 48 unique prompt groups per update.

The remaining convergence questions should therefore be attributed to other mismatches, such as rollout/log-prob plumbing, vLLM topology, or distributed backend differences.

### 3.2 Clip Range: Resolved for Current Default

The policy gradient loss in both implementations is:

```python
coef_1 = torch.exp(log_importance_weights)  # importance ratio
coef_2 = torch.clamp(coef_1, 1 - ε_low, 1 + ε_high)
per_token_loss = -torch.min(coef_1 * adv, coef_2 * adv)
```

The gradient flows through `coef_1` when it's within the clip range. When `coef_1` exceeds the clip boundary, the gradient is determined by `coef_2` (constant), giving zero gradient for the ratio.

With the batch-size mismatch resolved, both implementations receive similarly
sized advantage samples per optimizer update. The clip range is now aligned:

1. TRL clips item-level importance ratios to `[0.94, 1.08]`.
2. verl-gr now defaults to `[0.94, 1.08]`.
3. The previous `[0.8, 1.2]` behavior applies only to historical runs or explicit overrides.
4. The next comparison should evaluate reward curves, KL, and `actor/pg_clipfrac` with this aligned default.

### 3.3 ppo_epochs: Resolved for Current Default

In standard PPO, extra epochs help when the clip ratio prevents overfitting. With small ε, tokens stay within the clip window for many epochs, allowing the policy to extract more signal per batch.

The current verl-gr default uses `PPO_EPOCHS=1`, matching TRL's `mu=1`.
Historical runs with `ppo_epochs=12` may still exhibit:

- **Epochs 1-3**: Most tokens within clip window [0.8, 1.2] → full gradient, policy moves substantially
- **Epochs 4-6**: ~50% of tokens hit clip boundary → half gradient, diminishing returns
- **Epochs 7-9**: ~80% of tokens at boundary → mostly zero gradient
- **Epochs 10-12**: ~95% at boundary → negligible gradient

For the current aligned run, verify the single-epoch behavior with `actor/pg_clipfrac` and KL metrics instead of expecting across-epoch starvation.

### 3.4 Sample Diversity and Generalization

With a finite training dataset, the rate at which the model sees unique data points matters for generalization.

- **TRL**: 6 new unique prompts per optimizer step. If the training dataset has N prompts, TRL covers the dataset in about N/6 optimizer steps.
- **verl-gr**: 6 new unique prompts per optimizer step. Covers the dataset in about N/6 optimizer steps as well.

The previous 8× sample-diversity gap is resolved by the corrected batching default.

### 3.5 KL Divergence Dynamics

Both systems use the k3 KL estimator (`low_var_kl` in verl, identical to TRL's default):

```python
per_token_kl = torch.exp(ref_log_prob - log_prob) - (ref_log_prob - log_prob) - 1
```

With historical verl-gr runs that used wider clip and more epochs:
- The KL divergence per token could grow larger per step (wider clip allows more policy drift)
- 12 epochs on the same data could cause the policy to overfit to the current prompt batch
- The KL penalty (coefficient=1e-3) may be insufficient to regularize against this

For the current aligned verl-gr run, the observed flat `actor/kl_loss` was not
caused by the KL formula, clip range, or batch sizing. The actor was loaded
directly as `bfloat16`, so torch AdamW created bf16 moment tensors and applied
`lr=1e-6` updates into bf16 parameters. Checkpoint drift from step 50 to 150 was
only about `1.8e-5` relative on one FSDP shard, while the TRL checkpoint drift
over comparable early training was an order of magnitude larger. The default is
now changed to load the trainable actor in fp32 (`ACTOR_MODEL_DTYPE=fp32`) while
keeping FSDP mixed precision and rollout bf16 behavior.

---

## 4. Summary of Root Causes

### 4.1 Primary Root Causes (Convergence)

| # | Root Cause | TRL Value | verl-gr Value | Impact |
|---|---|---|---|---|
| 1 | **Clip epsilon** | 0.06/0.08 | 0.06/0.08 | **Resolved** — trust region is aligned |
| 2 | **Unique prompts/step** | 6 | 6 | **Resolved** — batching/optimizer-update behavior is aligned |
| 3 | **ppo_epochs** | 1 (mu=1) | 1 | **Resolved** — one pass per generated sequence |
| 4 | **Actor update precision** | fp32 master/mixed bf16 | fp32 actor load + mixed bf16 | **Resolved for KL** — good `fp32opt` run no longer has flat KL |

### 4.2 Secondary Root Causes (Speed)

| # | Root Cause | TRL | verl-gr | Impact |
|---|---|---|---|---|
| 4 | **Separate forward passes** | 1 consolidated | 3 separate (RPC) | Extra latency from Ray serialization and redundant computation |
| 5 | **vLLM TP configuration** | TP=2 | TP=1, DP=2 in good `fp32opt` run | Pending separate TP=2 experiment |
| 6 | **Distributed backend** | DeepSpeed ZeRO-3 | FSDP2 | ZeRO-3 has better memory efficiency for small models |
| 7 | **Ray RPC overhead** | None (colocated) | Present (hybrid engine) | Serialization, scheduling, dispatch latency at every data boundary |

### 4.3 KL Follow-Up Queue After the Confirmed fp32 Actor Run

The good `fp32opt` run starts cleanly with `ACTOR_MODEL_DTYPE=fp32` and confirms
that `actor/kl_loss` now grows in the same direction and range as TRL's
`train/kl`. If future reward, throughput, or longer-horizon KL behavior still
diverges, tackle the remaining structural/backend differences one by one:

Fresh-run update: `g2_3_trlmatch_ppoegradaccu6_trainshuffleOn_fp32opt` confirms
that the fp32 actor change fixed the flat-KL failure. `actor/kl_loss` increased
from `0.000167` at step 10 to `0.020529` at step 360. At the same optimizer
steps, the TRL reference run increased from `0.000064` to `0.021844`, so the
remaining gap is no longer "flat vs increasing" but early-step trajectory and
backend/sample-path alignment.

1. **Old/ref log-prob plumbing**: TRL computes policy and reference log-probs
   inline during generation, while verl-gr runs separate `old_log_prob` and
   `ref_log_prob` forward passes through Ray workers. If further convergence
   gaps remain, test TRL's aligned-generation old-logprob behavior separately:
   when `gradient_accumulation_steps` is divisible by
   `steps_per_generation * num_iterations`, TRL uses
   `per_token_logps.detach()` as the PPO anchor instead of a separate
   recomputed old-logprob tensor.
2. **vLLM topology**: TRL uses colocated vLLM with tensor parallel size 2. The
   good verl-gr `fp32opt` run kept TP=1/DP=2, so test
   `ROLLOUT_TENSOR_PARALLEL_SIZE=2` only as a separate follow-up run.
3. **Distributed backend**: TRL uses DeepSpeed ZeRO-3 with fp32 optimizer/master
   state; verl-gr uses FSDP2 with Ray-managed actor/ref/rollout workers. Compare
   checkpoint parameter drift and optimizer-state dtypes after the fresh run.
4. **Ray/RPC boundaries**: verl-gr crosses DataProto/Ray boundaries between
   generation, log-prob, reference, and actor-update phases; TRL is colocated.
   If the numerical paths above are aligned, instrument tensors at these
   boundaries to find where policy/ref/old log-probs diverge.

### 4.4 Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    verl-gr Convergence Problem                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Batching aligned (6 prompts)       Clip aligned (0.06/0.08)     │
│         │                                │                        │
│         ▼                                ▼                        │
│  ┌──────────────┐              ┌──────────────────┐              │
│  │ Same prompt  │              │ Same trust       │              │
│  │ groups as TRL│              │ region as TRL    │              │
│  └──────────────┘              └────────┬─────────┘              │
│                                         │                        │
│                                         ▼                        │
│                               ┌──────────────────┐              │
│                               │ Need rerun with  │              │
│                               │ aligned batching │              │
│                               │ to isolate       │              │
│                               │ backend, RPC, TP │              │
│                               └────────┬─────────┘              │
│                                        ▼                         │
│                         ┌──────────────────┐                      │
│                         │ Compare reward,  │                      │
│                         │ KL, clipfrac,    │                      │
│                         │ wall-clock       │                      │
│                         └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Recommended Fixes (Prioritized)

### Priority 1: Keep the Confirmed fp32 Actor Baseline

The confirmed good run is:

```bash
EXPERIMENT_NAME=g2_3_trlmatch_ppoegradaccu6_trainshuffleOn_fp32opt
ACTOR_MODEL_DTYPE=fp32
PPO_CLIP_RATIO=0.06
PPO_CLIP_RATIO_HIGH=0.08
DATA_SHUFFLE=True
VALIDATION_SHUFFLE=False
ROLLOUT_TENSOR_PARALLEL_SIZE=1
```

This is the baseline that fixed the flat-KL failure. Preserve it before testing
additional structural/backend changes.

### Priority 2: Follow-Up Experiments

```bash
# Test separately, not as part of the confirmed fp32opt baseline:
ROLLOUT_TENSOR_PARALLEL_SIZE=2
# Optional later experiment:
# old_log_prob_mode=current
```

These changes may further align backend behavior, but they were not responsible
for the good `fp32opt` KL result.

### Expected Impact Summary

| Fix | Expected Convergence Improvement | Expected Speed Improvement |
|---|---|---|
| `ACTOR_MODEL_DTYPE=fp32` | Confirmed — fixed flat KL in `fp32opt` run | Neutral |
| `old_log_prob_mode=current` | Unknown — not part of good run | None yet — still computes old log-probs for diagnostics |
| vLLM TP=2 | Unknown — not part of good run | Possible generation speedup |

Batching, clip ratio, `ppo_epochs`, training shuffle, and validation no-shuffle behavior are now aligned. The next run should isolate the remaining backend/RPC/TP differences.

---

## 6. Verification Plan

After applying fixes, compare the following metrics between TRL and verl-gr runs:

### Convergence Metrics
- `eval/reward_total` over steps: should converge at similar rate
- `kl_loss`: should stay in similar range (not exploding)
- `actor/pg_clipfrac`: should be < 0.3 (indicating most tokens within clip window)
- `actor/pg_loss`: should be non-zero throughout training

### Speed Metrics
- Wall-clock time per 100 optimizer steps
- Tokens processed per second (training throughput)
- Generation throughput (tokens/sec during vLLM generation)

### Stability Metrics
- Reward variance across steps (should be decreasing)
- KL divergence trajectory (should be smooth, not spiking)
- Gradient norm (should be stable)

---

## A. Reference: Key File Locations

### TRL Reference Implementation
| File | Purpose |
|---|---|
| `Rank-GRPO/scripts/run_rl.sh` | Training launch script |
| `Rank-GRPO/train_rank_grpo.py` | Training entry point, config construction |
| `Rank-GRPO/configs/qwen25_0.5b_grpo.yaml` | DeepSpeed/accelerate config |
| `rank-grpo/lib/python3.10/site-packages/trl/trainer/rank_grpo_trainer.py` | Trainer: data loading, generation, loss, advantages |
| `rank-grpo/lib/python3.10/site-packages/trl/trainer/grpo_config.py` | GRPO config, steps_per_generation calculation |
| `Rank-GRPO/libs/reward_funcs.py` | Reward function definitions |

### verl-gr Implementation
| File | Purpose |
|---|---|
| `verl-gr-fork-workingbranch/scripts/.match_rankgrpo.sh` | Training launch script (hyperparameter overrides) |
| `verl-gr-fork-workingbranch/scripts/run_rankgrpo.sh` | verl-gr runtime launcher |
| `verl-gr-fork-workingbranch/configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml` | Hydra config for RankGRPO |
| `verl-gr-fork-workingbranch/verl_gr/recipes/rankgrpo/rankgrpo_loss.py` | PPO loss with TRL-matched path |
| `verl-gr-fork-workingbranch/verl_gr/recipes/rankgrpo/rankgrpo_algorithm.py` | Rank-GRPO advantage computation |
| `verl-gr-fork-workingbranch/verl_gr/recipes/rankgrpo/rankgrpo_trainer.py` | Trainer adapter, validation |
| `verl-gr-fork-workingbranch/verl_gr/recipes/rankgrpo/rankgrpo_reward.py` | Reward computation |
| `verl-gr-fork-workingbranch/verl_gr/trainers/rl_trainer.py` | RLTrainer, compute_advantage override |
| `verl_080_dev/verl/trainer/ppo/ray_trainer.py` | Core verl PPO trainer (fit, _update_actor) |
| `verl_080_dev/verl/trainer/ppo/core_algos.py` | PPO core algorithms (agg_loss, kl_penalty) |
| `verl_080_dev/verl/workers/config/actor.py` | ActorConfig defaults (ppo_epochs, clip_ratio) |

---

*Analysis date: 2026-05-26*
