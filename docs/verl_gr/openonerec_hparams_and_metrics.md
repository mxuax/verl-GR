# OpenOneRec Hyperparameters and Metrics in `verl-GR`

This document summarizes the important hyperparameters, validation metrics, and training metrics for the OpenOneRec GRPO workflow in `verl-GR`, with a direct comparison to the legacy OpenOneRec runtime.

The key compatibility goal is that both runtimes use the same training reward semantics, even though their rollout, metric aggregation, and logging paths differ:

- Legacy OpenOneRec uses its forked `verl` 0.5.0 runtime and a synchronous/custom vLLM two-stage rollout path.
- `verl-GR` uses the current upstream `verl` branch with an async agent-loop rollout path and a `verl-GR` internal two-stage beam backend.

## Runtime Comparison

| Area | Legacy OpenOneRec | `verl-GR` |
| --- | --- | --- |
| Main launcher | `OpenOneRec/verl_rl/recipe/onerec/run_grpo.sh` | `verl-GR/scripts/run_openonerec_grpo.sh` |
| Trainer entrypoint | `recipe.onerec.main_onerec_ppo` | `verl_gr.trainers.main_ppo` |
| Trainer extension | `recipe.onerec.onerec_ray_trainer` | `verl_gr.trainers.rl_trainer.RLTrainer` |
| Reward function | `recipe/onerec/onerec_recipe.py::compute_score` | `verl_gr/recipes/openonerec/onerec_recipe.py::compute_score` |
| Rollout path | legacy two-stage vLLM rollout | async two-stage agent-loop rollout |
| `verl` version line | forked `verl` 0.5.0 | current upstream `verl` branch |
| vLLM use | legacy synchronous/custom beam path | async server plus `verl-GR` beam backend |

## Important Hyperparameters

The script-level values usually override YAML defaults. When debugging parity, compare the resolved Hydra config and the launcher environment, not only the YAML file.

| Hyperparameter | Legacy OpenOneRec default | `verl-GR` default | Meaning |
| --- | --- | --- | --- |
| `TRAIN_BATCH_SIZE` | `N_GPUS * N_NODES` | `N_GPUS * N_NODES`, with fallback to 2 GPUs if Ray discovery fails | Global training batch size per PPO step. Also controls the number of training steps for a fixed dataset size. |
| `data.train_batch_size` | `TRAIN_BATCH_SIZE` | `TRAIN_BATCH_SIZE` | Number of prompts per training step. |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `TRAIN_BATCH_SIZE` | `TRAIN_BATCH_SIZE` | PPO mini-batch size. In the OpenOneRec scripts this is intentionally aligned with the train batch size. |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `True` | `True` | Enables dynamic token-based micro-batching. |
| `MAX_TOKENS_PER_GPU` | `40960` | `40960` | Token budget for actor/ref log-prob and rollout-related computation. |
| `actor_rollout_ref.actor.optim.lr` | `2e-6` | `2e-6` | Actor learning rate. |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | `10` | `10` | Learning-rate warmup steps. |
| `actor_rollout_ref.actor.optim.weight_decay` | `0.1` | `0.1` | Actor optimizer weight decay. |
| `actor_rollout_ref.actor.use_kl_loss` | `True` | `True` | Adds reference-model KL loss to the actor objective. |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | `0.001` | Coefficient applied to actor-side KL loss. |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | `low_var_kl` | KL estimator used for actor-side KL loss. |
| `algorithm.use_kl_in_reward` | `False` | `False` | KL is not subtracted from reward. `token_level_rewards` equals `token_level_scores`. |
| `algorithm.adv_estimator` | `grpo` | `grpo` | Uses GRPO outcome advantage. |
| `algorithm.norm_adv_by_std_in_grpo` | `True` | `True` | Normalizes group advantages by group standard deviation when applicable. |
| `critic.enable` | `False` | `False` | No critic/value model is used. |
| `actor_rollout_ref.rollout.name` | `two_stage` | forced to `two_stage` by launcher | Enables OpenOneRec two-stage generation. |
| `actor_rollout_ref.rollout.mode` | legacy synchronous/custom path | `async` by default | `verl-GR` uses upstream async rollout infrastructure. |
| Stage-1 max tokens | `1024` | `1024` | Maximum reasoning/CoT tokens before item generation. |
| Stage-2 item tokens | `3` | `3` | Number of SID tokens for a three-level item codebook. |
| Beam width | `32` | `32` | Number of candidate item beams. |
| `actor_rollout_ref.rollout.temperature` | `1.0` | `1.0` | Training rollout sampling temperature. |
| `actor_rollout_ref.rollout.top_p` | `1.0` | `1.0` | Training rollout nucleus sampling threshold. |
| `actor_rollout_ref.rollout.max_num_seqs` | `2048` in the legacy launcher | `512` in the `verl-GR` launcher unless overridden | vLLM sequence concurrency limit. Affects throughput and memory pressure. |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.8` | `0.8` in YAML | vLLM GPU memory budget. |
| `trainer.test_freq` | `50` | `20` by default | Validation frequency. |
| `trainer.total_epochs` | `20` | `2` by default | Number of training epochs. Often overridden by launch commands. |
| `trainer.val_before_train` | `True` | `True` | Runs validation at global step 0 unless disabled. |
| `ENABLE_THINK` | `False` in the legacy launcher | `True` in the `verl-GR` launcher | Prompt-mode difference that can significantly affect stage-1 text and stage-2 SID results. |

For parity runs, explicitly align at least:

- `ENABLE_THINK`
- `TRAIN_BATCH_SIZE`
- `data.train_max_samples`
- `data.val_max_samples`
- validation seed and sampling behavior
- `beam_width`
- Stage-1 and Stage-2 token limits
- `max_num_seqs`
- checkpoint initialization and resume mode

## Validation Metrics

Validation metrics are produced from generated responses and rule-based reward outputs.

The common high-level flow is:

```text
generate responses
-> compute_score(...)
-> reward tensor and reward_extra_info
-> process_validation_metrics(...)
-> TensorBoard scalar tags under val-core/ and val-aux/
```

In `verl-GR`, this is implemented mainly in:

- `verl_gr/recipes/openonerec/onerec_trainer.py::openonerec_validate`
- `verl_gr/recipes/openonerec/onerec_recipe.py::compute_score`
- upstream `verl/trainer/ppo/metric_utils.py::process_validation_metrics`

### Reward Fields

| Field | Calculation | Training use |
| --- | --- | --- |
| `score` | Equal to `first_sid_hit_reward`: whether the first parsed SID triple matches a ground-truth SID. Binary 0/1. | Yes. This is the main training reward. |
| `pass_at_1` | Same semantic value as `score`. | Logged as extra info; training reads `score`. |
| `reward` | `reward_tensor.sum(-1)`. Because the rule reward is written to the final valid response token, this is normally equal to `score`. | Yes, as the tensor form used by PPO/GRPO. |
| `partial_hit_reward` | Hierarchical SID match: full triple match scores 100, first two levels score 10, first level scores 1, then averaged over predicted SIDs. | Logged only by default. |
| `hit_reward` | `len(predicted_sid_set ∩ gt_sid_set) / len(predicted_sid_tuples)`. | Logged only by default. |
| `pass_rate` | Whether any predicted SID intersects with the ground-truth SID set. Binary 0/1. | Logged only by default. |
| `format_reward` | Whether the generated response has valid thinking format and sufficient content. | Logged only by default. |

Only `score` is the default rule reward used for model updates. The other fields are useful diagnostics but do not drive training unless the reward manager or `compute_score` contract is changed.

### TensorBoard Tag Structure

Validation scalar tags use this shape:

```text
val-core/<data_source>/<variable>/<metric>
val-aux/<data_source>/<variable>/<metric>
```

For example:

```text
val-aux/RecIF_ProductRec/score/best@32/mean
val-core/RecIF_ProductRec/reward/best@32/mean
```

`RecIF_ProductRec` comes from the dataset `source` field.

For OneRec, `reward` is usually the core variable because there is no `acc` field. Therefore:

- `val-core/.../reward/...` is the main validation reward view.
- `val-aux/.../score/...` shows the raw `score` extra field.
- Since `reward` and `score` normally have the same scalar value, they should be close when they are grouped over the same samples.

### `best@N/mean`

`process_validation_metrics` groups samples by:

```text
data_source -> prompt/sample uid -> variable values
```

For each numeric variable and each prompt, it computes:

- `mean@K`: mean over K responses for that prompt.
- `std@K`: standard deviation over K responses.
- `best@N/mean`: bootstrap estimate of the expected best value among N responses.
- `worst@N/mean`: bootstrap estimate of the expected worst value among N responses.

For binary `score`, `best@32/mean` is close to the probability that at least one of 32 candidates hits the ground truth, with bootstrap sampling applied by the metric utility.

## Training Metrics

The training reward and update path is:

```text
compute_score()["score"]
-> rm_scores
-> token_level_scores
-> token_level_rewards
-> GRPO advantages
-> actor policy loss + actor KL loss
-> actor update
```

With the default OpenOneRec config:

```text
algorithm.use_kl_in_reward = false
```

so:

```text
token_level_rewards = token_level_scores
```

### Key Training TensorBoard Tags

| TensorBoard tag | Meaning |
| --- | --- |
| `critic/score/mean` | Mean sequence-level raw score in the current training batch. For OpenOneRec this is approximately the batch pass@1 hit rate. |
| `critic/score/max` | Maximum raw score in the current training batch. Since `score` is usually 0/1, this indicates whether any sample in the batch hit the ground truth. |
| `critic/score/min` | Minimum raw score in the current training batch. |
| `critic/rewards/mean` | Mean final reward used for advantage computation. With `use_kl_in_reward=false`, this should be close to `critic/score/mean`. |
| `critic/rewards/max` | Maximum final reward in the batch. |
| `critic/rewards/min` | Minimum final reward in the batch. |
| `critic/advantages/mean` | Mean token-level advantage over valid response tokens. In GRPO this is expected to be near zero because advantages are group centered. |
| `critic/advantages/max` | Largest advantage in the batch. Positive spikes indicate responses better than their group mean. |
| `critic/advantages/min` | Smallest advantage in the batch. Negative spikes indicate responses worse than their group mean. |
| `actor/pg_loss` | Actor policy-gradient loss after PPO clipping. This is the main policy update term driven by advantages. |
| `actor/kl_loss` | Actor-side reference KL loss before applying `kl_loss_coef`. This is logged as the raw KL loss scalar, not the scaled contribution. |
| `actor/ppo_kl` | Approximate KL between the old policy and the current actor during PPO update. This is not the same as reference KL. |
| `actor/grad_norm` | Actor gradient norm. Useful for checking whether updates are non-zero and stable. |
| `actor/lr` | Current actor learning rate. |

The `critic/*` prefix is inherited from upstream `verl` metric naming. In the OpenOneRec GRPO configuration, `critic.enable=false`; these tags do not imply that a critic model is active.

### Interpreting Sparse Rewards

GRPO only produces useful policy gradients when there is reward variation within a group.

```text
all group responses score 0 -> advantages are zero -> little or no task learning signal
all group responses score 1 -> advantages are also zero -> little or no relative learning signal
some responses score 1 and others score 0 -> non-zero advantages -> useful GRPO update
```

A practical health check is:

```text
critic/score/max occasionally reaches 1
critic/advantages/max is sometimes positive
critic/advantages/min is sometimes negative
actor/pg_loss has non-trivial variation
actor/grad_norm is non-zero
```

If `critic/score/max` is 0 for long stretches, the model is receiving no successful recommendation reward in those batches. In that case, GRPO can still execute, but it has little task-specific signal to learn from.

## Version-Sensitive Differences

Some metrics have the same names across the two runtimes but are not perfectly comparable.

| Area | Why it is version-sensitive |
| --- | --- |
| Beam search output | Legacy OpenOneRec uses an older vLLM/`verl` path. `verl-GR` uses async two-stage rollout and an internal beam loop. Candidate order, duplicate handling, and returned beams can differ. |
| `score` and validation metrics | The reward function is semantically aligned, but it is applied to outputs produced by different rollout implementations. Different beams can produce different `score` values. |
| `actor/kl_loss` magnitude | Newer upstream `verl` uses `Metric` aggregation with data-parallel and global token-aware normalization. Legacy OpenOneRec logs actor micro-batch scalars more directly. Same tag name does not guarantee identical aggregation semantics. |
| TensorBoard sparsity | Validation tags are only emitted at `test_freq` steps. Actor tags are emitted only when actor update metrics exist. Training scalar logging itself is simple, but some keys may not exist on every step. |
| vLLM batching | `max_num_seqs`, `max_num_batched_tokens`, prefix caching, async scheduling, and server concurrency can affect throughput and sometimes candidate ordering. |
| Reward plumbing | Legacy OpenOneRec uses its forked reward/trainer path. `verl-GR` follows newer upstream `rm_scores`, `reward_extra_keys`, and reward-loop conventions. The intended reward semantics remain aligned. |

## Recommended Parity Checklist

Before comparing curves across OpenOneRec and `verl-GR`, verify:

- The resolved config uses the same `ENABLE_THINK` value.
- `data.train_batch_size` and `actor_rollout_ref.actor.ppo_mini_batch_size` are equal.
- `data.val_max_samples` and validation sampling seed are aligned.
- `beam_width`, Stage-1 max tokens, and Stage-2 item token count are aligned.
- `critic/score/max` is not always zero during training.
- `critic/advantages/max` and `critic/advantages/min` show non-zero values on some steps.
- Validation dumps contain legal SID triples in the generated responses.
- `val-core/.../reward/...` and `val-aux/.../score/...` are interpreted as validation statistics, not as direct training loss values.

