# Rank-GRPO

## Sample Input & Results

### Task

Given a Reddit movie recommendation conversation, the model outputs 20 movie recommendations in `"Title (Year)"` format.

### Sample Input/Output

**Input (prompt):**
```
Pretend you are a movie recommender system.
I will give you a conversation between a user and you (a recommender system).
Based on the conversation, you need to reply with 20 recommendations.
List the standardized English title of each movie in each line in the form of
"movie name" (release_year) with NO extra words or sentences.

Here is the conversation: USER: Suggest me some thought provoking movies.
Hi everyone I'm in a mood to watch something very interesting clever and also
entertaining so please suggest me some entertaining thought provoking movies

I've already watched all Christopher Nolan and Charlie Kaufman movies
```

**Ground Truth (8 items):** November (2004), A Scanner Darkly (2006), Waking Life (2001), Pi (1998), Awakenings (1990), The Secret (2006), Eternal Sunshine of the Spotless Mind (2004), The Help (2011)

**SFT Model Output (top-5):**
```
The Matrix (1999)
Eternal Sunshine of the Spotless Mind (2004)
Inception (2010)
Fight Club (1999)
Memento (2000)
...
```

**GRPO Model Output (step 40200, top-5):**
```
Inception (2010)
Eternal Sunshine of the Spotless Mind (2004)
Interstellar (2014)
The Prestige (2006)
Memento (2000)
...
```

### SFT vs GRPO — Offline Evaluation

Both evaluated on the full test set: 2,050 unique contexts, 10,972 total samples.

| Metric | SFT (epoch 1.5) | GRPO (step 40200) | Delta |
|--------|-----------------|-------------------|-------|
| Recall@5 | 0.0681 | 0.0740 | +8.6% |
| Recall@10 | 0.1064 | 0.1188 | +11.7% |
| Recall@15 | 0.1343 | 0.1517 | +13.0% |
| Recall@20 | 0.1510 | 0.1734 | +14.8% |
| NDCG@5 | 0.0518 | 0.0574 | +10.8% |
| NDCG@10 | 0.0644 | 0.0722 | +12.1% |
| NDCG@15 | 0.0726 | 0.0819 | +12.7% |
| NDCG@20 | 0.0771 | 0.0876 | +13.6% |

GRPO consistently improves over SFT across all metrics. Gains increase with K, from +8.6% Recall@5 to +14.8% Recall@20.

---

## Learning Paradigm

The original TRL run uses:

```bash
--reward_func exp_inf
```

In `Rank-GRPO/libs/reward_funcs.py`, `exp_inf` calls
`evaluate_direct_match_aligned(...)` and returns a length-20 vector of binary
per-rank hits:

```text
rank_rewards[j] = 1 if recommendation j matches a ground-truth movie
rank_rewards[j] = 0 otherwise
```

This is not the paper-style DCG suffix return. The paper's rank-level return is
closer to TRL's `log_decay` path, where hits are discounted by rank and each
rank receives the remaining suffix DCG:

```text
gains = hits * discounts
reward_at_rank_i = sum(gains[i:])
```

However, the reference run we align against uses `exp_inf`, so verl-gr matches
that behavior rather than switching to `log_decay`.

### Reward Alignment

TRL computes `rewards_items` from `exp_inf`, normalizes them within each prompt's
8 generations, and broadcasts each item's advantage to the tokens belonging to
that recommendation. It also logs:

```text
train/reward_total = mean(sum(rewards_items per 20-item list))
```

So for the current TRL run, `train/reward_total` is the average number of matched
ground-truth items per generated list.

verl-gr mirrors the same training signal in
`verl_gr/recipes/rankgrpo/rankgrpo_algorithm.py`: it computes `rank_rewards`
with the same aligned matching logic, normalizes within each prompt group, and
broadcasts item-level advantages to recommendation tokens. It logs these
training TensorBoard scalars:

```text
train/rankgrpo/reward_total = mean(sum(rank_rewards per 20-item list))
train/rankgrpo/reward       = mean(rank_rewards over 20 positions)
train/rankgrpo/hit_any      = fraction of generated lists with at least one hit
```

`train/rankgrpo/reward_total` is the direct verl-gr counterpart for TRL's
`train/reward_total`. `critic/rewards/mean` is not that metric.

During validation, `eval/reward_total` is an alias for
`val-aux/rankgrpo/rank_reward_sum/mean@8`, so it is also an average hit count
per list. `eval/reward` similarly aliases
`val-aux/rankgrpo/rank_rewards/mean@8`, the per-position mean hit rate. For
example, `eval/reward_total = 0.4106` means about 0.41 matched ground-truth items
per 20-item generated list on average. It does not mean NDCG is 0.4106.

`critic/rewards/mean` is different. It is a generic verl PPO metric computed
from `token_level_rewards.sum(-1)`. In this Rank-GRPO rollout path the scalar
reward score is `float(any(rank_rewards))`, so `critic/rewards/mean` is closer
to a "hit-any" rate: the fraction of generated lists with at least one hit. It
is not the same as TRL `train/reward_total`, and it is not NDCG.

NDCG is computed only by the separate offline evaluation code using
`ndcg_at_k(...)`, where hits are discounted by rank and normalized by ideal DCG.
Do not infer NDCG directly from `eval/reward_total` or `critic/rewards/mean`.

---

## Training Convergence

<img width="400" alt="image" src="https://github.com/user-attachments/assets/b6b51358-c040-48c1-88a0-59c3b5523ddf" />

### Aligned Trace Comparison

Trace sources:

- Original TRL Rank-GRPO: `Rank-GRPO/results/grpo/new2/runs`
- verl_gr fork: `tensorboard_log/RankGRPO/g2_3_trlmatch_ppoegradaccu6_trainshuffleOn_fp32opt`

These are short aligned traces, not completed one-epoch/full-convergence runs. The comparison below uses the overlapping region around 600 optimizer steps and eval step 400.

| Metric | Original TRL Rank-GRPO | verl_gr fork (`fp32opt`) | Notes |
|--------|-------------------------|---------------------------|-------|
| Train scalar range | steps 10-820 across resumed traces | steps 10-590 | TensorBoard scalar coverage |
| Comparable train step | 600 | 590 | Nearest available overlap |
| Train KL | 0.0391 at step 600 | 0.0408 at step 590 | KL is now aligned in scale and trend |
| Train loss | -0.00052 at step 600 | -0.00012 at step 590 | Same small-loss regime |
| Grad norm | 0.4449 at step 600 | 0.0458 at step 590 | Different backend/optimizer dynamics |
| Train reward total | 0.4125 at step 600 | N/A | TRL logs `train/reward_total`; verl logs rollout reward under different names |
| Eval reward total | 0.3782 at step 400 | 0.3814 at step 400 | Comparable held-out reward |
| Eval KL | 0.0260 at step 400 | N/A | TRL logs eval KL; verl trace logs train actor KL |
| Clip fraction | 0.0 | 0.0 | Both stay inside the clip region |

The key alignment result is KL behavior. The older verl_gr runs had a flat `actor/kl_loss`; the `fp32opt` run no longer does. It rises from `0.000167` at step 10 to `0.0408` at step 590, closely matching TRL's `train/kl` of `0.0391` at step 600.

---

## Performance

### Timing Distribution Analysis

Measured means come from fork TensorBoard `timing_s/*` scalars during the sidecar run only (`RUN_DEBUG_STEP` set in `run_rankgrpo.sh`). Per-phase TRL times are pro-rata estimates from TRL total step time (tqdm train log); TRL does not export the same modular breakdown. Validation (`timing_s/testing`) and checkpoint save are excluded from the training-step table.

Mean `timing_s/*` over fork steps **2–29** (n=28, warmup skipped). Enabled only when `RUN_DEBUG_STEP` is set.
TRL total step time: avg from train log (train_20260707_070257_gpus6,7.log).
Per-phase TRL times are **pro-rata estimates** `TRL_total × (fork_phase / fork_total)` — TRL does not log modular `timing_s/*`.

| Phase | verl-gr Time | TRL Time | Delta Step Time |
|-------|--------------|----------|-----------------|
| gen (vLLM rollout) | 0.99s (22%) | 1.20s† | -0.21s |
| update_actor (FSDP train step) | 1.52s (33%) | 1.85s† | -0.33s |
| update_weights (actor → rollout sync) | 1.18s (26%) | 1.43s† | -0.25s |
| old_log_prob | — | — | — |
| ref | 0.75s (17%) | 0.92s† | -0.16s |
| adv | 0.10s (2%) | 0.12s† | -0.02s |
| reward | 0.00s (0%) | 0.00s† | -0.00s |
| Other/overhead | 0.00s (0%) | 0.00s† | +0.00s |
| **Total logged step** | 4.55s | 5.53s | -0.99s |

† TRL phase time estimated pro-rata from total step time (tqdm / TB).
Eval (`timing_s/testing`) and checkpoint (`timing_s/save_checkpoint`) are excluded from this per-step training breakdown.

**Key finding:** the current `fp32opt` trace is still not vLLM-rollout bound. vLLM rollout is ~10% of logged step time, while actor update plus weight synchronization is ~55%.

### Sidecar Trajectories

Criteria (each step): **logprob gate** and **KL gate** rel diff ≤20% vs TRL; **step time gate**: verl-gr s/it < TRL s/it at the same step. Header **step time** gate uses verl-gr vs TRL **average** s/it (excl. warmup step 1). Logprob/KL/step-time gates skip step 1 (vLLM compile + first-cycle warmup).

* Gate status (rel diff)

| step | logprob gate | KL gate | verl-gr KL | TRL KL | step time gate | verl-gr time | TRL time |
|------|--------------|---------|---------|--------|----------------|-----------|----------|
| 1 | — | — | 0.000161 | 0 | — | 58.380s | 10.420s |
| 2 | OK (0.055) | **FAIL** (1.002) | 0.0002115 | 0.0001056 | OK | 4.059s | 7.460s |
| 3 | OK (0.094) | **FAIL** (0.569) | 0.0002714 | 0.0001729 | OK | 4.358s | 6.360s |
| 4 | OK (0.052) | **FAIL** (0.566) | 0.0001925 | 0.0001229 | OK | 4.666s | 5.850s |
| 5 | OK (0.104) | **FAIL** (1.381) | 0.0004086 | 0.0001716 | OK | 5.132s | 5.780s |
| 6 | OK (0.152) | OK (0.050) | 0.0003692 | 0.0003515 | OK | 3.567s | 5.550s |
| 7 | OK (0.008) | **FAIL** (2.954) | 0.0001911 | 4.833e-05 | OK | 4.117s | 5.390s |
| 8 | OK (0.067) | **FAIL** (0.744) | 0.0001388 | 7.958e-05 | OK | 3.867s | 5.270s |
| 9 | OK (0.086) | **FAIL** (1.488) | 0.0001451 | 5.832e-05 | OK | 4.948s | 5.180s |
| 10 | OK (0.079) | **FAIL** (1.506) | 0.0001617 | 6.453e-05 | OK | 3.986s | 5.140s |
| 11 | OK (0.152) | **FAIL** (8.489) | 0.0001317 | 1.388e-05 | OK | 4.078s | 5.090s |
| 12 | OK (0.029) | **FAIL** (1.865) | 0.0001514 | 5.284e-05 | OK | 4.429s | 5.430s |
| 13 | OK (0.158) | **FAIL** (1.826) | 0.0002071 | 7.33e-05 | OK | 4.316s | 5.270s |
| 14 | OK (0.149) | **FAIL** (0.879) | 0.0001921 | 0.0001022 | OK | 4.069s | 5.180s |
| 15 | OK (0.018) | **FAIL** (1.085) | 0.0003298 | 0.0001582 | OK | 4.845s | 5.110s |
| 16 | OK (0.100) | **FAIL** (0.646) | 0.0003853 | 0.0002342 | OK | 4.877s | 5.050s |
| 17 | OK (0.024) | **FAIL** (0.516) | 0.0004617 | 0.0003046 | OK | 4.894s | 5.040s |
| 18 | OK (0.152) | **FAIL** (0.675) | 0.0005838 | 0.0003486 | OK | 4.937s | 5.030s |
| 19 | OK (0.105) | **FAIL** (0.611) | 0.0007312 | 0.0004538 | OK | 4.589s | 5.030s |
| 20 | OK (0.106) | **FAIL** (0.743) | 0.000851 | 0.0004883 | OK | 4.296s | 5.510s |
| 21 | OK (0.132) | **FAIL** (0.598) | 0.0008894 | 0.0005568 | OK | 4.903s | 5.580s |
| 22 | OK (0.101) | **FAIL** (0.531) | 0.0009592 | 0.0006263 | OK | 4.268s | 5.620s |
| 23 | OK (0.053) | **FAIL** (0.759) | 0.0006146 | 0.0003495 | OK | 4.057s | 5.480s |
| 24 | OK (0.030) | **FAIL** (1.401) | 0.0001811 | 7.541e-05 | OK | 4.144s | 5.560s |
| 25 | OK (0.094) | **FAIL** (1.380) | 0.0002875 | 0.0001208 | OK | 4.982s | 5.640s |
| 26 | OK (0.031) | **FAIL** (1.085) | 0.0003208 | 0.0001539 | OK | 5.039s | 5.570s |
| 27 | **FAIL** (0.226) | **FAIL** (2.336) | 0.0003597 | 0.0001078 | OK | 5.515s | 5.990s |
| 28 | **FAIL** (7.979) | **FAIL** (6.085) | 0.000677 | 9.555e-05 | OK | 5.192s | 5.950s |
| 29 | **FAIL** (1.454) | **FAIL** (4.134) | 0.0005013 | 9.763e-05 | OK | 5.226s | 5.840s |
| Avg Time (Except for 1st step) | | | | | | 4.548s | 5.534s |
