# OpenOneRec

## Sample Input & Results

### Task

Cross-domain next-item recommendation: given a user’s recent **video** SIDs and **product** SIDs, the model writes a short Chinese CoT (`<think>...</think>`) and then emits one candidate product SID
`<|sid_begin|><s_a_*><s_b_*><s_c_*><|sid_end|>`. Stage-2 beam search returns 32 SID candidates per prompt for pass@k / hit scoring.

### Sample Input/Output

**Input (prompt, abbreviated):**
```
system
你是一个智能跨域推荐助手，能够根据用户观看的视频内容和历史购物行为，预测用户接下来可能点击的商品。
user
用户浏览过的视频内容：
  <|sid_begin|><s_a_1669><s_b_2637><s_c_4410><|sid_end|>
  <|sid_begin|><s_a_2357><s_b_5599><s_c_7361><|sid_end|>
  ...  (100 video SIDs total)  ...
  <|sid_begin|><s_a_38><s_b_4385><s_c_445><|sid_end|>
用户浏览过的商品：
  <|sid_begin|><s_a_3057><s_b_486><s_c_3189><|sid_end|>
  <|sid_begin|><s_a_3882><s_b_6125><s_c_3497><|sid_end|>
  ...  (93 product SIDs total)  ...
  <|sid_begin|><s_a_2650><s_b_7990><s_c_3678><|sid_end|>
请推荐用户接下来可能感兴趣并点击的商品。
assistant
```

**Ground Truth (6 unique SIDs among 10 listed):**
```
<s_a_3864><s_b_1190><s_c_4289>
<s_a_3724><s_b_3970><s_c_7622>
<s_a_3724><s_b_3970><s_c_4198>
<s_a_1910><s_b_6941><s_c_5539>
<s_a_3742><s_b_4204><s_c_3385>
<s_a_5404><s_b_5889><s_c_4812>
```

**Step-0 model CoT (truncated):**
```
<think>
好的，用户最近在浏览了一些关于手机壳和穿搭的视频，比如华为nova11Pro的防摔壳，还有显瘦的牛仔裤。
同时，他购买了一些饰品，比如四叶草项链和四叶草手绳。现在他可能对手机壳有新的需求，或者想尝试其他配饰。
...
总结一下，用户的需求是手机保护、显瘦穿搭和幸运饰品。
</think>
```

**Step-0 beam outputs (first 5 of 32 SID candidates; `hit_reward=0`, `pass_at_1=0` on this prompt):**
```
<|sid_begin|><s_a_6996><s_b_4521><s_c_6314>
<|sid_begin|><s_a_254><s_b_214><s_c_4871>
<|sid_begin|><s_a_2657><s_b_5338><s_c_75>
<|sid_begin|><s_a_2628><s_b_7348><s_c_4410>
<|sid_begin|><s_a_2429><s_b_1699><s_c_3849>
...
```

That SID above is a **miss**: it is not in GT, so this beam has `hit_reward=0` and `pass_at_1=0`.
Scores are 1 only when a predicted SID matches GT (e.g. `<s_a_3864><s_b_1190><s_c_4289>`).
With one SID per beam, `hit_reward` and `pass_at_1` coincide (both `|pred ∩ GT| / |pred|` on the first/only SID).

## Overfitting Evaluation & Performance

We took 100 samples from [`sft_product_rec.parquet`](data/prepare_rl.sh) for overfitting. Training on 4 devices with 50 epochs in total, the train-batch response size is `4 x 32 beams = 128`. Following the [vanilla](https://github.com/Kuaishou-OneRec/OpenOneRec/blob/a969edcadd579a06c1966ae1db5984e02f48beff/verl_rl/recipe/onerec/onerec_recipe.py#L560), we use pass@1 (first_sid_hit_reward) as the sampling reward.

<img height="800" alt="image" src="https://github.com/user-attachments/assets/775237f6-fc94-481c-882f-87c84881f1aa" />

* `eval/reward_total`: fraction of all beam slots that hit, over responses (`100×32=3200`)
* `actor/kl_loss`: mean `low_var_kl(actor, ref)` over response tokens (raw; before `kl_loss_coef`)
* `critic/rewards/mean`: train-batch mean of sequence `token_level_rewards` (= first-SID 0/1 score), over responses (128)
* `val-aux/RecIF_ProductRec/pass_at_32`: **pass@32** — per prompt: 1 if *any* of 32 beams hits GT, else 0; then mean over **100 prompts** (denominator 100)
* `val-aux/RecIF_ProductRec/pass_at_1/mean@32`: **mean pass@1** — each beam already has a 0/1 first-SID score (`pass_at_1`); take the *mean* of those 32 scores per prompt (not max), then mean over **100 prompts** (≈ hit rate over `3200` beam slots; same scale as `eval/reward_total`)

**Naming tips:** `pass_at_N` = “success if ≥1 hit among N tries” (OR over beams). `pass_at_1/mean@K` = “variable is per-beam pass@1; aggregate with mean over K beams” (average, not OR). Do not read `mean@32` as pass@32.

**With less than 5 hours / 50 epochs of overfitting, `pass@1` increases from 0.003 (~10/3200) to 0.01 (~32).**
