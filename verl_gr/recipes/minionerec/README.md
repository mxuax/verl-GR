# MiniOneRec

## Sample Input & Results

### Task

Sequential next-item recommendation: given a user's historical item sequence,
the model emits one catalog-valid SID in the form
`<s_a_*><s_b_*><s_c_*>`. Training rollout uses constrained HF beam search
(`beam=16`) over the MiniOneRec item trie; same-evaluator validation follows the
original MiniOneRec `evaluate.py` + `calc.py` contract with `beam=50`.

### Sample Input/Output

**Input (prompt, abbreviated):**

```text
User history:
  <s_a_1203><s_b_881><s_c_4502>
  <s_a_0187><s_b_991><s_c_7310>
  ...  (historical item SIDs)  ...
Please predict the next item.
```

**Ground Truth:**
```text
<s_a_2214><s_b_1006><s_c_3377>
```

**Beam outputs (abbreviated):**
```text
<s_a_0421><s_b_3372><s_c_0934>
<s_a_2214><s_b_1006><s_c_3377>
<s_a_5199><s_b_0028><s_c_4401>
...
```

Scores are 1 only when a predicted SID matches the target item under the
MiniOneRec catalog contract. HR@20 / NDCG@20 are computed by the original
MiniOneRec evaluator rather than inferred from generic trainer reward scalars.

## Learning Paradigm

This implementation is aligned to the original MiniOneRec GRPO recipe instead
of introducing a new reward, penalty, or frozen-reference variant.

The policy-gradient term follows the original score-function estimator:

```text
exp(logp - logp.detach()) * advantage
```

The KL term uses the original unclamped low-variance KL path:

```text
exp(ref_logp - logp) - (ref_logp - logp) - 1
```

The reference model is synchronized by EMA with `sync_freq=512` and
`ref_model_mixup_alpha=0.6`. Actor/ref forward stays in bf16, while the DDP
optimizer path uses `paged_adamw_32bit` for the intended fp32 master-state
update behavior.

## Training Convergence

The current branch has implemented the mechanism and update-precision alignment
needed by MiniOneRec GRPO:

- rollout transition logprobs are carried through the constrained beam path;
- completion-only actor/ref logprob follows the original `logits_to_keep`
  contract;
- the DDP config composes at Hydra root (`# @package _global_`) so the optimizer
  is not silently replaced by torch AdamW stubs;
- same-batch DDP-vs-DeepSpeed replay probes match loss and visible parameter
  deltas under production-like micro-batching / accumulation settings;
- same-evaluator validation uses the original MiniOneRec beam-50 metric path.

| Setting | Value |
| --- | --- |
| Backend | DDP |
| Train beam | 16 |
| Validation beam | 50 |
| Optimizer | `paged_adamw_32bit` |
| LR | `1e-5` |
| KL | `minionerec_low_var_kl`, coef `0.001` |
| Ref sync | every 512 steps, mixup alpha `0.6` |
| Forward dtype | bf16 |

Recommended launcher:

```bash
cd verl-GR
export BASE_MODEL=/path/to/MiniOneRec/output_dir/xxx/checkpoint-390
export PYTHON_BIN=/path/to/vllm-gr/bin/python
bash scripts/run_minionerec_grpo_rl_aligned.sh
```

Same-evaluator validation should be run with:

```bash
BASE_MODEL=/path/to/checkpoint/actor/huggingface \
RESULT_NAME=minionerec_valid_stepXXXX \
bash /path/to/local/eval_minionerec_valid_beam50.sh
```

Local convergence/debug helper scripts are intentionally kept outside the
tracked repository; the committed recipe keeps only the runtime path needed for
the aligned MiniOneRec implementation.

## Performance

The current MiniOneRec path prioritizes faithful alignment first. It already
avoids several unnecessary costs for this recipe:

- constrained HF beam search prevents invalid catalog SIDs;
- DDP actor/ref workers avoid vLLM initialization for MiniOneRec training;
- completion-only logprob avoids computing unused prompt-token LM-head outputs;
- padded and remove-padding logprob paths are available for parity checks.

There is still optimization room in constrained decoding throughput, actor/ref
logprob scheduling overhead, rollout/update overlap, and validation/checkpoint
cost. These are performance-engineering items on top of the aligned training
contract, not changes to the MiniOneRec objective.
