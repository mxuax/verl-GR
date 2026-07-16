# MiniOneRec

## Sample Input & Results

### Task

MiniOneRec is a sequential recommendation task over catalog SIDs. Given a
user's historical item sequence, the model generates one catalog-valid SID in
the form:

```text
<s_a_*><s_b_*><s_c_*>
```

During RL training, MiniOneRec uses constrained HF beam search (`beam=16`) to
produce candidates that are valid under the item trie. Same-evaluator
validation uses the original MiniOneRec `evaluate.py` contract with `beam=50`.

### Sample Input/Output

**Input (prompt, abbreviated):**

```text
You are a recommendation assistant.
User history:
  <s_a_...><s_b_...><s_c_...>
  <s_a_...><s_b_...><s_c_...>
  ...
Please predict the next item SID.
```

**Output:**

```text
<s_a_123><s_b_456><s_c_789>
```

The reward is based on whether the generated SID matches the target item under
the MiniOneRec catalog contract.

---

## Alignment Status

The current `verl-GR` MiniOneRec path is designed to match the original
MiniOneRec GRPO training recipe rather than introducing a new objective. The
implemented alignment covers the mechanism and update-precision contracts that
matter for convergence:

- **Rollout / logprob contract**: rollout transition logprobs are preserved and
  used where required; MiniOneRec completion logprob follows the original
  `logits_to_keep` contract.
- **Policy loss**: `minionerec_reinforce` matches the original score-function
  estimator:

  ```text
  exp(logp - logp.detach()) * advantage
  ```

- **KL loss**: `minionerec_low_var_kl` implements the original unclamped
  low-variance KL path:

  ```text
  exp(ref_logp - logp) - (ref_logp - logp) - 1
  ```

- **Reference sync**: the reference policy uses EMA sync with
  `sync_freq=512` and `ref_model_mixup_alpha=0.6`, matching the target
  MiniOneRec training contract.
- **Optimizer precision**: the DDP recipe uses `paged_adamw_32bit` instead of
  silently falling back to torch AdamW. Same-batch replay probes show aligned
  loss and visible parameter deltas against the original DeepSpeed ZeRO-2 path.
- **Data order**: shuffled prompt order and seed handling are aligned with the
  original `Dataset.shuffle(seed=42)` + repeated-generation contract.

In short, the current branch has the MiniOneRec mechanism and update-precision
alignment in place. Remaining work is primarily performance engineering and
continued end-to-end throughput optimization.

---

## Training Convergence

Recommended launcher:

```bash
cd verl-GR
export BASE_MODEL=/path/to/MiniOneRec/output_dir/xxx/checkpoint-390
export PYTHON_BIN=/path/to/vllm-gr/bin/python
bash scripts/run_minionerec_grpo_rl_aligned.sh
```

Important defaults:

| Setting | Value |
| --- | --- |
| Backend | DDP |
| Train beam | 16 |
| Validation beam | 50 |
| Optimizer | `paged_adamw_32bit` |
| LR | `1e-5` |
| KL | `minionerec_low_var_kl`, coef `0.001` |
| Ref sync | every 512 steps, alpha `0.6` |
| Forward dtype | bf16 |

Same-evaluator validation should be run with:

```bash
BASE_MODEL=/path/to/checkpoint/actor/huggingface \
RESULT_NAME=minionerec_valid_stepXXXX \
bash scripts/convergence/eval_minionerec_valid_beam50.sh
```

This uses the original MiniOneRec `evaluate.py` + `calc.py` pipeline on the
valid split with `beam=50`.

---

## Performance

MiniOneRec in `verl-GR` prioritizes faithful mechanism alignment first:

- constrained HF beam search avoids invalid catalog SIDs;
- DDP actor/ref workers avoid unnecessary vLLM initialization for this recipe;
- completion-only logprob reduces wasted LM-head work;
- padded and remove-padding paths are both supported for parity experiments.

There is still room for further performance optimization. The main areas are:

- faster constrained decoding for large batches;
- lower-overhead actor/ref logprob scheduling;
- improved overlap between rollout, reward computation, and actor update;
- reducing validation/checkpoint overhead during long runs.

These are engineering optimizations on top of the aligned training contract.

---

## Key Files

- `verl_gr/recipes/minionerec/minionerec_recipe.py`: task runtime and dataset
  wiring.
- `verl_gr/recipes/minionerec/minionerec_loss.py`: MiniOneRec policy loss
  registration.
- `verl_gr/recipes/minionerec/constrained_beam_agent_loop.py`: constrained HF
  beam rollout path.
- `verl_gr/workers/engine/completion_only_logprob.py`: completion-token logprob
  path with original-style `logits_to_keep` support.
- `configs/verl_gr/minionerec/grpo_trainer_ddp.yaml`: DDP GRPO recipe config.
- `scripts/run_minionerec_grpo_rl_aligned.sh`: primary aligned launcher.
