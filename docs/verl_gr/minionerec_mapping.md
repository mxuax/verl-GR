# MiniOneRec Mapping

This document records the behavior contract used when adapting MiniOneRec to
`verl_gr`. MiniOneRec remains the reference implementation even when the code is
rewritten to fit the verl rollout and reward pipeline.

## Source Reference

- `MiniOneRec/data.py`: prompt construction, target formatting, and
  prompt/history/target lookup tables.
- `MiniOneRec/LogitProcessor.py`: constrained decoding behavior.
- `MiniOneRec/minionerec_trainer.py`: GRPO sampling, grouped generations,
  constrained beam training path, reward invocation, and train-time metrics.
- `MiniOneRec/rl.py`: reward functions and training dataset composition.
- `MiniOneRec/evaluate.py`: constrained beam evaluation and HR/NDCG postprocess.

## Dataset Contract

The first verl-GR implementation supports the MiniOneRec `SidDataset` main
recommendation task.

- Input CSV fields: `history_item_sid`, `item_sid`, optional `history_item_id`,
  `item_id`, `user_id`, `history_item_title`, and `item_title`.
- Prompt template:

```text
### User Input:
The user has interacted with items <sid history> in chronological order. Can you predict the next possible item that the user may expect?

### Response:
```

- History formatting follows `SidDataset`: `history_item_sid` is parsed as a
  Python list and joined with `, ` for the visible prompt.
- Target formatting follows `SidDataset`: `item_sid + "\n"`.
- Reward routing keeps enough metadata to reconstruct MiniOneRec's
  `prompt2history` and `history2target` behavior: `history_key`,
  `target_sid`, `dedup`, and row index.

Auxiliary RL alignment datasets from MiniOneRec, such as `RLTitle2SidDataset`
and `RLSeqTitle2SidDataset`, are intentionally left as follow-up work. Their
prompt templates and target formatting should be ported from `data.py` before
enabling them.

## Constrained Decoding Contract

MiniOneRec builds a prefix trie from `info_file`, using entries formatted as:

```text
### Response:
<a_i><b_j><c_k>
```

The original `ConstrainedLogitsProcessor`:

- tokenizes each formatted SID with the base tokenizer;
- appends `eos_token_id` to every tokenized SID path;
- uses `prefix_index = 4` for GPT-2-like models and `3` otherwise;
- for the first decoding step, hashes the final `prefix_index` tokens from the
  prompt;
- for later steps, hashes the generated prefix after `### Response:\n`;
- returns an empty allowed-token list when the prefix is not in the trie;
- optionally forces EOS when no allowed token exists.

The verl-GR implementation cannot reuse HuggingFace `LogitsProcessor` directly
inside async vLLM. It instead mirrors the same prefix-trie semantics inside the
Python beam backend. Beam state must be per-candidate, not a shared global
counter, because async beam expansion runs candidates concurrently.

## Reward Contract

The first implementation ports MiniOneRec's rule and ranking reward behavior:

- `rule_reward`: exact match after stripping newline, quote, and whitespace.
- `ndcg_rule_reward`: within each generation group, incorrect items receive
  rank-aware values only when the group contains at least one exact hit;
  otherwise the whole group receives zero.

Semantic embedding reward and SASRec/CF reward require additional artifacts and
runtime dependencies. They should be implemented as optional reward backends,
not mixed into the core rule reward.

## Evaluation Contract

Validation should follow `MiniOneRec/evaluate.py`:

- decode completions;
- use `split("Response:\n")[-1].strip()`-style normalization when full prompts
  leak into decoded strings;
- group outputs by beam count;
- compute HR@K and NDCG@K by the first rank where prediction equals target;
- track invalid generation rate so constrained decoding regressions are visible.

## Verl-GR Mapping

| MiniOneRec behavior | verl-GR implementation target |
| --- | --- |
| `SidDataset` | `verl_gr.recipes.minionerec.minionerec_dataset.MiniOneRecDataset` |
| `ConstrainedLogitsProcessor` | `verl_gr.workers.rollout.constraints.PrefixTrieConstraint` plus constrained `run_async_beam_search` |
| `rule_reward`, `ndcg_rule_reward` | `verl_gr.recipes.minionerec.minionerec_reward.compute_score` |
| train/eval grouped generations | verl GRPO rollout `n` expanded by `beam_width` and validation `all_beams` |
| train-time HR/NDCG logging | `verl_gr.recipes.minionerec.minionerec_trainer.minionerec_validate` |

## Training launch

- **Recommended**: `bash scripts/run_minionerec_grpo_rl_aligned.sh` (DDP, aligned with `MiniOneRec/rl.sh`).
- **Generic**: `bash scripts/run_minionerec_grpo.sh` with `CONFIG_NAME=minionerec/grpo_trainer_ddp`.
- Script index: [scripts/README.md](../../scripts/README.md).
- PR-level summary vs `main`: [minionerec_pr_changes.md](./minionerec_pr_changes.md).

## Performance optimizations (vs naive verl port)

| Optimization | Implementation |
| --- | --- |
| Completion-only logprob | `CompletionOnlyLogprobMixin` — ref uses `logits_to_keep` (padded); actor uses rmpad + completion indices. |
| Optimizer | `paged_adamw_32bit` (`verl_gr/workers/optimizer.py`) for actor. |
| Skip old logprob forward | `minionerec_reinforce` + `RLTrainer._compute_old_log_prob` bypass. |
| Memory | `use_remove_padding`, `entropy_from_logits_with_chunking`, `entropy_checkpointing`. |
| Profiling | Shared NVTX names; `scripts/compare_nsys_nvtx.py` for A/B vs MiniOneRec traces. |

## Reward shaping (training)

Configured under `task.reward_penalties` in `grpo_trainer_ddp.yaml`:

- `empty_completion: -1.0`
- `invalid_sid: -0.5`

Applied in `compute_group_training_rewards()` during trainer reward postprocess (not in per-sample `compute_score` alone).

## Known Intentional Differences

- TRL `ReReTrainer._prepare_inputs` is not reused; verl owns rollout,
  log-prob recomputation, reference policy, and advantage calculation.
- HuggingFace full-vocabulary logits masking is not available in the current
  async vLLM path. The first implementation filters vLLM top-logprobs in Python
  and records empty/missed constraint statistics.
- Remote experiments remain required for throughput and parity validation
  because local dependencies and hardware cannot run MiniOneRec training.
