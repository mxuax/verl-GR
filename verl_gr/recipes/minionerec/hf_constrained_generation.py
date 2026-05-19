"""MiniOneRec HF constrained beam generation.

Mirrors original ``MiniOneRec/minionerec_trainer.py`` HF ``model.generate()``
beam sampling / deterministic beam with prefix-trie constraint.

Provides:
- ``HfConstrainedBeamGenerator``  — per-prompt constrained beam generate
- ``generate_constrained_batch``   — batched wrapper that returns DataProto tensors
"""

from __future__ import annotations

import logging
import contextlib
from collections import defaultdict
from typing import Any

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

logger = logging.getLogger(__name__)


class HfConstrainedBeamGenerator:
    """Generate constrained beam completions using HF ``model.generate()``.

    Replicates the original MiniOneRec beam sampling / deterministic beam
    behaviour with a prefix-trie constraint.
    """

    def __init__(
        self,
        info_file: str,
        tokenizer: Any,
        *,
        beam_width: int = 16,
        val_beam_width: int = 50,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        length_penalty: float = 0.0,
        prefix_index: int | None = None,
        micro_batch_size: int = 16,
    ):
        self._tokenizer = tokenizer
        self._beam_width = beam_width
        self._val_beam_width = val_beam_width
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._length_penalty = length_penalty
        self._micro_batch_size = micro_batch_size
        self._eos_token_id = tokenizer.eos_token_id
        self._pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        if prefix_index is None:
            prefix_index = 4 if "gpt2" in str(type(tokenizer)).lower() else 3
        self._prefix_index = prefix_index

        self._tokenizer.padding_side = "left"  # decoder-only models require left-padding
        self._hash_dict = _build_hash_dict(info_file, tokenizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_train(
        self, model: torch.nn.Module, prompts: list[str], *, prompt_token_ids: list[list[int]] | None = None
    ) -> list[HfBeamOutput]:
        """Training: constrained stochastic beam sampling (do_sample=True)."""
        return self._generate(model, prompts, beam_width=self._beam_width, do_sample=True,
                              prompt_token_ids=prompt_token_ids)

    def generate_eval(
        self, model: torch.nn.Module, prompts: list[str], *, prompt_token_ids: list[list[int]] | None = None
    ) -> list[HfBeamOutput]:
        """Evaluation: constrained deterministic beam (do_sample=False)."""
        return self._generate(model, prompts, beam_width=self._val_beam_width, do_sample=False,
                              prompt_token_ids=prompt_token_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate(
        self, model: torch.nn.Module, prompts: list[str], *, beam_width: int, do_sample: bool,
        prompt_token_ids: list[list[int]] | None = None,
    ) -> list[HfBeamOutput]:
        """Batched constrained beam generation — micro-batched to control VRAM."""
        if not prompts:
            return []

        outputs: list[HfBeamOutput] = []
        micro_batch = self._micro_batch_size
        for batch_start in range(0, len(prompts), micro_batch):
            batch_end = min(batch_start + micro_batch, len(prompts))
            chunk = prompts[batch_start:batch_end]
            ids_chunk = prompt_token_ids[batch_start:batch_end] if prompt_token_ids else None
            outputs.extend(self._generate_chunk(model, chunk, beam_width=beam_width, do_sample=do_sample,
                                                 prompt_token_ids=ids_chunk))
        return outputs

    def _generate_chunk(
        self, model: torch.nn.Module, prompts: list[str], *, beam_width: int, do_sample: bool,
        prompt_token_ids: list[list[int]] | None = None,
    ) -> list[HfBeamOutput]:
        device = next(model.parameters()).device

        if prompt_token_ids is not None and len(prompt_token_ids) == len(prompts):
            # Use pre-tokenized prompt IDs from the dataset (already truncated).
            # Left-pad to equal length.
            max_len = max(len(ids) for ids in prompt_token_ids)
            pad_id = self._pad_token_id
            input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long, device=device)
            attn_mask = torch.zeros((len(prompts), max_len), dtype=torch.long, device=device)
            for i, ids in enumerate(prompt_token_ids):
                L = len(ids)
                input_ids[i, max_len - L:] = torch.tensor(ids, dtype=torch.long, device=device)
                attn_mask[i, max_len - L:] = 1
            prompt_lens = [len(ids) for ids in prompt_token_ids]
        else:
            encoded = self._tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True)
            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded["attention_mask"].to(device)
            prompt_lens = attn_mask.sum(dim=1).tolist()
        n_prompts = input_ids.shape[0]

        logits_processor = LogitsProcessorList([
            _make_constraint_processor(
                hash_dict=self._hash_dict,
                prefix_index=self._prefix_index,
                num_beams=beam_width,
                eos_token_id=self._eos_token_id,
            )
        ])

        gen_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            length_penalty=self._length_penalty,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            do_sample=do_sample,
            temperature=self._temperature if do_sample else 1.0,
            top_k=None,
            top_p=None,
            pad_token_id=self._pad_token_id,
            eos_token_id=self._eos_token_id,
        )

        param_dtype = next(model.parameters()).dtype
        autocast_ctx = contextlib.nullcontext()
        if device.type in {"cuda", "npu"} and param_dtype in {torch.float16, torch.bfloat16}:
            autocast_ctx = torch.autocast(device_type=device.type, dtype=param_dtype)

        with torch.no_grad(), autocast_ctx:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                generation_config=gen_config,
                logits_processor=logits_processor,
                output_scores=False,
                return_dict_in_generate=True,
                use_cache=True,
            )

        seqs = output.sequences
        padded_input_len = input_ids.shape[1]  # uniform after left-padding
        chunk_outputs: list[HfBeamOutput] = []
        for p_idx in range(n_prompts):
            prompt_len = int(prompt_lens[p_idx])
            completions: list[list[int]] = []
            for b_idx in range(beam_width):
                seq_idx = p_idx * beam_width + b_idx
                # Slice from the uniform padded length, NOT individual prompt_len.
                # Left-padding means shorter prompts have PAD tokens before the real
                # prompt; using prompt_len would include real prompt tokens in the
                # completion.  Mirrors evaluate.py: sequences[:, maxLen:].
                gen_ids = seqs[seq_idx, padded_input_len:].tolist()
                # Keep up to (including) first EOS, matching original MiniOneRec
                # where ``completion_mask = (seq_idx <= eos_idx)``.
                # HF beam search pads with pad_token_id which equals eos_token_id
                # for Qwen2; stripping all of them turns "first token EOS" into
                # an empty response → response_mask=0 → no policy/KL gradient.
                if self._eos_token_id in gen_ids:
                    eos_pos = gen_ids.index(self._eos_token_id)
                    clean_ids = gen_ids[:eos_pos + 1]
                else:
                    clean_ids = gen_ids
                completions.append(clean_ids)
            decoded = [self._tokenizer.decode(c, skip_special_tokens=True) for c in completions]
            # Prompt tokens: slice from the RIGHT side of the padded input,
            # keeping only the actual prompt tokens (last prompt_len positions).
            prompt_start = padded_input_len - prompt_len
            chunk_outputs.append(HfBeamOutput(
                prompt_token_ids=input_ids[p_idx, prompt_start:padded_input_len].tolist(),
                response_token_ids=completions,
                decoded_completions=decoded,
                beam_width=beam_width,
            ))
        return chunk_outputs


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

class HfBeamOutput:
    __slots__ = ("prompt_token_ids", "response_token_ids", "decoded_completions", "beam_width")

    def __init__(self, prompt_token_ids, response_token_ids, decoded_completions, beam_width):
        self.prompt_token_ids = prompt_token_ids
        self.response_token_ids = response_token_ids
        self.decoded_completions = decoded_completions
        self.beam_width = beam_width


# ---------------------------------------------------------------------------
# Constraint: mirrors MiniOneRec evaluate.py hash_dict + LogitProcessor
# ---------------------------------------------------------------------------

def _build_hash_dict(info_file: str, tokenizer) -> dict[str, list[int]]:
    """Build prefix-trie constraint hash dict (evaluate.py:61-98)."""
    prefix_index = 4 if "gpt2" in str(type(tokenizer)).lower() else 3

    with open(info_file, encoding="utf-8") as f:
        info = f.readlines()
    semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
    info_semantic = [f"### Response:\n{_}" for _ in semantic_ids]
    prefixID = [tokenizer(_).input_ids for _ in info_semantic]

    def _hash(x):
        return '-'.join(str(_) for _ in x)

    hash_dict: dict[str, list[int]] = {}
    for ID in prefixID:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                h = _hash(ID[:i])
            else:
                h = _hash(ID[prefix_index:i])
            if h not in hash_dict:
                hash_dict[h] = []
            hash_dict[h].append(ID[i])
        _hash(ID[prefix_index:])

    for k in hash_dict:
        hash_dict[k] = sorted(set(hash_dict[k]))
    return hash_dict


class _ConstrainedLogitsProcessor(LogitsProcessor):
    """Exact mirror of MiniOneRec ``LogitProcessor.py:24-72``."""

    def __init__(self, hash_dict: dict[str, list[int]], prefix_index: int, num_beams: int, eos_token_id: int):
        self._hash_dict = hash_dict
        self._prefix_index = prefix_index
        self._num_beams = num_beams
        self._eos_token_id = eos_token_id
        self.count = 0

    @staticmethod
    def _get_hash(x):
        return '-'.join(str(_) for _ in x)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))

        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self._prefix_index:].tolist()
                else:
                    hash_key = sent[-self.count:].tolist()
                allowed = self._hash_dict.get(self._get_hash(hash_key), [])
                if not allowed:
                    mask[batch_id * self._num_beams + beam_id, self._eos_token_id] = 0
                else:
                    mask[batch_id * self._num_beams + beam_id, allowed] = 0

        self.count += 1
        return scores + mask


def _make_constraint_processor(hash_dict, prefix_index, num_beams, eos_token_id):
    return _ConstrainedLogitsProcessor(hash_dict, prefix_index, num_beams, eos_token_id)
