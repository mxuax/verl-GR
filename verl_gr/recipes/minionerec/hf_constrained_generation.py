"""MiniOneRec HF constrained beam generation.

Mirrors original ``MiniOneRec/minionerec_trainer.py`` HF ``model.generate()``
beam sampling / deterministic beam with prefix-trie constraint.

Provides:
- ``HfConstrainedBeamGenerator``  — per-prompt constrained beam generate
- ``generate_constrained_batch``   — batched wrapper that returns DataProto tensors
"""

from __future__ import annotations

import logging
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
    ):
        self._tokenizer = tokenizer
        self._beam_width = beam_width
        self._val_beam_width = val_beam_width
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._length_penalty = length_penalty
        self._eos_token_id = tokenizer.eos_token_id
        self._pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        if prefix_index is None:
            prefix_index = 4 if "gpt2" in str(type(tokenizer)).lower() else 3
        self._prefix_index = prefix_index

        self._hash_dict = _build_hash_dict(info_file, tokenizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_train(self, model: torch.nn.Module, prompts: list[str]) -> list[HfBeamOutput]:
        """Training: constrained stochastic beam sampling (do_sample=True)."""
        return self._generate(model, prompts, beam_width=self._beam_width, do_sample=True)

    def generate_eval(self, model: torch.nn.Module, prompts: list[str]) -> list[HfBeamOutput]:
        """Evaluation: constrained deterministic beam (do_sample=False)."""
        return self._generate(model, prompts, beam_width=self._val_beam_width, do_sample=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate(
        self, model: torch.nn.Module, prompts: list[str], *, beam_width: int, do_sample: bool
    ) -> list[HfBeamOutput]:
        """Batched constrained beam generation — all prompts in one ``model.generate()`` call."""
        if not prompts:
            return []

        device = next(model.parameters()).device

        # Tokenize and pad all prompts to the same length
        encoded = self._tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True)
        input_ids = encoded["input_ids"].to(device)
        attn_mask = encoded["attention_mask"].to(device)
        n_prompts = input_ids.shape[0]
        prompt_lens = attn_mask.sum(dim=1).tolist()

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
            pad_token_id=self._pad_token_id,
            eos_token_id=self._eos_token_id,
        )

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                generation_config=gen_config,
                logits_processor=logits_processor,
                output_scores=False,
                return_dict_in_generate=True,
                use_cache=True,
            )

        # output.sequences: (n_prompts * beam_width, total_len)
        seqs = output.sequences
        total_seqs = seqs.shape[0]

        outputs: list[HfBeamOutput] = []
        for p_idx in range(n_prompts):
            prompt_len = int(prompt_lens[p_idx])
            completions: list[list[int]] = []
            for b_idx in range(beam_width):
                seq_idx = p_idx * beam_width + b_idx
                gen_ids = seqs[seq_idx, prompt_len:].tolist()
                clean_ids = []
                for tid in gen_ids:
                    if tid == self._pad_token_id:
                        break
                    clean_ids.append(tid)
                completions.append(clean_ids)
            decoded = [self._tokenizer.decode(c, skip_special_tokens=True) for c in completions]
            outputs.append(HfBeamOutput(
                prompt_token_ids=input_ids[p_idx, :prompt_len].tolist(),
                response_token_ids=completions,
                decoded_completions=decoded,
                beam_width=beam_width,
            ))
        return outputs


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
