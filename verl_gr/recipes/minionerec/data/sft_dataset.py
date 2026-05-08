"""MiniOneRec SFT dataset — supervised fine-tuning with CE loss + loss_mask.

Converts CSV/parquet recommendation data into the format expected by
verl's ``SFTTrainer`` (``input_ids``, ``attention_mask``, ``position_ids``,
``loss_mask``).  Builds three sub-tasks to match the original MiniOneRec
``sft.py`` / ``data.py`` behaviour:

* **SidSFT**  — predict the next semantic ID from an SID interaction history
* **Title2sid / Description2sid** — map item text to semantic IDs
* **FusionSeqRec** — predict the next item *title* from an SID history
  (matches ``FusionSeqRecDataset`` from the original ``data.py``)

Every sample is wrapped with a task-specific instruction prefix, matching
the original ``data.py`` ``pre()`` format.  BOS is prepended to the
instruction and EOS is appended to the target.

SID tokens are read from ``config.sid_index_path`` and injected into the
tokenizer via ``add_tokens`` (only individual component tokens, matching
the original ``TokenExtender``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import datasets
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl_gr.recipes.minionerec.minionerec_format import (
    build_title2sid_prompt,
    parse_maybe_list,
    sample_records,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Instruction prefixes — mirror the original data.py pre() wrappers
# ------------------------------------------------------------------

_INSTRUCTION_HEADER = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
)

_SID_INSTRUCTION = "### Instruction:\nCan you predict the next possible item that the user may expect?\n\n"
_ALIGN_INSTRUCTION = "### Instruction:\nAnswer the question about item identification.\n\n"
_FUSION_INSTRUCTION = "### Instruction:\nCan you recommend the next item for the user based on their interaction history?\n\n"


def _wrap_instruction(prompt: str, instruction: str) -> str:
    """Return ``instruction_header + instruction + user_input``."""
    return _INSTRUCTION_HEADER + instruction + prompt


# ------------------------------------------------------------------
# FusionSeqRec prompt — SID history → item title (original data.py)
# ------------------------------------------------------------------

def _build_fusion_prompt(history_item_sid: list[Any]) -> str:
    history = ", ".join(str(item) for item in history_item_sid)
    return (
        "### User Input:\n"
        f"The user has interacted with items {history} in chronological order. "
        "Try to predict the next item.\n\n"
        "### Response:\n"
    )


class MiniOneRecSFTDataset(Dataset):
    """SFT dataset that mirrors the MiniOneRec ``sft.py`` multi-task setup.

    Returns every sample as a dict with:

    * ``input_ids``
    * ``attention_mask``
    * ``position_ids``
    * ``loss_mask``  — ``1`` on target tokens, ``0`` on prompt / padding
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: ProcessorMixin | None = None,
        max_samples: int = -1,
    ) -> None:
        if processor is not None:
            logger.warning("MiniOneRecSFTDataset ignores processor / multimodal handling.")
        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]

        self._data_files = [os.path.expanduser(str(p)) for p in parquet_files]
        self._tokenizer = tokenizer
        self._config = config

        self._max_length = int(config.get("max_length", 512))
        self._truncation = config.get("truncation", "right")
        # Mirror original sft.py: set pad_token = eos_token explicitly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self._pad_token_id = tokenizer.pad_token_id
        self._shuffle = config.get("shuffle", False)
        self._seed = config.get("seed", None)
        self._category = config.get("category", "Industrial_and_Scientific")
        self._include_alignment = bool(config.get("include_alignment_tasks", True))
        self._sid_index_path = config.get("sid_index_path")
        self._item_meta_path = config.get("item_meta_path")
        self._fusion_sample = int(config.get("fusion_sample", 10000))
        self._cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/sft"))

        # NOTE: SID token extension (tokenizer.add_tokens + model.resize_token_embeddings)
        # must be done OFFLINE before SFT training because verl's SFTTrainer loads the model
        # inside the engine before the dataset is created.  Use the companion script:
        #   python -m verl_gr.recipes.minionerec.data.expand_vocab \
        #       --base_model /path/to/model --sid_index_path /path/to/index.json --output /path/to/expanded_model
        # Then set model.path=<expanded_model> in the run script.
        self._records: list[dict[str, Any]] = []
        self._build_records(max_samples)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_file(self, path: str) -> datasets.Dataset:
        local = copy_to_local(path, cache_dir=self._cache_dir, use_shm=self._config.get("use_shm", False))
        suffix = os.path.splitext(local)[1].lower()
        if suffix == ".parquet":
            return datasets.load_dataset("parquet", data_files=local)["train"]
        if suffix == ".csv":
            return datasets.load_dataset("csv", data_files=local)["train"]
        raise ValueError(f"Unsupported file type: {local}")

    def _build_records(self, max_samples: int) -> None:
        frames = [self._load_file(f) for f in self._data_files]
        df = datasets.concatenate_datasets(frames)
        logger.info("MiniOneRec SFT — source rows: %d", len(df))

        records: list[dict[str, Any]] = []
        records.extend(self._build_sid_records(df))

        if self._include_alignment:
            records.extend(self._build_title2sid())
            records.extend(self._build_fusion_records(df))

        if self._shuffle:
            import random

            rng = random.Random(self._seed)
            rng.shuffle(records)

        if max_samples > 0 and max_samples < len(records):
            records = records[:max_samples]

        self._records = records
        logger.info("MiniOneRec SFT — total records: %d", len(records))

    # -- sub-task builders -------------------------------------------------

    @staticmethod
    def _build_sid_records(df: datasets.Dataset) -> list[dict[str, Any]]:
        """SidSFT: SID history → next SID (original SidSFTDataset)."""
        out: list[dict[str, Any]] = []
        for row in df:
            row = dict(row)
            history = parse_maybe_list(row.get("history_item_sid"))
            target = str(row.get("item_sid", "")).strip()
            if not history or not target:
                continue
            # Original SidSFTDataset pre() formats output as: {target_sid}\n
            out.append({
                "instruction": _SID_INSTRUCTION,
                "prompt": _build_sid_prompt_inner(history),
                "target": target + "\n",
                "task": "sid",
            })
        return out

    def _build_title2sid(self) -> list[dict[str, Any]]:
        """Title2sid / Description2sid (original SidItemFeatDataset)."""
        if not self._sid_index_path or not self._item_meta_path:
            return []

        with open(self._item_meta_path, encoding="utf-8") as fh:
            item_feat = json.load(fh)
        with open(self._sid_index_path, encoding="utf-8") as fh:
            indices = json.load(fh)

        title2sid: dict[str, str] = {}
        for item_id, sids in indices.items():
            if item_id not in item_feat or len(sids) < 3:
                continue
            sid = str(sids[0]) + str(sids[1]) + str(sids[2])
            title = str(item_feat[item_id].get("title", "")).strip()
            if title:
                title2sid[title] = sid

        records: list[dict[str, Any]] = []
        # title2sid — "Which item has the title: X?" → SID
        for text, sid in title2sid.items():
            prompt, _ = build_title2sid_prompt("title2sid", text)
            records.append({
                "instruction": _ALIGN_INSTRUCTION,
                "prompt": prompt,
                "target": sid + "\n",
                "task": "title2sid",
            })
        # sid2title — "What is the title of item with ID X?" → title
        for title, sid in title2sid.items():
            prompt = _build_sid2title_prompt(sid)
            records.append({
                "instruction": _ALIGN_INSTRUCTION,
                "prompt": prompt,
                "target": title + "\n",
                "task": "sid2title",
            })
        return records

    def _build_fusion_records(self, df: datasets.Dataset) -> list[dict[str, Any]]:
        """FusionSeqRec: SID history → next item title (original FusionSeqRecDataset)."""
        records: list[dict[str, Any]] = []
        for row in df:
            row = dict(row)
            history_sid = parse_maybe_list(row.get("history_item_sid"))
            target_title = str(row.get("item_title", "")).strip()
            if not history_sid or not target_title:
                continue
            prompt = _build_fusion_prompt(history_sid)
            records.append({
                "instruction": _FUSION_INSTRUCTION,
                "prompt": prompt,
                "target": target_title + "\n",
                "task": "fusion",
            })
        return sample_records(records, self._fusion_sample, seed=self._seed)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rec = self._records[index]
        full_prompt = _wrap_instruction(rec["prompt"], rec["instruction"])
        target = rec["target"]

        # Mirror original: BOS at start of instruction, EOS after target
        prompt_enc = self._tokenizer(full_prompt, add_special_tokens=True, return_tensors="pt")
        target_enc = self._tokenizer(target, add_special_tokens=False, return_tensors="pt")

        prompt_ids = prompt_enc["input_ids"][0]
        target_ids = target_enc["input_ids"][0]

        # Append EOS to target (mirrors original bos=False, eos=True)
        eos = torch.tensor([self._tokenizer.eos_token_id], dtype=target_ids.dtype)
        target_ids = torch.cat([target_ids, eos], dim=0)

        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
        # Loss only on target tokens (including EOS) — matches original
        # labels=[-100]*input_prompt_len + tokens[input_prompt_len:]
        loss_mask[len(prompt_ids):] = 1

        total = len(input_ids)
        if total > self._max_length:
            if self._truncation == "left":
                input_ids = input_ids[-self._max_length:]
                attention_mask = attention_mask[-self._max_length:]
                loss_mask = loss_mask[-self._max_length:]
            else:
                raise ValueError(
                    f"Sample length {total} exceeds max_length {self._max_length}. "
                    f"Use truncation='left' to handle this."
                )
        elif total < self._max_length:
            pad_len = self._max_length - total
            input_ids = torch.cat([torch.full((pad_len,), self._pad_token_id, dtype=input_ids.dtype), input_ids])
            attention_mask = torch.cat([torch.zeros(pad_len, dtype=attention_mask.dtype), attention_mask])
            loss_mask = torch.cat([torch.zeros(pad_len, dtype=loss_mask.dtype), loss_mask])

        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


# --------------------------------------------------------------------------
# Helper: SidSFT prompt (inline, matches original SidDataset prompt format)
# --------------------------------------------------------------------------

def _build_sid2title_prompt(sid: str) -> str:
    return (
        "### User Input:\n"
        f"What is the title of the item with ID {sid}?\n\n"
        "### Response:\n"
    )


def _build_sid_prompt_inner(history_item_sid: list[Any]) -> str:
    history = ", ".join(str(item) for item in history_item_sid)
    return (
        "### User Input:\n"
        f"The user has interacted with items {history} in chronological order. "
        "Can you predict the next possible item that the user may expect?\n\n"
        "### Response:\n"
    )
