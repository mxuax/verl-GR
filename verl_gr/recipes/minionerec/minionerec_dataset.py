"""MiniOneRec dataset adapter following the original SidDataset behavior."""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Optional

import datasets
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl_gr.recipes.minionerec.minionerec_format import (
    build_seq_title2sid_prompt,
    build_sid_prompt,
    build_title2sid_prompt,
    maybe_parse_description,
    parse_maybe_list,
    sample_records,
)

logger = logging.getLogger(__name__)

MINIONEREC_SOURCE = "minionerec"

CATEGORY_DESCRIPTIONS = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}


def build_minionerec_record(
    *,
    prompt: str,
    target: str,
    history_key: str,
    task: str,
    index: int | None = None,
    dedup: bool = False,
) -> dict[str, Any]:
    target_sid = target.strip()
    return {
        "prompt": prompt,
        "reward_model": {"ground_truth": target, "style": "rule"},
        "source": MINIONEREC_SOURCE,
        "data_source": MINIONEREC_SOURCE,
        "extra_info": {
            "history_key": history_key,
            "target_sid": target_sid,
            "dedup": dedup,
            "task": task,
            **({"index": index} if index is not None else {}),
        },
    }


def build_sid_record(row: dict[str, Any], *, index: int | None = None) -> dict[str, Any]:
    history_item_sid = parse_maybe_list(row.get("history_item_sid"))
    if not history_item_sid:
        raise ValueError("MiniOneRec sample has empty history_item_sid.")

    target_sid = str(row.get("item_sid", "")).strip()
    if not target_sid:
        raise ValueError("MiniOneRec sample has empty item_sid.")

    prompt, history_key = build_sid_prompt(history_item_sid)
    target = f"{target_sid}\n"
    history_item_ids = parse_maybe_list(row.get("history_item_id"))
    last_history_item_id = history_item_ids[-1] if history_item_ids else None
    dedup = str(row.get("item_id")) == str(last_history_item_id) if last_history_item_id is not None else False

    return build_minionerec_record(
        prompt=prompt,
        target=target,
        history_key=history_key,
        task="sid",
        index=index,
        dedup=dedup,
    )


def extract_minionerec_prompt_fields(row: dict[str, Any], *, prompt_key: str) -> dict[str, Any]:
    """Build prompt/reward fields compatible with verl reward routing."""

    record = build_sid_record(row)
    row[prompt_key] = record["prompt"]
    row["reward_model"] = record["reward_model"]
    row["source"] = row.get("source", record["source"])
    row["data_source"] = row.get("data_source", row["source"])
    row["extra_info"] = {**(row.get("extra_info") or {}), **record["extra_info"]}
    return row


class MiniOneRecDataset(Dataset):
    """Dataset adapter for MiniOneRec CSV/parquet files.

    The original MiniOneRec trainer tokenizes plain prompt strings directly
    rather than applying chat templates. This adapter preserves that behavior
    and also exposes `raw_prompt_text` for the custom async agent loop.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ) -> None:
        if processor is not None:
            logger.warning("MiniOneRecDataset ignores processor/chat-template multimodal handling.")
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(list(data_files))
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)
        self.category = config.get("category", "Industrial_and_Scientific")
        self.category_text = CATEGORY_DESCRIPTIONS.get(self.category, self.category)
        requested_alignment = bool(config.get("include_alignment_tasks", True))
        include_alignment_tasks_for_val = config.get("include_alignment_tasks_for_val", False)
        self.is_val_split = self._is_val_split(self.original_data_files, config.get("val_files"))
        self.include_alignment_tasks = (
            bool(include_alignment_tasks_for_val) if self.is_val_split else requested_alignment
        )
        self.sid_index_path = config.get("sid_index_path")
        self.item_meta_path = config.get("item_meta_path")
        self.seq_title_sample = int(config.get("seq_title_sample", 10000))
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        if self.num_workers is not None:
            self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.serialize_dataset = False

        self._download()
        self._read_files_and_tokenize()

    @staticmethod
    def _normalize_paths(paths: Any) -> list[str]:
        if paths is None:
            return []
        if isinstance(paths, (list, ListConfig)):
            candidates = list(paths)
        else:
            candidates = [paths]
        normalized = []
        for value in candidates:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            normalized.append(os.path.normcase(os.path.abspath(os.path.expanduser(text))))
        return normalized

    @classmethod
    def _is_val_split(cls, data_files: list[str], val_files_cfg: Any) -> bool:
        data_paths = cls._normalize_paths(data_files)
        val_paths = cls._normalize_paths(val_files_cfg)
        return bool(data_paths and val_paths and set(data_paths) == set(val_paths))

    def _download(self, use_origin_parquet: bool = False) -> None:
        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, data_file in enumerate(target_files):
            local_path = copy_to_local(src=data_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
            target_files[idx] = local_path
        if use_origin_parquet:
            self.data_files = target_files

    def _load_file(self, data_file: str) -> datasets.Dataset:
        suffix = os.path.splitext(str(data_file))[1].lower()
        if suffix == ".parquet":
            return datasets.load_dataset("parquet", data_files=data_file)["train"]
        if suffix == ".csv":
            return datasets.load_dataset("csv", data_files=data_file)["train"]
        raise ValueError(f"Unsupported MiniOneRec data file type: {data_file}")

    def _read_files_and_tokenize(self) -> None:
        dataframes = [self._load_file(data_file) for data_file in self.data_files]
        source_dataframe = datasets.concatenate_datasets(dataframes)
        logger.info("MiniOneRec source dataset len: %s", len(source_dataframe))
        print(f"[MiniOneRec] source dataset (CSV rows): {len(source_dataframe)}", flush=True)

        records = self._build_sid_records(source_dataframe)
        if self.include_alignment_tasks:
            title2sid_records = self._build_title2sid_records()
            seq_records = self._build_seq_title2sid_records(source_dataframe)
            print(f"[MiniOneRec] title2sid+desc2sid records: {len(title2sid_records)}", flush=True)
            print(f"[MiniOneRec] seq_title2sid records: {len(seq_records)}", flush=True)
            records.extend(title2sid_records)
            records.extend(seq_records)

        self.dataframe = datasets.Dataset.from_list(records)
        print(f"[MiniOneRec] combined (before filter): {len(self.dataframe)}", flush=True)
        logger.info("MiniOneRec combined dataset len: %s", len(self.dataframe))
        if self.shuffle:
            self.dataframe = self.dataframe.shuffle(seed=self.seed)
        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            self.dataframe = self.dataframe.select(list(range(self.max_samples)))
        logger.info("MiniOneRec processed dataset len: %s", len(self.dataframe))

    def _build_sid_records(self, dataframe: datasets.Dataset) -> list[dict[str, Any]]:
        return [build_sid_record(dict(row), index=idx) for idx, row in enumerate(dataframe)]

    def _build_title2sid_records(self) -> list[dict[str, Any]]:
        if not self.sid_index_path or not self.item_meta_path:
            logger.warning("Skip RLTitle2SidDataset parity task: sid_index_path or item_meta_path is missing.")
            return []
        with open(self.item_meta_path, "r", encoding="utf-8") as f:
            item_feat = json.load(f)
        with open(self.sid_index_path, "r", encoding="utf-8") as f:
            indices = json.load(f)

        title2sid: dict[str, str] = {}
        description2sid: dict[str, str] = {}
        for item_id, sids in indices.items():
            if item_id not in item_feat or len(sids) < 3:
                continue
            combined_sid = str(sids[0]) + str(sids[1]) + str(sids[2])
            title = str(item_feat[item_id].get("title", ""))
            description = maybe_parse_description(item_feat[item_id].get("description", ""))
            title2sid[title] = combined_sid
            description2sid[description] = combined_sid

        records: list[dict[str, Any]] = []
        for task, mapping in (("title2sid", title2sid), ("description2sid", description2sid)):
            for text, sid in mapping.items():
                prompt, history_key = build_title2sid_prompt(task, text)
                records.append(
                    build_minionerec_record(
                        prompt=prompt,
                        target=f"{sid}\n",
                        history_key=history_key,
                        task=task,
                    )
                )
        return records

    def _build_seq_title2sid_records(self, dataframe: datasets.Dataset) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for idx, row in enumerate(dataframe):
            row = dict(row)
            history_item_title = parse_maybe_list(row.get("history_item_title"))
            target_sid = str(row.get("item_sid", "")).strip()
            is_duplicate = False
            if "history_item_id" in row:
                history_item_id = parse_maybe_list(row.get("history_item_id"))
                last_history_item_id = history_item_id[-1] if history_item_id else None
                is_duplicate = str(row.get("item_id")) == str(last_history_item_id) if last_history_item_id is not None else False
            prompt, history_key = build_seq_title2sid_prompt(history_item_title)
            records.append(
                build_minionerec_record(
                    prompt=prompt,
                    target=f"{target_sid}\n",
                    history_key=history_key,
                    task="seq_title2sid",
                    index=idx,
                    dedup=is_duplicate,
                )
            )
        return sample_records(records, self.seq_title_sample, seed=self.seed)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        def doc_length(doc: dict[str, Any]) -> int:
            return len(tokenizer.encode(doc[prompt_key], add_special_tokens=False))

        filtered = dataframe.filter(
            lambda doc: doc_length(doc) <= self.max_prompt_length - 10,
            num_proc=self.num_workers,
            desc=f"Filtering MiniOneRec prompts longer than {self.max_prompt_length - 10} tokens",
        )
        logger.info("MiniOneRec filtered dataset len: %s", len(filtered))
        return filtered

    def resume_dataset_state(self) -> None:
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            logger.warning("resume with serialized dataloader, consider restarting from scratch for better perf")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = dict(self.dataframe[index])
        raw_prompt = str(row[self.prompt_key])
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)
        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = position_ids[0]
        row["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)[-self.max_prompt_length :]
        # verl's AgentLoopWorker postprocess expects a `raw_prompt` field.
        # MiniOneRec prompts are plain strings, while chat recipes use message lists.
        row["raw_prompt"] = raw_prompt
        row["raw_prompt_text"] = raw_prompt
        row["index"] = (row.get("extra_info") or {}).get("index", index)
        row["tools_kwargs"] = {}
        row["interaction_kwargs"] = {}
        if "uid" not in row:
            row["uid"] = str(row["index"])
        return row

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()
