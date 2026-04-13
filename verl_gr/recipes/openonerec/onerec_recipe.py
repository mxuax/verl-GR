"""Local OpenOneRec dataset/reward helpers for GRPO runtime."""

from __future__ import annotations

import ast
import copy
import logging
import os
import re
from importlib import import_module
from typing import Any

import torch
from torch.utils.data import Dataset

from verl_gr.contracts.tokenizer_contract import ChatMessage, PromptPackage

logger = logging.getLogger(__name__)


class OneRecDataset(Dataset):
    """OpenOneRec-compatible dataset for GRPO data formatting."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        if not isinstance(data_files, list):
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
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.serialize_dataset = False

        self.enable_think = config.get("enable_think", False)
        self.enable_nonthink = config.get("enable_nonthink", False)
        self.use_force_prefix = config.get("use_force_prefix", False)
        if self.enable_think and self.enable_nonthink:
            raise ValueError("enable_think and enable_nonthink cannot both be True")

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet: bool = False) -> None:
        copy_to_local = getattr(import_module("verl.utils.fs"), "copy_to_local")
        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, parquet_file in enumerate(target_files):
            target_files[idx] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
        if use_origin_parquet:
            self.data_files = target_files

    def _read_files_and_tokenize(self) -> None:
        datasets = import_module("datasets")
        np = import_module("numpy")
        dataframes = []
        for parquet_file in self.data_files:
            file_path = str(parquet_file)
            if file_path.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=file_path)["train"]
            elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=file_path)["train"]
            else:
                raise ValueError(f"Unsupported file format: {parquet_file}")
            dataframes.append(dataframe)

        self.dataframe = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())

        self.dataframe = self.dataframe.map(
            self._extract_prompt_fields,
            num_proc=self.num_workers,
            desc="Extract prompts and reward annotations",
        )
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe):
        if not self.filter_overlong_prompts:
            return dataframe

        def doc_length(doc: dict[str, Any]) -> int:
            tokenized_prompt = self.tokenizer.apply_chat_template(
                doc[self.prompt_key],
                add_generation_prompt=True,
                tokenize=True,
            )
            return len(tokenized_prompt)

        return dataframe.filter(
            lambda doc: doc_length(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

    def _extract_prompt_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        raw_messages = row.get("messages")
        if isinstance(raw_messages, str):
            messages = ast.literal_eval(raw_messages)
        else:
            messages = raw_messages or []

        clean_chats: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if isinstance(content, str):
                text_content = content
            else:
                text_chunks = []
                for segment in content or []:
                    if isinstance(segment, dict) and segment.get("type") == "text":
                        text_chunks.append(segment.get("text", ""))
                text_content = "".join(text_chunks)
            clean_chats.append({"role": role, "content": text_content})

        if not clean_chats:
            raise ValueError("Sample has empty messages; expected OpenOneRec chat format.")

        prompt_messages = copy.deepcopy(clean_chats[:-1])
        for message in prompt_messages:
            if message.get("role") == "user":
                if self.enable_think:
                    message["content"] = f"{message['content']}/think"
                elif self.enable_nonthink:
                    message["content"] = f"{message['content']}/no_think"

        ground_truth_message = clean_chats[-1]["content"]
        prompt_package = PromptPackage(
            prompt_messages=tuple(
                ChatMessage(
                    role=str(message.get("role", "")),
                    content=str(message.get("content", "")),
                )
                for message in prompt_messages
            ),
            reward_payload={"ground_truth": ground_truth_message, "style": "rule"},
        )
        row[self.prompt_key] = [
            {"role": message.role, "content": message.content}
            for message in prompt_package.prompt_messages
        ]
        row["reward_model"] = dict(prompt_package.reward_payload)
        return row

    def resume_dataset_state(self) -> None:
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            logger.warning("Using serialized dataloader state; restarting from scratch is preferred.")

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = dict(self.dataframe[index])
        row["raw_prompt"] = copy.deepcopy(row[self.prompt_key])
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        extra_info = row.get("extra_info", {}) or {}
        row["index"] = extra_info.get("index", index)
        row["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})

        if "source" not in row and "data_source" not in row:
            row["data_source"] = "unknown"

        need_tools_kwargs = extra_info.get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not row["tools_kwargs"]:
            logger.warning(
                "tools_kwargs is empty for index %s, data source: %s",
                row["index"],
                row.get("data_source", row.get("source", "unknown")),
            )
        return row


_TUPLE_PATTERN = re.compile(r"<\|sid_begin\|>\s*(.*?)\s*<\|sid_end\|>", re.DOTALL)


def _extract_all_tuples(text: str) -> list[str]:
    return [match.strip() for match in _TUPLE_PATTERN.findall(text or "") if match.strip()]


def think_format_reward(prediction: str) -> float:
    return float("</think>" in (prediction or ""))


def partial_hit_reward(prediction: str, ground_truth: str) -> float:
    pred_items = set(_extract_all_tuples(prediction))
    gt_items = set(_extract_all_tuples(ground_truth))
    if not pred_items or not gt_items:
        return 0.0
    return float(len(pred_items & gt_items) / max(len(gt_items), 1))


def hit_reward(prediction: str, ground_truth: str) -> float:
    pred_items = set(_extract_all_tuples(prediction))
    gt_items = set(_extract_all_tuples(ground_truth))
    return float(bool(pred_items and gt_items and pred_items == gt_items))


def first_sid_hit_reward(prediction: str, ground_truth: str) -> float:
    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    return float(pred_tuples[0] in set(gt_tuples))


def pass_rate(prediction: str, ground_truth: str) -> float:
    pred_set = set(_extract_all_tuples(prediction))
    gt_set = set(_extract_all_tuples(ground_truth))
    if not pred_set or not gt_set:
        return 0.0
    return float(len(pred_set & gt_set) > 0)


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],  # noqa: ARG001
) -> dict[str, float]:
    """Compute reward bundle aligned to OpenOneRec GRPO usage."""

    prediction = solution_str or ""
    format_reward_value = think_format_reward(prediction)
    partial_hit_reward_value = partial_hit_reward(prediction, ground_truth)
    hit_reward_value = hit_reward(prediction, ground_truth)
    pass_rate_value = pass_rate(prediction, ground_truth)
    pass_at_1_value = first_sid_hit_reward(prediction, ground_truth)
    return {
        "score": pass_at_1_value,
        "format_reward": format_reward_value,
        "partial_hit_reward": partial_hit_reward_value,
        "hit_reward": hit_reward_value,
        "pass_rate": pass_rate_value,
        "pass_at_1": pass_at_1_value,
    }

