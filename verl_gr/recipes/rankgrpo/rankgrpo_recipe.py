"""Minimal Rank-GRPO recipe built on the verl/verl_gr stack."""

from __future__ import annotations

import ast
import copy
import logging
import os
from collections import defaultdict
from typing import Any

import datasets
import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from verl_gr.recipes.rankgrpo.rankgrpo_reward import compute_score
from verl_gr.recipes.rankgrpo.rankgrpo_tokenizer import build_rankgrpo_tokenizer_and_processor

logger = logging.getLogger(__name__)

__all__ = ["RankGRPODataset", "RankGRPOTask", "collate_fn", "compute_score"]


def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    non_tensors: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)
    batch: dict[str, Any] = {}
    for key, value in tensors.items():
        batch[key] = torch.stack(value, dim=0)
    for key, value in non_tensors.items():
        batch[key] = np.array(value, dtype=object)
    return batch


def _maybe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return ast.literal_eval(stripped)
    except Exception:
        return value


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    value = _maybe_literal(value)
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    messages: list[dict[str, str]] = []
    for message in value or []:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(segment.get("text", "") for segment in content if segment.get("type") == "text")
        messages.append({"role": message.get("role", "user"), "content": str(content)})
    return messages


class RankGRPODataset(Dataset):
    """Text-only Rank-GRPO dataset using verl's RL dataset contract."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: ProcessorMixin | None = None,
        max_samples: int = -1,
    ) -> None:
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]
        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(list(data_files))
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.max_prompt_length = config.get("max_prompt_length", 2048)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        if self.num_workers is not None:
            self.num_workers = min(self.num_workers, os.cpu_count())
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)
        self.use_shm = config.get("use_shm", False)
        rank_cfg = config.get("rankgrpo", {}) or {}
        self.use_chat_template = rank_cfg.get("use_chat_template", True)
        self.add_generation_prompt = rank_cfg.get("add_generation_prompt", True)
        self.rec_num = int(rank_cfg.get("rec_num", 20))
        self.data_source_key = config.get("data_source_key", "source")
        self.serialize_dataset = False

        self._download()
        self._read_files()

    def _download(self, use_origin_parquet: bool = False) -> None:
        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, parquet_file in enumerate(target_files):
            target_files[idx] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
        if use_origin_parquet:
            self.data_files = target_files

    def _read_files(self) -> None:
        frames = []
        for path in self.data_files:
            if os.path.isdir(path):
                if os.path.exists(os.path.join(path, "state.json")):
                    frames.append(datasets.load_from_disk(path))
                elif os.path.exists(os.path.join(path, "train", "state.json")):
                    frames.append(datasets.load_from_disk(os.path.join(path, "train")))
                else:
                    raise FileNotFoundError(
                        f"RankGRPO dataset directory must be a HuggingFace dataset split "
                        f"or contain a train split: {path}"
                    )
            else:
                frames.append(datasets.load_dataset("parquet", data_files=path)["train"])
        self.dataframe = datasets.concatenate_datasets(frames)
        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(len(self.dataframe), size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
        if self.filter_overlong_prompts:
            filter_fn = lambda row: len(self._encode_prompt(dict(row))) <= self.max_prompt_length - 10
            try:
                self.dataframe = self.dataframe.filter(
                    filter_fn,
                    num_proc=self.num_workers,
                    desc=f"Filtering prompts longer than {self.max_prompt_length - 10} tokens",
                )
            except TypeError as exc:
                if "cannot pickle" not in str(exc):
                    raise
                logger.warning("Falling back to single-process prompt filtering: %s", exc)
                self.dataframe = self.dataframe.filter(
                    filter_fn,
                    num_proc=None,
                    desc=f"Filtering prompts longer than {self.max_prompt_length - 10} tokens",
                )
        logger.info("RankGRPO dataset len: %s", len(self.dataframe))

    def resume_dataset_state(self) -> None:
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files()

    def __len__(self) -> int:
        return len(self.dataframe)

    def _build_messages(self, row: dict[str, Any]) -> list[dict[str, str]]:
        if "messages" in row and row.get("messages") is not None:
            messages = _normalize_messages(row.get("messages"))
            return messages[:-1] if len(messages) > 1 and messages[-1].get("role") == "assistant" else messages
        return _normalize_messages(row.get(self.prompt_key, ""))

    def _format_prompt(self, row: dict[str, Any]) -> str:
        raw_prompt = row.get(self.prompt_key, "")
        if self.use_chat_template:
            messages = self._build_messages(row)
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=self.add_generation_prompt,
                tokenize=False,
            )
        return str(raw_prompt)

    def _encode_prompt(self, row: dict[str, Any]) -> list[int]:
        return self.tokenizer.encode(self._format_prompt(row), add_special_tokens=False)

    def _build_reward_model(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        reward_model = _maybe_literal(row.get("reward_model", {})) or {}
        if not isinstance(reward_model, dict):
            reward_model = {}
        for key in ("ground_truth", "groundtruth_with_release_year", "seen_titles"):
            if key in row and key not in reward_model:
                reward_model[key] = _maybe_literal(row.get(key))
        if "ground_truth" not in reward_model and "answer" in row:
            reward_model["ground_truth"] = _maybe_literal(row.get("answer"))
        if "ground_truth" not in reward_model and "groundtruth_with_release_year" in reward_model:
            reward_model["ground_truth"] = reward_model["groundtruth_with_release_year"]
        if "groundtruth_with_release_year" not in reward_model and "ground_truth" in reward_model:
            reward_model["groundtruth_with_release_year"] = reward_model["ground_truth"]
        reward_model.setdefault("ground_truth", [])
        reward_model.setdefault("rec_num", self.rec_num)
        reward_model.setdefault("style", "rule")
        reward_model.setdefault("index", index)
        return reward_model

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.dataframe[index])
        messages = self._build_messages(dict(row))
        raw_prompt = self._format_prompt(row)
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

        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = compute_position_id_with_mask(attention_mask)[0]
        row["raw_prompt_ids"] = self._encode_prompt(row)[-self.max_prompt_length :]
        row["raw_prompt"] = messages
        row["full_prompts"] = raw_prompt
        row["reward_model"] = self._build_reward_model(row, index)
        row["index"] = row.get("index", index)
        row.setdefault("uid", str(row["index"]))
        if "source" not in row and "data_source" not in row:
            row["data_source"] = "rankgrpo"
        elif "source" in row and "data_source" not in row:
            row["data_source"] = row["source"]
        elif "data_source" in row and "source" not in row:
            row["source"] = row["data_source"]
        row.setdefault("tools_kwargs", {})
        row.setdefault("interaction_kwargs", {})
        return row

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()


class RankGRPOTask:
    """Rank-GRPO task-specific runtime preparation."""

    @staticmethod
    def _normalize_layer_wrap_value(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, set):
            return sorted(str(item.__name__ if hasattr(item, "__name__") else item) for item in value)
        if isinstance(value, tuple):
            return list(value)
        return value

    def sanitize_fsdp2_wrap_policy(self, config) -> None:
        actor_rollout_ref = config.get("actor_rollout_ref")
        if actor_rollout_ref is None:
            return
        for role_name in ("actor", "ref"):
            role_cfg = actor_rollout_ref.get(role_name)
            if role_cfg is None or str(role_cfg.get("strategy", "")) != "fsdp2":
                continue
            fsdp_cfg = role_cfg.get("fsdp_config")
            if fsdp_cfg is None or fsdp_cfg.get("wrap_policy") is None:
                continue
            normalized = self._normalize_layer_wrap_value(fsdp_cfg.wrap_policy.get("transformer_layer_cls_to_wrap"))
            if normalized is not None:
                fsdp_cfg.wrap_policy["transformer_layer_cls_to_wrap"] = normalized

    def prepare(self, config) -> dict[str, Any]:
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        rank_cfg = config.data.get("rankgrpo", {}) or {}
        built = build_rankgrpo_tokenizer_and_processor(
            local_path,
            trust_remote_code=config.data.get("trust_remote_code", False),
            use_processor=rank_cfg.get("use_processor", False),
            rank_separator=rank_cfg.get("rank_separator", "\n"),
            force_pad_to_eos=rank_cfg.get("force_pad_to_eos", True),
        )

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2", "megatron"}:
            ray_worker_group_cls = RayWorkerGroup
            actor_rollout_cls = ActorRolloutRefWorker
            critic_worker = TrainingWorker
        else:
            raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")

        return {
            "tokenizer": built["tokenizer"],
            "processor": built["processor"],
            "rank_separator_token_ids": built["rank_separator_token_ids"],
            "actor_rollout_cls": actor_rollout_cls,
            "critic_worker": critic_worker,
            "reward_model_cfg": None,
            "ray_worker_group_cls": ray_worker_group_cls,
        }

