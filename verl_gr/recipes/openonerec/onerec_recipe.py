import ast
import copy
import functools
import logging
import os
import re
from collections import defaultdict
from importlib import import_module
from typing import Any, Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import verl.utils.torch_functional as verl_F
from verl.single_controller.ray import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.vision_utils import process_image, process_video
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

logger = logging.getLogger(__name__)

__all__ = ["collate_fn", "OneRecDataset", "compute_score"]


def build_hf_tokenizer_and_processor(
    model_path: str,
    *,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    """Build HuggingFace tokenizer and processor for OpenOneRec paths."""

    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(model_path, trust_remote_code=trust_remote_code, use_fast=True)
    return tokenizer, processor


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


def extract_prompt_fields(
    row: dict[str, Any],
    *,
    prompt_key: str,
    enable_think: bool,
    enable_nonthink: bool,
) -> dict[str, Any]:
    raw_messages = row.get("messages")
    if isinstance(raw_messages, str):
        messages = ast.literal_eval(raw_messages)
    else:
        messages = raw_messages or []

    clean_chats = [
        {
            "role": message.get("role"),
            "content": "".join(
                segment.get("text", "")
                for segment in message.get("content", [])
                if segment.get("type") == "text"
            ),
        }
        for message in messages
    ]
    if not clean_chats:
        raise ValueError("Sample has empty messages; please check data integrity.")

    prompt_messages = clean_chats[:-1]
    if enable_think:
        for message in prompt_messages:
            if message["role"] == "user":
                message["content"] = message["content"] + "/think"
    if enable_nonthink:
        for message in prompt_messages:
            if message["role"] == "user":
                message["content"] = message["content"] + "/no_think"

    ground_truth_message = clean_chats[-1]["content"]
    row[prompt_key] = prompt_messages
    row["reward_model"] = {"ground_truth": ground_truth_message, "style": "rule"}
    return row


class OneRecDataset(Dataset):
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ) -> None:
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
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)
        self.enable_think = config.get("enable_think", True)
        self.enable_nonthink = config.get("enable_nonthink", False)

        self.use_force_prefix = config.get("use_force_prefix", False)
        self._FORCE_PREFIX_CONTENT = "<think>\n</think><|sid_begin|>"
        if self.enable_think and self.enable_nonthink:
            raise ValueError("enable_think and enable_nonthink cannot be both True")

        configured_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = configured_workers
        if self.num_workers is not None:
            self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.serialize_dataset = False

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet: bool = False) -> None:
        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, parquet_file in enumerate(target_files):
            local_path = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
            target_files[idx] = local_path
        if use_origin_parquet:
            self.data_files = target_files

    def _read_files_and_tokenize(self) -> None:
        dataframes: list[datasets.Dataset] = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)
        logger.info("dataset len: %s", len(self.dataframe))

        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(len(self.dataframe), size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {len(self.dataframe)}")

        extract_fn = functools.partial(
            extract_prompt_fields,
            prompt_key=self.prompt_key,
            enable_think=self.enable_think,
            enable_nonthink=self.enable_nonthink,
        )
        try:
            self.dataframe = self.dataframe.map(
                extract_fn,
                num_proc=self.num_workers,
                desc="Extract prompts and reward annotations",
            )
        except TypeError as exc:
            if "cannot pickle" not in str(exc):
                raise
            logger.warning(
                "Falling back to single-process map due to pickle error: %s",
                exc,
            )
            self.dataframe = self.dataframe.map(
                extract_fn,
                num_proc=None,
                desc="Extract prompts and reward annotations",
            )
        logger.info("processed dataset len: %s", len(self.dataframe))
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _extract_prompt_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        return extract_prompt_fields(
            row,
            prompt_key=self.prompt_key,
            enable_think=self.enable_think,
            enable_nonthink=self.enable_nonthink,
        )

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        if not self.filter_overlong_prompts:
            return dataframe
        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key

        if processor is not None:
            def doc_length(doc: dict[str, Any]) -> int:
                messages = self._build_messages(dict(doc))
                raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = [process_image(image) for image in doc.get(image_key, [])]
                videos = [process_video(video) for video in doc.get(video_key, [])]
                encoded = processor(
                    text=[raw_prompt],
                    images=images or None,
                    videos=videos or None,
                    return_tensors="pt",
                )
                return int(encoded["input_ids"].shape[-1])

        else:

            def doc_length(doc: dict[str, Any]) -> int:
                messages = doc[prompt_key]
                return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))

        filter_fn = lambda doc: doc_length(doc) <= self.max_prompt_length - 10
        try:
            filtered = dataframe.filter(
                filter_fn,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length - 10} tokens",
            )
        except TypeError as exc:
            if "cannot pickle" not in str(exc):
                raise
            logger.warning(
                "Falling back to single-process filter due to pickle error: %s",
                exc,
            )
            filtered = dataframe.filter(
                filter_fn,
                num_proc=None,
                desc=f"Filtering prompts longer than {self.max_prompt_length - 10} tokens",
            )
        logger.info("filtered dataset len: %s", len(filtered))
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

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = example.pop(self.prompt_key)
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                segments = [segment for segment in re.split(r"(<image>|<video>)", content) if segment]
                parsed_segments = []
                for segment in segments:
                    if segment == "<image>":
                        parsed_segments.append({"type": "image"})
                    elif segment == "<video>":
                        parsed_segments.append({"type": "video"})
                    else:
                        parsed_segments.append({"type": "text", "text": segment})
                message["content"] = parsed_segments
        return messages

    def __getitem__(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = dict(self.dataframe[index])
        messages = self._build_messages(dict(row))
        model_inputs: dict[str, Any] = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            if self.use_force_prefix:
                raw_prompt = raw_prompt + self._FORCE_PREFIX_CONTENT
            multi_modal_data: dict[str, Any] = {}

            images = None
            if self.image_key in row and row.get(self.image_key):
                images = [process_image(image) for image in row.pop(self.image_key)]
                multi_modal_data["image"] = images
            videos = None
            if self.video_key in row and row.get(self.video_key):
                videos = [process_video(video) for video in row.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            row["multi_modal_data"] = multi_modal_data
            if self.return_multi_modal_inputs:
                mm_inputs = dict(model_inputs)
                mm_inputs.pop("second_per_grid_ts", None)
                row["multi_modal_inputs"] = mm_inputs
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            if self.use_force_prefix:
                raw_prompt = raw_prompt + self._FORCE_PREFIX_CONTENT
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

        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = self._truncate_ids(raw_prompt_ids)
        row["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row["raw_prompt"] = messages
        if self.return_full_prompt:
            row["full_prompts"] = raw_prompt

        extra_info = row.get("extra_info", {}) or {}
        row["index"] = extra_info.get("index", index)
        row["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})
        if "source" not in row and "data_source" not in row:
            row["data_source"] = "unknown"
            logger.warning("No source/data_source field found for index %s, set to 'unknown'", row["index"])
        if self.need_tools_kwargs and not row["tools_kwargs"]:
            logger.warning(
                "tools_kwargs is empty for index %s, data source: %s",
                row["index"],
                row.get("data_source", row.get("source", "unknown")),
            )
        return row

    def _truncate_ids(self, token_ids: list[int]) -> list[int]:
        if self.truncation == "left":
            return token_ids[-self.max_prompt_length :]
        if self.truncation == "right":
            return token_ids[: self.max_prompt_length]
        if self.truncation == "middle":
            left = self.max_prompt_length // 2
            right = self.max_prompt_length - left
            return token_ids[:left] + token_ids[-right:]
        if self.truncation == "error":
            raise RuntimeError(
                f"Prompt length {len(token_ids)} exceeds max_prompt_length={self.max_prompt_length}. "
                "Consider increasingmax_prompt_length or enabling truncation."
            )
        raise ValueError(f"Unsupported truncation mode: {self.truncation}")

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()


class OneRecTask:
    """OpenOneRec task-specific runtime preparation logic."""

    @staticmethod
    def _normalize_layer_wrap_value(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, set):
            normalized: list[str] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append(item)
                elif hasattr(item, "__name__"):
                    normalized.append(str(item.__name__))
                else:
                    normalized.append(str(item))
            return sorted(normalized)
        if isinstance(value, tuple):
            return list(value)
        if value is None:
            return None
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
            if fsdp_cfg is None:
                continue
            wrap_policy = fsdp_cfg.get("wrap_policy")
            if wrap_policy is None:
                continue
            normalized = self._normalize_layer_wrap_value(wrap_policy.get("transformer_layer_cls_to_wrap"))
            if normalized is not None:
                wrap_policy["transformer_layer_cls_to_wrap"] = normalized

    @staticmethod
    def get_reward_model_cfg(config):
        reward_root = config.get("reward")
        if reward_root is not None and reward_root.get("reward_model") is not None:
            return reward_root.reward_model
        legacy_cfg = config.get("reward_model")
        if legacy_cfg is not None:
            return legacy_cfg
        return None

    def prepare(self, config) -> dict[str, Any]:
        reward_model_cfg = self.get_reward_model_cfg(config)
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer, processor = build_hf_tokenizer_and_processor(
            local_path,
            trust_remote_code=trust_remote_code,
        )

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            ray_worker_group_cls = RayWorkerGroup
            one_rec_actor_rollout_ref_worker = getattr(
                import_module("verl_gr.recipes.openonerec.onerec_fsdp_workers"),
                "OneRecActorRolloutRefWorker",
            )
            # one_rec_async_actor_rollout_ref_worker = getattr(
            #     import_module("verl_gr.recipes.openonerec.onerec_fsdp_workers"),
            #     "OneRecAsyncActorRolloutRefWorker",
            # )
            async_actor_rollout_ref_worker = AsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.get("name") == "two_stage":
                # Agent loop in verl>=0.7 resolves rollout backend via RolloutReplicaRegistry.
                # Register two_stage as a vLLM-backed alias so replica lookup never fails.
                RolloutReplicaRegistry = getattr(import_module("verl.workers.rollout.replica"), "RolloutReplicaRegistry")
                vLLMReplica = getattr(
                    import_module("verl.workers.rollout.vllm_rollout.vllm_async_server"),
                    "vLLMReplica",
                )
                RolloutReplicaRegistry.register("two_stage", lambda: vLLMReplica)
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in {"auto", "enable"}:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker as CriticWorker
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
            critic_worker = CriticWorker
            if config.actor_rollout_ref.rollout.get("name") == "two_stage":
                # Force legacy two-stage worker path. In verl>=0.7, rollout config
                # dataclass may normalize mode to async for validation purposes,
                # but the OneRec two-stage implementation itself is sync-style.
                actor_rollout_cls = one_rec_actor_rollout_ref_worker
            else:
                actor_rollout_cls = (
                    async_actor_rollout_ref_worker
                    if config.actor_rollout_ref.rollout.mode == "async"
                    else one_rec_actor_rollout_ref_worker
                )
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker as MegatronActorRolloutRefWorker,
                AsyncActorRolloutRefWorker as MegatronAsyncActorRolloutRefWorker,
                CriticWorker as MegatronCriticWorker,
            )

            try:
                from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            except ModuleNotFoundError:
                NVMegatronRayWorkerGroup = RayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup
            actor_rollout_ref_worker = MegatronActorRolloutRefWorker
            async_actor_rollout_ref_worker = MegatronAsyncActorRolloutRefWorker
            critic_worker = MegatronCriticWorker
            actor_rollout_cls = (
                async_actor_rollout_ref_worker
                if config.actor_rollout_ref.rollout.mode == "async"
                else actor_rollout_ref_worker
            )
        else:
            raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")

        reward_model_worker = None
        if reward_model_cfg is not None and reward_model_cfg.get("enable", False):
            if reward_model_cfg.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker

                reward_model_worker = RewardModelWorker
            elif reward_model_cfg.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker as MegatronRewardModelWorker

                reward_model_worker = MegatronRewardModelWorker
            else:
                raise NotImplementedError(f"Unknown reward model strategy: {reward_model_cfg.strategy}")

        return {
            "tokenizer": tokenizer,
            "processor": processor,
            "actor_rollout_cls": actor_rollout_cls,
            "critic_worker": critic_worker,
            "reward_model_worker": reward_model_worker,
            "reward_model_cfg": reward_model_cfg,
            "ray_worker_group_cls": ray_worker_group_cls,
        }


SLOT_PATTERN = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def _extract_all_tuples(text: Any) -> list[tuple[str, str, str]]:
    if not isinstance(text, str):
        logger.warning("_extract_all_tuples received non-string input: %s", type(text))
        return []
    matches = SLOT_PATTERN.findall(text)
    return [tuple(match) for match in matches] if matches else []


def think_format_reward(prediction: str) -> float:
    if "<think>" not in prediction or "</think>" not in prediction:
        return 0.0
    start_idx = prediction.find("<think>") + len("<think>")
    end_idx = prediction.find("</think>")
    if end_idx < start_idx:
        return 0.0
    content = prediction[start_idx:end_idx]
    content_stripped = content.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    return 1.0 if len(content_stripped) > 10 else 0.0


def _strip_think_prefix_if_present(prediction: str) -> str:
    """Return content after </think> when present, else original prediction."""
    if "<think>" in prediction and "</think>" in prediction:
        think_end_idx = prediction.find("</think>") + len("</think>")
        return prediction[think_end_idx:]
    return prediction


def partial_hit_reward(prediction: str, ground_truth: str) -> float:
    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    total_reward = 0.0
    for pred_tuple in pred_tuples:
        max_score = 0.0
        for gt_tuple in gt_tuples:
            if pred_tuple == gt_tuple:
                max_score = max(max_score, 100.0)
            elif pred_tuple[:2] == gt_tuple[:2]:
                max_score = max(max_score, 10.0)
            elif pred_tuple[0] == gt_tuple[0]:
                max_score = max(max_score, 1.0)
        total_reward += max_score
    return total_reward / len(pred_tuples)


def hit_reward(prediction: str, ground_truth: str) -> float:
    prediction = _strip_think_prefix_if_present(prediction)
    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    pred_set = set(pred_tuples)
    gt_set = set(gt_tuples)
    return len(pred_set & gt_set) / len(pred_tuples)


def first_sid_hit_reward(prediction: str, ground_truth: str) -> float:
    prediction = _strip_think_prefix_if_present(prediction)
    pred_tuples = _extract_all_tuples(prediction)
    if not pred_tuples:
        return 0.0
    first_pred_tuple = pred_tuples[0]
    gt_tuples = _extract_all_tuples(ground_truth)
    if not gt_tuples:
        return 0.0
    gt_set = set(gt_tuples)
    return float(first_pred_tuple in gt_set)


def pass_rate(prediction: str, ground_truth: str) -> float:
    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0
    pred_set = set(pred_tuples)
    gt_set = set(gt_tuples)
    return float(len(pred_set & gt_set) > 0)


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],  # noqa: ARG001
) -> dict[str, float]:
    prediction = solution_str
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

