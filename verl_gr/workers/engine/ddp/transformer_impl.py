"""DDP engine for verl-GR — thin override of the FSDP engine.

``DDPEngine`` inherits from ``FSDPEngine`` and replaces only the
FSDP-specific parts: model wrapping (DDP instead of FSDP), checkpoint
save/load, gradient clipping, and parameter offload semantics.

``DDPEngineWithLMHead`` mixes in the LM-head forward logic from
``FSDPEngineWithLMHead`` — zero code duplication.
"""

from __future__ import annotations

import logging
import os
import json
from contextlib import nullcontext

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name
from verl.workers.config import HFModelConfig, FSDPOptimizerConfig
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead
from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches
from verl_gr.workers.optimizer import build_actor_optimizer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


# ---------------------------------------------------------------------------
# Simple checkpoint manager — DDP keeps full state dicts, no sharding.
# ---------------------------------------------------------------------------


class _SimpleCheckpointManager:
    """Minimal checkpoint manager for DDP (full state dicts, no sharding)."""

    def __init__(self, model, optimizer, lr_scheduler, checkpoint_config, model_config_path=None, **_kwargs):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.checkpoint_config = checkpoint_config
        self._model_config_path = model_config_path  # base HF model path for config export

    @staticmethod
    def _unwrap_model(model):
        from torch.nn.parallel import DistributedDataParallel as DDP

        return model.module if isinstance(model, DDP) else model

    @staticmethod
    def _is_peft_model(model) -> bool:
        return hasattr(model, "peft_config")

    def save_checkpoint(self, *, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **_kwargs):
        os.makedirs(local_path, exist_ok=True)
        rank = torch.distributed.get_rank()
        model = self._unwrap_model(self.model)
        if rank == 0:
            torch.save(model.state_dict(), os.path.join(local_path, "model.pt"))
            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), os.path.join(local_path, "optimizer.pt"))
            if self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(local_path, "lr_scheduler.pt"))
            torch.save({"global_step": global_step}, os.path.join(local_path, "extra.pt"))
            self._export_hf(model, local_path)
        torch.distributed.barrier()

    def _export_hf(self, model, local_path):
        """Export HuggingFace-compatible artifacts for eval / inference."""
        hf_dir = os.path.join(local_path, "huggingface")
        os.makedirs(hf_dir, exist_ok=True)
        base_model = self._model_config_path

        if self._is_peft_model(model):
            adapter_dir = os.path.join(local_path, "lora_adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            try:
                model.save_pretrained(adapter_dir)
                model.save_pretrained(hf_dir)
                logger.info("Exported LoRA adapter to %s", adapter_dir)
            except Exception:
                logger.warning("Failed to save LoRA adapter to %s", adapter_dir, exc_info=True)
            if base_model:
                with open(os.path.join(local_path, "lora_base_model.txt"), "w", encoding="utf-8") as handle:
                    handle.write(str(base_model).strip() + "\n")
            self._copy_tokenizer_files(base_model, hf_dir)
            return

        try:
            model.save_pretrained(hf_dir)
            logger.info("Exported HF checkpoint to %s", hf_dir)
        except Exception:
            logger.warning("Failed to save HF checkpoint to %s", hf_dir, exc_info=True)
        self._copy_tokenizer_files(base_model, hf_dir)

    @staticmethod
    def _copy_tokenizer_files(base_model, hf_dir):
        if not base_model or not os.path.isdir(base_model):
            return
        import shutil

        for fname in os.listdir(base_model):
            src = os.path.join(base_model, fname)
            if not os.path.isfile(src):
                continue
            if (
                fname.startswith("tokenizer")
                or fname in {"vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"}
            ):
                shutil.copy2(src, hf_dir)
                logger.info("Copied %s to HF checkpoint", fname)

    def load_checkpoint(self, *, local_path, hdfs_path=None, del_local_after_load=True, **_kwargs):
        model = self._unwrap_model(self.model)
        state_dict = torch.load(os.path.join(local_path, "model.pt"), map_location=device_name, weights_only=False)
        model.load_state_dict(state_dict)
        if self.optimizer is not None:
            opt_path = os.path.join(local_path, "optimizer.pt")
            if os.path.exists(opt_path):
                self.optimizer.load_state_dict(
                    torch.load(opt_path, map_location=device_name, weights_only=False)
                )
        if self.lr_scheduler is not None:
            sched_path = os.path.join(local_path, "lr_scheduler.pt")
            if os.path.exists(sched_path):
                self.lr_scheduler.load_state_dict(
                    torch.load(sched_path, map_location=device_name, weights_only=False)
                )
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# DDP engine
# ---------------------------------------------------------------------------


class DDPEngine(FSDPEngine):
    """Training engine using PyTorch DistributedDataParallel.

    Inherits all framework-agnostic logic from ``FSDPEngine`` (model
    loading, optimizer/scheduler construction, forward/backward loop,
    mini-batch iteration, entropy computation).

    Overrides only the FSDP-coupled methods.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config,  # DDPEngineConfig (from verl_gr)
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        # Skip FSDPEngine.__init__ — call BaseEngine directly to avoid
        # FSDP device-mesh creation.
        BaseEngine.__init__(self)

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = (
            optimizer_config
            if isinstance(optimizer_config, FSDPOptimizerConfig)
            else omega_conf_to_dataclass(optimizer_config, dataclass_type=FSDPOptimizerConfig)
        )
        self.checkpoint_config = checkpoint_config

        self.mode = None
        self.rank = torch.distributed.get_rank()

        self.use_remove_padding = self.model_config.use_remove_padding

        if getattr(self.engine_config, "full_determinism", False):
            from verl.workers.engine.utils import enable_full_determinism
            enable_full_determinism(seed=getattr(self.engine_config, "seed", 42))

        if getattr(self.engine_config, "disable_flash_sdp", False) and torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        from verl_gr.utils.lora_config import is_lora_enabled, trainable_parameters

        # DDP has no parameter/optimizer offloading
        self._is_offload_param = False
        self._is_offload_optimizer = False
        self._is_lora = is_lora_enabled(self.model_config)

        # QAT
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            logger.info(f"QAT enabled: mode={self._qat_config.mode}")

        # Ulysses sequence parallelism disabled for DDP
        self.device_mesh = None
        self.use_ulysses_sp = False
        self.ulysses_sequence_parallel_size = 1
        self.ulysses_device_mesh = None
        self.ulysses_parallel_group = None

        # Entropy
        use_chunked = getattr(self.engine_config, "entropy_from_logits_with_chunking", False)
        entropy_fn = verl_F.entropy_from_logits_with_chunking if use_chunked else verl_F.entropy_from_logits
        use_compile = getattr(self.engine_config, "use_torch_compile", True)
        self.compute_entropy_from_logits = torch.compile(entropy_fn, dynamic=True) if use_compile else entropy_fn

    # ------------------------------------------------------------------
    # Overrides — properties
    # ------------------------------------------------------------------

    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    def is_mp_src_rank_with_outputs(self):
        return True  # no sequence parallelism

    # ------------------------------------------------------------------
    # Overrides — model wrapping
    # ------------------------------------------------------------------

    def _build_ddp_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Wrap the HF model with DistributedDataParallel."""
        module = module.to(get_device_id())

        find_unused = getattr(self.engine_config, "ddp_find_unused_parameters", False)
        module = DDP(module, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused)

        # verl activation offload only supports FSDP/FSDP2. Refuse silently here
        # so a stale `enable_activation_offload=true` does not crash DDP init.
        if getattr(self.model_config, "enable_activation_offload", False):
            print(
                "[DDPEngine] enable_activation_offload=true is unsupported on DDP; "
                "skipping activation offload (set model.enable_activation_offload=false).",
                flush=True,
            )

        return module

    def _build_model_optimizer(self):
        """Same as FSDP but uses DDP wrapping."""
        from verl.utils.model import print_model_size

        module = self._build_module()
        if self._is_lora:
            module = self._build_lora_module(module)

        if self._qat_enabled and not getattr(self.engine_config, "forward_only", False):
            module = self._apply_qat(module)

        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        from verl.utils.debug import log_gpu_memory_usage
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        log_gpu_memory_usage("Before DDP", logger=None)
        module = self._build_ddp_module(module)
        log_gpu_memory_usage("After DDP", logger=None)

        if not getattr(self.engine_config, "forward_only", False):
            optimizer = self._build_optimizer(module)
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    # ------------------------------------------------------------------
    # Overrides — training / eval contexts
    # ------------------------------------------------------------------

    def train_mode(self, **kwargs):
        return DDPEngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        return DDPEngineEvalModeCtx(self, **kwargs)

    # ------------------------------------------------------------------
    # Overrides — data parallel
    # ------------------------------------------------------------------

    def get_data_parallel_rank(self):
        return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size()

    def get_data_parallel_group(self):
        return torch.distributed.group.WORLD

    # ------------------------------------------------------------------
    # Overrides — optimizer
    # ------------------------------------------------------------------

    def _build_optimizer(self, module):
        from verl_gr.utils.lora_config import trainable_parameters

        params = trainable_parameters(module) if self._is_lora else module.parameters()
        return build_actor_optimizer(params, self.optimizer_config)

    def _debug_enabled(self) -> bool:
        return bool(os.getenv("MINIONEREC_DEBUG_DUMP_DIR"))

    def _debug_path(self) -> str:
        dump_dir = os.getenv("MINIONEREC_DEBUG_DUMP_DIR", "")
        os.makedirs(dump_dir, exist_ok=True)
        return os.path.join(dump_dir, f"verl_ddp_rank{self.rank}.jsonl")

    def _debug_write(self, payload: dict):
        if not self._debug_enabled():
            return
        payload = {"rank": int(self.rank), **payload}
        with open(self._debug_path(), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _debug_tensor_summary(value) -> dict | None:
        if not isinstance(value, torch.Tensor):
            return None
        tensor = value.detach()
        summary = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        if tensor.numel() > 0 and tensor.is_floating_point():
            tf = tensor.float()
            summary.update(
                mean=float(tf.mean().item()),
                std=float(tf.std(unbiased=False).item()) if tf.numel() > 1 else 0.0,
                min=float(tf.min().item()),
                max=float(tf.max().item()),
            )
        elif tensor.numel() > 0:
            summary.update(
                min=int(tensor.min().item()),
                max=int(tensor.max().item()),
                checksum=int(tensor.long().sum().item()),
            )
        return summary

    def _debug_grad_norm(self) -> float:
        total = torch.zeros((), device=get_device_name())
        for param in self.module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total = total + grad.norm(2).pow(2)
        return float(total.sqrt().item())

    def _debug_selected_params(self) -> list[tuple[str, torch.nn.Parameter]]:
        substrings = os.getenv(
            "MINIONEREC_DEBUG_PARAM_SUBSTR",
            "layers.0.self_attn.q_proj.weight,layers.27.mlp.down_proj.weight",
        ).split(",")
        substrings = [item.strip() for item in substrings if item.strip()]
        selected = []
        for name, param in self.module.named_parameters():
            if any(substr in name for substr in substrings):
                selected.append((name, param))
        return selected

    def _debug_param_snapshot(self) -> dict[str, dict]:
        snapshot = {}
        for name, param in self._debug_selected_params():
            data = param.detach()
            entry = {
                "dtype": str(data.dtype),
                "shape": list(data.shape),
                "norm": float(data.float().norm().item()),
                "mean": float(data.float().mean().item()),
            }
            if param.grad is not None:
                entry["grad_dtype"] = str(param.grad.dtype)
                entry["grad_norm"] = float(param.grad.detach().float().norm().item())
            snapshot[name] = entry
        return snapshot

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        if forward_only or not self._debug_enabled():
            return super().forward_backward_batch(data, loss_function, forward_only=forward_only)

        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )
        self._debug_write(
            {
                "event": "forward_backward_start",
                "num_micro_batches": len(micro_batches),
                "batch_shape": list(data.shape),
                "batch_num_tokens": float(batch_num_tokens.item()),
                "dp_size": int(self.get_data_parallel_size()),
                "global_batch_size": tu.get(data, "global_batch_size", default=None),
                "batch_num_tokens_meta": tu.get(data, "batch_num_tokens", default=None),
            }
        )

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()
        for micro_idx, micro_batch in enumerate(micro_batches):
            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                loss_value = float(loss.detach().float().item()) if isinstance(loss, torch.Tensor) else None
                metrics = {}
                if isinstance(meta_info, dict):
                    metrics = meta_info.get("metrics", {})
                self._debug_write(
                    {
                        "event": "micro_forward",
                        "micro_idx": micro_idx,
                        "micro_shape": list(micro_batch.shape),
                        "loss": loss_value,
                        "loss_mask": self._debug_tensor_summary(micro_batch.get("loss_mask", None)),
                        "advantages": self._debug_tensor_summary(micro_batch.get("advantages", None)),
                        "responses": self._debug_tensor_summary(micro_batch.get("responses", None)),
                        "global_batch_size": tu.get(micro_batch, "global_batch_size", default=None),
                        "batch_num_tokens": tu.get(micro_batch, "batch_num_tokens", default=None),
                        "update_lr_scheduler": tu.get(micro_batch, "update_lr_scheduler", default=None),
                        "metric_keys": sorted(list(metrics.keys()))[:32] if isinstance(metrics, dict) else [],
                    }
                )
                if not forward_only:
                    loss.backward()
                    self._debug_write(
                        {
                            "event": "micro_backward",
                            "micro_idx": micro_idx,
                            "grad_norm_after_backward": self._debug_grad_norm(),
                            "params": self._debug_param_snapshot(),
                        }
                    )
            output_lst.append(meta_info)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def optimizer_step(self):
        assert self.optimizer_config.clip_grad is not None

        debug_enabled = self._debug_enabled()
        before_params = {}
        before_master_params = {}
        if debug_enabled:
            for name, param in self._debug_selected_params():
                before_params[name] = param.detach().float().clone()
                master_param_for_visible = getattr(self.optimizer, "master_param_for_visible", None)
                if master_param_for_visible is not None:
                    master = master_param_for_visible(param)
                    if master is not None:
                        before_master_params[name] = master.detach().float().clone()
            self._debug_write(
                {
                    "event": "optimizer_pre_clip",
                    "optimizer_class": type(self.optimizer).__name__,
                    "optimizer_module": type(self.optimizer).__module__,
                    "inner_optimizer_class": type(getattr(self.optimizer, "inner_optimizer", self.optimizer)).__name__,
                    "inner_optimizer_module": type(getattr(self.optimizer, "inner_optimizer", self.optimizer)).__module__,
                    "has_fp32_master": hasattr(self.optimizer, "master_params"),
                    "lr": float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else None,
                    "param_group_lrs": [float(group["lr"]) for group in self.optimizer.param_groups],
                    "clip_grad": float(self.optimizer_config.clip_grad),
                    "grad_norm_raw": self._debug_grad_norm(),
                    "params": self._debug_param_snapshot(),
                }
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.module.parameters(), max_norm=self.optimizer_config.clip_grad
        )

        if debug_enabled:
            self._debug_write(
                {
                    "event": "optimizer_post_clip",
                    "grad_norm_returned_preclip": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
                    "grad_norm_after_clip": self._debug_grad_norm(),
                    "params": self._debug_param_snapshot(),
                }
            )

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        if debug_enabled:
            deltas = {}
            master_deltas = {}
            for name, param in self._debug_selected_params():
                if name not in before_params:
                    continue
                delta = param.detach().float() - before_params[name]
                deltas[name] = {
                    "max_abs": float(delta.abs().max().item()),
                    "mean_abs": float(delta.abs().mean().item()),
                    "norm": float(delta.norm().item()),
                    "param_norm_after": float(param.detach().float().norm().item()),
                    "param_dtype_after": str(param.dtype),
                }
                master_param_for_visible = getattr(self.optimizer, "master_param_for_visible", None)
                if master_param_for_visible is not None and name in before_master_params:
                    master = master_param_for_visible(param)
                    if master is not None:
                        master_delta = master.detach().float() - before_master_params[name]
                        master_deltas[name] = {
                            "max_abs": float(master_delta.abs().max().item()),
                            "mean_abs": float(master_delta.abs().mean().item()),
                            "norm": float(master_delta.norm().item()),
                            "param_norm_after": float(master.detach().float().norm().item()),
                            "param_dtype_after": str(master.dtype),
                        }
            opt_state = {}
            for name, param in self._debug_selected_params()[:1]:
                state_param = param
                master_param_for_visible = getattr(self.optimizer, "master_param_for_visible", None)
                if master_param_for_visible is not None:
                    master_state_param = master_param_for_visible(param)
                    state_param = master_state_param if master_state_param is not None else param
                state = self.optimizer.state.get(state_param, {})
                opt_state[name] = {
                    key: str(value.dtype) if isinstance(value, torch.Tensor) else type(value).__name__
                    for key, value in state.items()
                }
            self._debug_write(
                {
                    "event": "optimizer_post_step",
                    "optimizer_class": type(self.optimizer).__name__,
                    "optimizer_module": type(self.optimizer).__module__,
                    "inner_optimizer_class": type(getattr(self.optimizer, "inner_optimizer", self.optimizer)).__name__,
                    "inner_optimizer_module": type(getattr(self.optimizer, "inner_optimizer", self.optimizer)).__module__,
                    "has_fp32_master": hasattr(self.optimizer, "master_params"),
                    "grad_norm_returned_preclip": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
                    "lr": float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else None,
                    "param_group_lrs": [float(group["lr"]) for group in self.optimizer.param_groups],
                    "deltas": deltas,
                    "fp32_master_deltas": master_deltas,
                    "optimizer_state": opt_state,
                }
            )

        if self._qat_enabled:
            from verl.utils.qat.core import invalidate_all_scales
            invalidate_all_scales(self.module)

        return grad_norm.item() if hasattr(grad_norm, "item") else grad_norm

    def lr_scheduler_step(self):
        """Advance scheduler and optionally dump LR before/after the step."""

        debug_enabled = self._debug_enabled()
        before_lrs = [float(group["lr"]) for group in self.optimizer.param_groups] if self.optimizer else []
        before_scheduler_lrs = []
        if self.lr_scheduler is not None:
            try:
                before_scheduler_lrs = [float(value) for value in self.lr_scheduler.get_last_lr()]
            except Exception:
                before_scheduler_lrs = []

        lr = super().lr_scheduler_step()

        if debug_enabled:
            after_lrs = [float(group["lr"]) for group in self.optimizer.param_groups] if self.optimizer else []
            after_scheduler_lrs = []
            if self.lr_scheduler is not None:
                try:
                    after_scheduler_lrs = [float(value) for value in self.lr_scheduler.get_last_lr()]
                except Exception:
                    after_scheduler_lrs = []
            self._debug_write(
                {
                    "event": "lr_scheduler_step",
                    "before_param_group_lrs": before_lrs,
                    "before_scheduler_lrs": before_scheduler_lrs,
                    "returned_lr": float(lr) if lr is not None else None,
                    "after_param_group_lrs": after_lrs,
                    "after_scheduler_lrs": after_scheduler_lrs,
                }
            )
        return lr

    # ------------------------------------------------------------------
    # Overrides — checkpoint
    # ------------------------------------------------------------------

    def initialize(self):
        self._build_model_optimizer()
        self.checkpoint_manager = _SimpleCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            checkpoint_config=self.checkpoint_config,
            model_config_path=self.model_config.local_path or self.model_config.path,
        )
        from verl.utils.debug import log_gpu_memory_usage
        log_gpu_memory_usage("After DDP initialize", logger=logger)

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path,
            global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep,
        )
        torch.distributed.barrier()

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **kwargs):
        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        torch.distributed.barrier()

    # ------------------------------------------------------------------
    # Overrides — parameter export
    # ------------------------------------------------------------------

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        from verl.utils.fsdp_utils import convert_weight_keys

        module = self.module.module if isinstance(self.module, DDP) else self.module

        peft_config = None
        merge_lora = self.model_config.lora.get("merge", False)

        if hasattr(module, "peft_config") and not merge_lora:
            from verl.utils.fsdp_utils import collect_lora_params, replace_lora_wrapper
            peft_config = module.peft_config.get("default", None)
            params = collect_lora_params(module=self.module, layered_summon=layered_summon, base_sync_done=base_sync_done)
            if not base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = module.state_dict()

        params = convert_weight_keys(params, module)
        return params.items(), peft_config

    def disable_adapter(self):
        """Delegate to the inner PEFT module (DDP wraps the PeftModel)."""
        module = self.module.module if isinstance(self.module, DDP) else self.module
        return module.disable_adapter()

    # ------------------------------------------------------------------
    # Overrides — device movement (simpler than FSDP)
    # ------------------------------------------------------------------

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Move DDP model / optimizer to CPU or GPU.  No FSDP offload APIs."""
        from verl.utils.debug import log_gpu_memory_usage
        BaseEngine.to(self, device=device, model=model, optimizer=optimizer, grad=grad)

        assert device in (get_device_name(), "cpu")
        if device == get_device_name():
            if model:
                self.module.to(device)
            if optimizer and self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
        elif device == "cpu":
            if model:
                self.module.to("cpu")
            if optimizer and self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to("cpu")
        else:
            raise ValueError(f"Invalid device type: {device}")

        log_gpu_memory_usage("After DDP to()", logger=logger)


class DDPEngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: DDPEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        super().__enter__()
        self.engine.module.eval()


class DDPEngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: DDPEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        super().__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


# ---------------------------------------------------------------------------
# DDP engine with LM head — reuse FSDPEngineWithLMHead forward logic
# ---------------------------------------------------------------------------


class DDPEngineWithLMHead(FSDPEngineWithLMHead, DDPEngine):
    """DDP engine with language-model head forward logic.

    ``CompletionOnlyLogprobMixin`` is applied once on ``FSDPEngineWithLMHead`` via
    ``apply_minionerec_engine_patches()`` (see ``ddp/__init__.py``). Do not
    mix it in again here — that breaks MRO after the global patch.

    MRO: DDPEngineWithLMHead → MiniOneRecFSDPEngineWithLMHead → DDPEngine → …
    """


EngineRegistry.register(
    model_type="language_model", backend=["ddp"], device=["cuda"]
)(DDPEngineWithLMHead)
