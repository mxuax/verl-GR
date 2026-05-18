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

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name
from verl.workers.config import HFModelConfig, FSDPOptimizerConfig
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead

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

    def save_checkpoint(self, *, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **_kwargs):
        os.makedirs(local_path, exist_ok=True)
        rank = torch.distributed.get_rank()
        # Unwrap DDP to get the raw model
        model = self.model.module if isinstance(self.model, DDP) else self.model
        if rank == 0:
            # Raw state_dict for resuming training
            torch.save(model.state_dict(), os.path.join(local_path, "model.pt"))
            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), os.path.join(local_path, "optimizer.pt"))
            if self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(local_path, "lr_scheduler.pt"))
            torch.save({"global_step": global_step}, os.path.join(local_path, "extra.pt"))
            # Also export HF format for evaluation (eval_compare_ckpts.py etc.)
            self._export_hf(model, local_path)
        torch.distributed.barrier()

    def _export_hf(self, model, local_path):
        """Export the raw HF model as a HuggingFace checkpoint.

        Writes ``huggingface/`` subdirectory with ``config.json``, weights,
        and tokenizer files so that ``AutoModelForCausalLM.from_pretrained``
        and downstream eval scripts can load the checkpoint directly.
        """
        hf_dir = os.path.join(local_path, "huggingface")
        os.makedirs(hf_dir, exist_ok=True)
        try:
            model.save_pretrained(hf_dir)
            logger.info("Exported HF checkpoint to %s", hf_dir)
        except Exception:
            logger.warning("Failed to save HF checkpoint to %s", hf_dir, exc_info=True)
        # Copy tokenizer files from the base model if available
        base_model = self._model_config_path
        if base_model and os.path.isdir(base_model):
            import shutil
            for fname in os.listdir(base_model):
                src = os.path.join(base_model, fname)
                if not os.path.isfile(src):
                    continue
                if fname.startswith("tokenizer") or fname == "vocab.json" or fname == "merges.txt" or fname == "special_tokens_map.json" or fname == "added_tokens.json":
                    shutil.copy2(src, hf_dir)
                    logger.info("Copied %s to HF checkpoint", fname)

    def load_checkpoint(self, *, local_path, hdfs_path=None, del_local_after_load=True, **_kwargs):
        model = self.model.module if isinstance(self.model, DDP) else self.model
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

        # DDP has no parameter/optimizer offloading
        self._is_offload_param = False
        self._is_offload_optimizer = False
        self._is_lora = self.model_config.lora_rank > 0

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
        from verl.utils.activation_offload import enable_activation_offloading

        module = module.to(get_device_id())

        find_unused = getattr(self.engine_config, "ddp_find_unused_parameters", False)
        module = DDP(module, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused)

        # Apply activation offloading (if configured)
        if getattr(self.model_config, "enable_activation_offload", False):
            enable_activation_offloading(module, "ddp")

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

    def optimizer_step(self):
        assert self.optimizer_config.clip_grad is not None

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.module.parameters(), max_norm=self.optimizer_config.clip_grad
        )

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        if self._qat_enabled:
            from verl.utils.qat.core import invalidate_all_scales
            invalidate_all_scales(self.module)

        return grad_norm.item() if hasattr(grad_norm, "item") else grad_norm

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

    MRO: DDPEngineWithLMHead → FSDPEngineWithLMHead → DDPEngine →
    FSDPEngine → BaseEngine.

    - ``__init__`` and DDP-specific overrides come from ``DDPEngine``.
    - ``prepare_model_inputs`` / ``prepare_model_outputs`` /
      ``forward_step`` come from ``FSDPEngineWithLMHead`` — zero copy.
    """


EngineRegistry.register(
    model_type="language_model", backend=["ddp"], device=["cuda"]
)(DDPEngineWithLMHead)
