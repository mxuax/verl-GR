"""DDP engine and actor config dataclasses for verl-GR.

All classes live in verl-GR so that `strategy: ddp` works without
modifying any ``verl/`` source file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verl.workers.config.actor import ActorConfig
from verl.workers.config.engine import EngineConfig, QATEngineConfig
from verl.workers.config.optimizer import FSDPOptimizerConfig


@dataclass
class DDPEngineConfig(EngineConfig):
    """Configuration for PyTorch DistributedDataParallel (DDP) engine.

    Inherits directly from ``EngineConfig`` — no FSDP-specific fields.
    DDP keeps a full model replica on every GPU.
    """

    _mutable_fields = EngineConfig._mutable_fields | {"ddp_find_unused_parameters"}

    strategy: str = "ddp"
    ddp_find_unused_parameters: bool = False
    model_dtype: str = "fp32"
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    # Original MiniOneRec disables flash and memory-efficient SDP.
    disable_flash_sdp: bool = False
    # MiniOneRec: only compute LM logits for completion tokens (logits_to_keep).
    completion_only_logprob: bool = False
    # Force padded HF forward even when the engine input is nested/rmpad.
    completion_only_force_padded: bool = False
    # MiniOneRec original ReReTrainer does not pass position_ids to HF forward.
    completion_only_drop_position_ids: bool = False
    qat: QATEngineConfig = field(default_factory=QATEngineConfig)

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["ddp"], f"strategy {self.strategy} not supported"


@dataclass
class DDPActorConfig(ActorConfig):
    """Configuration for DDP actor models.

    DDP is suitable for small models where per-GPU memory is sufficient.
    """

    strategy: str = "ddp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    engine_config: DDPEngineConfig = field(default_factory=DDPEngineConfig)
    optim: FSDPOptimizerConfig = field(default_factory=FSDPOptimizerConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False
    # Ref-policy sync fields are consumed by MiniOneRecRayPPOTrainer.  They
    # live on the ref actor config in YAML, so the DDP dataclass must accept
    # them even though regular actor instances ignore them.
    sync_freq: int = 0
    ref_model_mixup_alpha: float = 0.6

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.engine_config
        # Keep engine strategy in sync with actor strategy, matching
        # upstream FSDPActorConfig's frozen-config update pattern.
        object.__setattr__(self.engine, "strategy", self.strategy)

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        super().validate(n_gpus, train_batch_size, model_config)
