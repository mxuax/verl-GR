"""DDP engine and actor config dataclasses for verl-GR.

All classes live in verl-GR so that `strategy: ddp` works without
modifying any ``verl/`` source file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verl.workers.config.actor import ActorConfig
from verl.workers.config.engine import EngineConfig, QATEngineConfig


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
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.engine_config
        self.engine.strategy = self.strategy

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        super().validate(n_gpus, train_batch_size, model_config)
