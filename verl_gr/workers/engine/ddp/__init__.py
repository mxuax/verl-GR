"""DDP engine for verl-GR — PyTorch DistributedDataParallel backend.

Import this module to register the ``"ddp"`` backend with verl's
``EngineRegistry``.  The engine is a thin override of the FSDP engine
that replaces FSDP wrapping with ``DistributedDataParallel`` and uses
simple (full-state-dict) checkpoint save/load.
"""

from verl_gr.workers.engine.ddp.transformer_impl import DDPEngine, DDPEngineWithLMHead  # noqa: F401
