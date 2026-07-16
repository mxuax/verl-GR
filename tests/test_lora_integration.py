"""Integration tests for LoRA in verl-GR (CPU/GPU as available)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from verl_gr.utils.lora_config import is_lora_enabled, normalize_lora_config
from verl_gr.workers.engine.ddp.transformer_impl import _SimpleCheckpointManager


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.proj = nn.Linear(16, 32)

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))


@pytest.fixture
def peft_available():
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    return get_peft_model, LoraConfig


def test_hydra_compose_minionerec_model_has_lora_defaults():
    source = Path("configs/verl_gr/model/minionerec_hf_model.yaml").read_text(encoding="utf-8")
    assert "lora_defaults" in source
    assert "lora_rank" in Path("configs/verl_gr/model/lora_defaults.yaml").read_text(encoding="utf-8")


def test_lora_override_applies_via_main_ppo():
    import subprocess
    import sys

    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "verl_gr.trainers.main_ppo",
            "--config-name",
            "minionerec/grpo_trainer_ddp",
            "++actor_rollout_ref.model.lora_rank=8",
            "--cfg",
            "job",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert "lora_rank: 8" in out


def test_normalize_lora_config_noop_when_disabled():
    config = OmegaConf.create({"actor_rollout_ref": {"model": {"lora_rank": 0}}})
    normalize_lora_config(config)
    assert config.actor_rollout_ref.model.lora_rank == 0


def test_simple_checkpoint_manager_exports_lora_adapter(peft_available, tmp_path):
    get_peft_model, LoraConfig = peft_available
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained("gpt2")
    model = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"),
    )
    ddp_model = DDP(model, device_ids=[0]) if torch.cuda.is_available() else model

    mgr = _SimpleCheckpointManager(
        model=ddp_model,
        optimizer=None,
        lr_scheduler=None,
        checkpoint_config=None,
        model_config_path="/tmp/base_model",
    )
    ckpt_dir = tmp_path / "ckpt"
    with patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.barrier"):
        mgr.save_checkpoint(local_path=str(ckpt_dir), global_step=1)

    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "lora_adapter" / "adapter_config.json").exists()
    assert (ckpt_dir / "lora_base_model.txt").read_text(encoding="utf-8").strip() == "/tmp/base_model"

    adapter_cfg = json.loads((ckpt_dir / "lora_adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_cfg["r"] == 4


def test_simple_checkpoint_manager_full_model_path_unchanged(tmp_path):
    model = _TinyLM()
    mgr = _SimpleCheckpointManager(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        checkpoint_config=None,
        model_config_path=None,
    )
    ckpt_dir = tmp_path / "ckpt_full"
    with patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.barrier"):
        mgr.save_checkpoint(local_path=str(ckpt_dir), global_step=1)

    assert (ckpt_dir / "model.pt").exists()
    assert not (ckpt_dir / "lora_adapter").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ddp_lora_optimizer_only_trainable_params(peft_available):
    get_peft_model, LoraConfig = peft_available
    from verl.workers.config import FSDPOptimizerConfig
    from verl_gr.workers.engine.ddp.transformer_impl import DDPEngine

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=0, world_size=1)

    from verl.workers.config.model import HFModelConfig
    from verl_gr.workers.config.ddp_engine import DDPEngineConfig

    model_cfg = HFModelConfig(
        path="gpt2",
        lora_rank=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        trust_remote_code=False,
    )
    engine_cfg = DDPEngineConfig(strategy="ddp", forward_only=False, model_dtype="bf16", use_torch_compile=False)
    optim_cfg = FSDPOptimizerConfig(lr=1e-4, clip_grad=1.0)

    engine = DDPEngine(model_cfg, engine_cfg, optim_cfg, checkpoint_config=None)
    assert engine._is_lora is True
    assert is_lora_enabled(model_cfg) is True
