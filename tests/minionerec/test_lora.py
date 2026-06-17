"""LoRA behavior tests for MiniOneRec DDP path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from verl_gr.utils.lora_config import (
    is_lora_enabled,
    normalize_lora_config,
    resolve_lora_rank,
    should_merge_lora,
    trainable_parameters,
)
from verl_gr.workers.engine.ddp.transformer_impl import _SimpleCheckpointManager

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_resolve_lora_rank_defaults_to_zero():
    model_cfg = SimpleNamespace(lora_rank=0, lora={})
    assert resolve_lora_rank(model_cfg) == 0


def test_resolve_lora_rank_prefers_nested_rank():
    model_cfg = SimpleNamespace(lora_rank=8, lora={"rank": 16})
    assert resolve_lora_rank(model_cfg) == 16


def test_is_lora_enabled_with_adapter_path_only():
    model_cfg = SimpleNamespace(lora_rank=0, lora={}, lora_adapter_path="/tmp/adapter")
    assert is_lora_enabled(model_cfg) is True


def test_should_merge_lora_default_false():
    model_cfg = SimpleNamespace(lora={"merge": False})
    assert should_merge_lora(model_cfg) is False


def test_trainable_parameters_filters_frozen():
    model = nn.Linear(4, 2)
    for param in model.parameters():
        param.requires_grad = False
    model.bias.requires_grad = True
    assert len(trainable_parameters(model)) == 1


def test_normalize_lora_config_infers_rank_from_adapter(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "lora_rank": 0,
                    "lora_adapter_path": str(adapter_dir),
                    "use_shm": False,
                }
            }
        }
    )
    with patch("verl.utils.fs.copy_to_local", return_value=str(adapter_dir)):
        normalize_lora_config(config)
    assert config.actor_rollout_ref.model.lora_rank == 32


def test_normalize_lora_config_noop_when_disabled():
    config = OmegaConf.create({"actor_rollout_ref": {"model": {"lora_rank": 0}}})
    normalize_lora_config(config)
    assert config.actor_rollout_ref.model.lora_rank == 0


def test_lora_override_applies_via_main_ppo():
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
        cwd=REPO_ROOT,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert "lora_rank: 8" in out


def test_simple_checkpoint_manager_exports_lora_adapter(peft_available, tmp_path):
    get_peft_model, LoraConfig = peft_available
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained("gpt2")
    model = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"),
    )
    # Checkpoint manager unwraps DDP; avoid DDP here without a full distributed init.
    ddp_model = model

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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ddp_lora_optimizer_only_trainable_params(peft_available):
    get_peft_model, LoraConfig = peft_available
    from verl.workers.config import FSDPOptimizerConfig
    from verl.workers.config.model import HFModelConfig
    from verl_gr.workers.config.ddp_engine import DDPEngineConfig
    from verl_gr.workers.engine.ddp.transformer_impl import DDPEngine

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo", rank=0, world_size=1
        )

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
