from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from verl_gr.utils.lora_config import (
    is_lora_enabled,
    normalize_lora_config,
    resolve_lora_rank,
    should_merge_lora,
    trainable_parameters,
)


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
    import torch
    import torch.nn as nn

    model = nn.Linear(4, 2)
    for param in model.parameters():
        param.requires_grad = False
    model.bias.requires_grad = True
    assert len(trainable_parameters(model)) == 1


def test_normalize_lora_config_infers_rank_from_adapter(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")

    from omegaconf import OmegaConf

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


def test_lora_env_script_only_adds_overrides_when_enabled():
    source = Path("scripts/lora_env.sh").read_text(encoding="utf-8")
    assert "LORA_RANK" in source
    assert "LORA_OVERRIDES" in source
    assert "++actor_rollout_ref.model.lora_rank" in source


def test_ddp_engine_exposes_disable_adapter():
    source = Path("verl_gr/workers/engine/ddp/transformer_impl.py").read_text(encoding="utf-8")
    assert "is_lora_enabled" in source
    assert "trainable_parameters" in source
    assert "lora_adapter" in source
    assert "def disable_adapter" in source


def test_task_runtime_configures_lora_for_ddp():
    source = Path("verl_gr/recipes/task_runtime.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    assert "RecipeTaskRuntime" in class_names
    assert "configure_lora" in source
    assert "ddp_find_unused_parameters" in source


def test_openonerec_config_keeps_lora_disabled_by_default():
    source = Path("configs/verl_gr/openonerec/grpo_trainer.yaml").read_text(encoding="utf-8")
    assert "lora_rank: 0" in source


def test_minionerec_hf_model_composes_lora_defaults():
    source = Path("configs/verl_gr/model/minionerec_hf_model.yaml").read_text(encoding="utf-8")
    assert "lora_defaults" in source
