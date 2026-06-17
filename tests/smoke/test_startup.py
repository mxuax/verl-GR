"""Smoke tests: Hydra compose and optional training startup for each recipe."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

RECIPE_CONFIGS = [
    ("minionerec", "minionerec/grpo_trainer_ddp"),
    ("openonerec", "openonerec/grpo_trainer"),
    ("rankgrpo", "rankgrpo/rankgrpo_trainer"),
]


@pytest.mark.parametrize("recipe,config_name", RECIPE_CONFIGS)
def test_hydra_compose(recipe, config_name):
    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "verl_gr.trainers.main_ppo",
            "--config-name",
            config_name,
            "--cfg",
            "job",
        ],
        cwd=REPO_ROOT,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert recipe in out


@pytest.mark.gpu
@pytest.mark.skipif(not os.environ.get("VERL_GR_SMOKE_TRAIN"), reason="Set VERL_GR_SMOKE_TRAIN=1 with local model/data")
@pytest.mark.parametrize("recipe,config_name", RECIPE_CONFIGS)
def test_training_startup_one_step(recipe, config_name, tmp_path):
    """Optional: full Ray training startup (requires model path + parquet data)."""
    out_dir = tmp_path / recipe
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "verl_gr.trainers.main_ppo",
            "--config-name",
            config_name,
            "trainer.total_training_steps=1",
            "trainer.val_before_train=false",
            "trainer.n_gpus_per_node=1",
            "trainer.nnodes=1",
            f"trainer.default_local_dir={out_dir}",
            "trainer.logger=[]",
            "data.train_batch_size=1",
            "data.val_batch_size=1",
        ],
        cwd=REPO_ROOT,
        timeout=600,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
