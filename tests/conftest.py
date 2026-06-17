"""Shared pytest fixtures and markers."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
