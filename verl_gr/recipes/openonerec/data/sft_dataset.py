"""OpenOneRec SFT dataset.

Wraps verl's ``MultiTurnSFTDataset`` after parsing the JSON-string
``messages`` column produced by ``product_rec.py``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

logger = logging.getLogger(__name__)


class OneRecSFTDataset(Dataset):
    """Thin wrapper that parses ``messages`` from JSON strings before SFT.

    ``product_rec.py`` serialises messages with ``json.dumps``, producing a
    string column that ``MultiTurnSFTDataset`` cannot consume directly.
    This class parses the column into Python lists, writes a cleaned
    parquet to a temporary location, and delegates everything else to
    ``MultiTurnSFTDataset``.
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: ProcessorMixin | None = None,
        max_samples: int = -1,
    ) -> None:
        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]

        self._config = config
        self._parsed_parquet_path: str | None = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None

        parsed_path = self._prepare_parsed_parquet(parquet_files)

        self._inner = MultiTurnSFTDataset(
            parquet_files=parsed_path,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_parsed_parquet(self, paths: list[str]) -> str:
        frames = [pd.read_parquet(p) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        logger.info("OneRec SFT — rows before parse: %d", len(df))

        messages_key = self._config.get("messages_key", "messages")

        if messages_key in df.columns:
            df[messages_key] = df[messages_key].apply(_parse_messages_column)

        self._tmpdir = tempfile.TemporaryDirectory(prefix="onerec_sft_")
        out_path = os.path.join(self._tmpdir.name, "train.parquet")
        df.to_parquet(out_path, index=False)
        return out_path

    # ------------------------------------------------------------------
    # Dataset protocol (delegated)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._inner[index]


def _parse_messages_column(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse messages JSON; keeping as-is.")
            return value
    return value
