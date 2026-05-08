"""Offline SID token expansion for MiniOneRec models.

Loads a base model + tokenizer, injects all SID tokens from
``sid_index.json``, resizes the model's embedding matrix, and saves
the expanded checkpoint.  The output should be used as
``model.path`` for SFT (and optionally RL) training.

Usage::

    python -m verl_gr.recipes.minionerec.data.expand_vocab \\
        --base_model /path/to/Qwen2-0.5B \\
        --sid_index_path data/Amazon/index/Industrial_and_Scientific.index.json \\
        --output ./expanded_model

Mirrors the original MiniOneRec ``sft.py`` ``TokenExtender`` logic
(lines 30-56, 149-159).
"""

from __future__ import annotations

import argparse
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def collect_sid_tokens(sid_index_path: str) -> set[str]:
    """Read sid_index.json and return the set of individual SID component tokens."""
    with open(sid_index_path, encoding="utf-8") as fh:
        sid_index: dict[str, list[str]] = json.load(fh)

    tokens: set[str] = set()
    for sids in sid_index.values():
        for token in sids:
            tokens.add(str(token))
    return tokens


def expand_and_save(base_model: str, sid_index_path: str, output_dir: str) -> None:
    print(f"Loading base model from {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # Collect SID tokens
    sid_tokens = collect_sid_tokens(sid_index_path)
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = sorted(t for t in sid_tokens if t not in existing)

    if not new_tokens:
        print("No new tokens to add — vocab already contains all SID tokens.")
        print(f"Saving tokenizer + model to {output_dir} ...")
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        return

    print(f"Adding {len(new_tokens)} new tokens to tokenizer ...")
    tokenizer.add_tokens(new_tokens)

    print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)} ...")
    model.resize_token_embeddings(len(tokenizer))

    # Mirror original: pad_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Set pad_token = eos_token")

    print(f"Saving expanded model + tokenizer to {output_dir} ...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand vocab with SID tokens for MiniOneRec SFT")
    parser.add_argument("--base_model", required=True, help="Path to base HF model")
    parser.add_argument("--sid_index_path", required=True, help="Path to .index.json file")
    parser.add_argument("--output", required=True, help="Output directory for expanded model")
    args = parser.parse_args()

    expand_and_save(args.base_model, args.sid_index_path, args.output)


if __name__ == "__main__":
    main()
