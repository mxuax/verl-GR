"""Merge verl FSDP sharded checkpoint into a single HuggingFace model.

Usage:
    python scripts/merge_fsdp_ckpt.py \
        --ckpt /path/to/global_step_1050/actor \
        --base_model /path/to/base/Qwen2-0.5B \
        --output /path/to/merged_hf

If --base_model is not given, tries to infer from the checkpoint's extra_state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def find_base_model_from_extra(ckpt_path: Path) -> str | None:
    """Try to extract base model path from FSDP checkpoint metadata."""
    shards = sorted(ckpt_path.glob("model_world_size_*_rank_0.pt"))
    if not shards:
        # Try optim/extra state
        extras = sorted(ckpt_path.glob("extra_state_world_size_*_rank_0.pt"))
        if extras:
            data = torch.load(str(extras[0]), map_location="cpu", weights_only=False)
            print(f"[infer] extra_state keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        return None

    data = torch.load(str(shards[0]), map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        # Look for model path in metadata
        for key in ("model_path", "base_model", "hf_config_path", "tokenizer_path"):
            if key in data:
                return data[key]
    return None


def merge(ckpt_path: str, base_model: str | None, output_dir: str) -> None:
    ckpt = Path(ckpt_path)
    shards = sorted(ckpt.glob("model_world_size_*_rank_*.pt"))

    if not shards:
        raise FileNotFoundError(f"No model_world_size_*_rank_*.pt files found in {ckpt_path}")

    world_size = len(set(
        int(re.search(r"world_size_(\d+)", s.name).group(1))
        for s in shards
    ))
    print(f"Detected world_size={world_size} with {len(shards)} shard files")

    # Determine base model
    if base_model is None:
        base_model = find_base_model_from_extra(ckpt)
    if base_model is None:
        raise ValueError(
            "Cannot determine base model. Please provide --base_model.\n"
            "This should be the same model path used during training."
        )
    print(f"Base model: {base_model}")

    # Load config and model structure
    print("Loading model config ...")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    print("Creating empty model ...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # Merge shards: simplest approach — load each rank's shard and merge into full state dict
    full_sd = {}
    for shard_path in shards:
        print(f"  loading {shard_path.name} ...")
        sd = torch.load(str(shard_path), map_location="cpu", weights_only=False)

        # FSDP state dicts may have _flat_param keys or regular keys
        if isinstance(sd, dict):
            # Check if this is a "model" shard (contains state dict under a key)
            if "model" in sd:
                sd = sd["model"]
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]

            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    # FSDP1: flat params need to be unflattened
                    # FSDP2: regular named params
                    full_sd[k] = v.clone()

    if not full_sd:
        raise RuntimeError("Failed to extract parameters from shards. "
                           "The checkpoint format may not be supported by this simple merge script.")

    print(f"Merged {len(full_sd)} parameter keys. Loading into model ...")

    # Try to load — may need key name remapping for FSDP wrappers
    # FSDP wraps module names with _fsdp_wrapped_module prefix
    model_sd = model.state_dict()
    remapped = {}
    skipped = 0
    for k, v in full_sd.items():
        clean = k
        # Strip FSDP wrapper prefixes
        for prefix in ("_fsdp_wrapped_module.", "module.", "model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        if clean in model_sd:
            remapped[clean] = v.to(dtype=model_sd[clean].dtype)
        else:
            skipped += 1

    if skipped:
        print(f"Warning: {skipped} keys could not be mapped to model. "
              f"First few: {[k for k in full_sd if k not in {v: True for v in remapped}][:5]}")

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP sharded checkpoint into HF format")
    parser.add_argument("--ckpt", required=True, help="Path to FSDP checkpoint directory")
    parser.add_argument("--base_model", default=None, help="Base model path (auto-detected if not given)")
    parser.add_argument("--output", required=True, help="Output directory for merged HF model")
    args = parser.parse_args()
    merge(args.ckpt, args.base_model, args.output)


if __name__ == "__main__":
    main()
