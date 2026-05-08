"""Merge verl FSDP2 sharded checkpoint into a single HuggingFace model.

Run with the SAME world_size as the checkpoint was saved with::

    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        scripts/merge_fsdp_ckpt.py \\
        --ckpt /path/to/global_step_N/actor \\
        --base_model /path/to/base/Qwen2-0.5B \\
        --output /path/to/merged_hf

The script uses verl's ``get_fsdp_full_state_dict`` to gather the full
(unsharded) state dict on rank 0, then saves a standard HF model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _load_rank_shard(ckpt_path: str, rank: int, world_size: int):
    """Load the model state dict shard for *rank* from the FSDP checkpoint."""
    shard_path = os.path.join(ckpt_path, f"model_world_size_{world_size}_rank_{rank}.pt")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    return torch.load(shard_path, map_location="cpu", weights_only=False)


def _merge_fsdp2_state_dicts(shards: list[dict], world_size: int) -> dict[str, torch.Tensor]:
    """Merge per-rank FSDP2 DTensor shards into a single full state dict.

    Each shard contains ``DTensor`` objects.  We extract ``_local_tensor``
    from each, then reassemble the full tensor by concatenating along the
    placement dimension (shard(0) for most parameters).
    """
    from torch.distributed.tensor import DTensor
    from torch.distributed._tensor.placement_types import Shard

    merged: dict[str, torch.Tensor] = {}
    all_keys = sorted(set().union(*(s.keys() for s in shards)))

    for key in all_keys:
        rank_tensors = []
        for shard in shards:
            if key not in shard:
                break
            val = shard[key]
            if isinstance(val, DTensor):
                rank_tensors.append(val._local_tensor)
            elif isinstance(val, torch.Tensor):
                rank_tensors.append(val)
            else:
                break

        if len(rank_tensors) != world_size:
            # Non-sharded param (same on all ranks) — just use first
            for shard in shards:
                if key in shard:
                    val = shard[key]
                    merged[key] = val._local_tensor.clone() if isinstance(val, DTensor) else val.clone()
                    break
            continue

        # DTensor with Shard(0) placement — concatenate along dim 0
        sample = shards[0][key]
        if isinstance(sample, DTensor):
            placement = sample._spec.placements[0] if sample._spec.placements else None
            if isinstance(placement, Shard):
                merged[key] = torch.cat(rank_tensors, dim=placement.dim)
            else:
                # Replicate placement — all ranks have full copy
                merged[key] = rank_tensors[0].clone()
        else:
            merged[key] = rank_tensors[0].clone()

    return merged


def merge(ckpt_path: str, base_model: str, output_dir: str) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Merging FSDP checkpoint: {ckpt_path}")
        print(f"  world_size = {world_size}")
        print(f"  base_model = {base_model}")

    # Load config and tokenizer from huggingface/ subdir
    hf_path = os.path.join(ckpt_path, "huggingface")
    if not os.path.isdir(hf_path):
        hf_path = base_model

    # Each rank loads its own shard
    shard = _load_rank_shard(ckpt_path, rank, world_size)
    if rank == 0:
        print(f"  loaded {len(shard)} entries from rank shard")

    # Gather all shards to rank 0
    if rank == 0:
        all_shards = [shard]
        for r in range(1, world_size):
            obj = [None]
            dist.recv_object_list(obj, src=r)
            all_shards.append(obj[0])
    else:
        dist.send_object_list([shard], dst=0)

    # Rank 0 merges and saves
    if rank == 0:
        # Merge
        merged_sd = _merge_fsdp2_state_dicts(all_shards, world_size)
        print(f"  merged {len(merged_sd)} parameters")

        # Load model config
        config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

        # Create model and load weights
        print("  building model from config ...")
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)

        model_sd = model.state_dict()
        remapped = {}
        skipped = []
        for k_orig, v in merged_sd.items():
            k_clean = k_orig
            for prefix in ("_fsdp_wrapped_module.",):
                if k_clean.startswith(prefix):
                    k_clean = k_clean[len(prefix):]
            if k_clean in model_sd:
                remapped[k_clean] = v.to(dtype=model_sd[k_clean].dtype)
            else:
                skipped.append(k_orig)

        if skipped:
            print(f"  Warning: {len(skipped)}/{len(merged_sd)} keys unmapped (first 5: {skipped[:5]})")

        model.load_state_dict(remapped, strict=False)

        # Save
        os.makedirs(output_dir, exist_ok=True)
        print(f"  saving to {output_dir} ...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        config.save_pretrained(output_dir)
        print("  Done.")

    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not dist.is_initialized():
        raise RuntimeError("This script must be launched with torchrun.")

    merge(args.ckpt, args.base_model, args.output)


if __name__ == "__main__":
    main()
