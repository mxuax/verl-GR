"""Convert DDP model.pt checkpoint to HuggingFace format.

DDP engine saves ``model.pt`` (raw state dict via ``torch.save``) for
resuming training.  This script converts it into a HuggingFace checkpoint
(``config.json`` + ``pytorch_model.bin`` + tokenizer files) that
``AutoModelForCausalLM.from_pretrained`` and ``eval_compare_ckpts.py``
can load directly.

Usage:
    # Convert a single checkpoint
    python scripts/convert_ddp_to_hf.py \\
        --ckpt outputs/.../global_step_165/actor \\
        --base_model /path/to/Qwen2-0.5B

    # Batch convert all actor checkpoints under a directory
    python scripts/convert_ddp_to_hf.py \\
        --ckpt outputs/my_run/ckpt \\
        --base_model /path/to/Qwen2-0.5B \\
        --batch

The output goes to ``<ckpt>/huggingface/`` by default.  In batch mode,
omit ``--output`` so each checkpoint writes into its own actor directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def convert_one(ckpt_dir: str, base_model: str, output_dir: str | None = None) -> str:
    """Convert a single DDP model.pt checkpoint to HF format.

    Args:
        ckpt_dir: Directory containing ``model.pt``.
        base_model: Path to the original HuggingFace model (for config/tokenizer).
        output_dir: Output directory (default: ``<ckpt_dir>/huggingface``).

    Returns:
        Path to the output directory.
    """
    import torch
    from transformers import AutoConfig

    ckpt_dir = os.path.abspath(ckpt_dir)
    model_pt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.isfile(model_pt):
        raise FileNotFoundError(f"model.pt not found in {ckpt_dir}")

    hf_dir = os.path.abspath(output_dir or os.path.join(ckpt_dir, "huggingface"))
    os.makedirs(hf_dir, exist_ok=True)

    print(f"[convert] Loading state_dict from {model_pt}")
    state_dict = torch.load(model_pt, map_location="cpu")
    # Handle wrapped state_dict (some checkpoints may have an extra 'module.' prefix
    # from DDP unwrapping, though our DDP engine strips it during save).
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        print("[convert] Stripped 'module.' prefix from state_dict keys")

    print(f"[convert] Loading config from {base_model}")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.save_pretrained(hf_dir)
    print(f"[convert] Saved config.json to {hf_dir}")

    # Save weights as pytorch_model.bin (HF-compatible)
    weight_file = os.path.join(hf_dir, "pytorch_model.bin")
    torch.save(state_dict, weight_file)
    print(f"[convert] Saved weights ({len(state_dict)} keys) to {weight_file}")

    # Copy tokenizer files from base model
    tokenizer_files = [
        f for f in os.listdir(base_model)
        if os.path.isfile(os.path.join(base_model, f))
        and (
            "tokenizer" in f.lower()
            or f in ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
                     "tokenizer_config.json", "chat_template.jinja")
        )
    ]
    for fname in tokenizer_files:
        shutil.copy2(os.path.join(base_model, fname), os.path.join(hf_dir, fname))
    print(f"[convert] Copied tokenizer files: {tokenizer_files}")

    print(f"[convert] Done: {hf_dir}")
    return hf_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert DDP model.pt checkpoints to HuggingFace format"
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint directory (or parent dir with --batch)")
    parser.add_argument("--base_model", required=True, help="Path to base HuggingFace model")
    parser.add_argument("--output", default=None,
                        help="Output directory for a single checkpoint (default: <ckpt>/huggingface)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch convert all global_step_*/actor subdirs under --ckpt")
    args = parser.parse_args()

    if args.batch:
        if args.output is not None:
            parser.error(
                "--output cannot be used with --batch because a single output directory "
                "would be overwritten by each checkpoint; omit --output to write each "
                "checkpoint to its own <actor>/huggingface directory."
            )
        ckpt_root = Path(args.ckpt)
        actor_dirs = sorted(ckpt_root.glob("global_step_*/actor"))
        if not actor_dirs:
            print(f"No global_step_*/actor directories found under {args.ckpt}")
            sys.exit(1)
        print(f"Found {len(actor_dirs)} actor checkpoint(s)")
        for d in actor_dirs:
            try:
                convert_one(str(d), args.base_model)
            except Exception as e:
                print(f"[error] {d}: {e}")
    else:
        convert_one(args.ckpt, args.base_model, args.output)


if __name__ == "__main__":
    main()
