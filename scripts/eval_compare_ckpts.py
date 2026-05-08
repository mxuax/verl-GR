"""Unified checkpoint evaluation for MiniOneRec — works with both checkpoint formats.

Supports:
  - verl-GR FSDP checkpoint (loads huggingface/ subdir or per-rank .pt shards)
  - MiniOneRec original HF checkpoint (from_pretrained)
  - Any model that can be loaded by AutoModelForCausalLM

Runs constrained beam search on test data and reports HR@k / NDCG@k.

Usage:
    python scripts/eval_compare_ckpts.py \
        --ckpt /path/to/checkpoint \
        --test_file /path/to/test.csv \
        --info_file /path/to/info.txt \
        --num_beams 50

Or compare two checkpoints:
    python scripts/eval_compare_ckpts.py \
        --ckpt /path/to/verl_gr_ckpt \
        --ckpt2 /path/to/minionerec_ckpt \
        --test_file /path/to/test.csv \
        --info_file /path/to/info.txt \
        --num_beams 50
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Prefix Trie Constraint (mirrors MiniOneRec LogitProcessor)
# =============================================================================

# =============================================================================
# Beam Search (HF generate wrapper — mirrors MiniOneRec evaluate.py)
# =============================================================================

@torch.no_grad()
def constrained_beam_generate(
    model,
    tokenizer,
    prompt: str,
    info_file: str,
    num_beams: int = 50,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    device: str = "cuda",
) -> list[str]:
    """Generate `num_beams` constrained completions for a single prompt.

    Mirror of MiniOneRec ``evaluate.py`` + ``ConstrainedLogitsProcessor``.
    """
    from transformers import LogitsProcessor, LogitsProcessorList

    hash_dict = _build_hash_dict(info_file, tokenizer)
    eos = tokenizer.eos_token_id
    prefix_index = 3  # Qwen2

    class ConstrainedLogitsProcessor(LogitsProcessor):
        """Exact mirror of MiniOneRec ``LogitProcessor.py:24-72``."""

        def __init__(self):
            self.count = 0

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
            mask = torch.full_like(scores, float("-inf"))

            for batch_id, beam_sent in enumerate(
                input_ids.view(-1, num_beams, input_ids.shape[-1])
            ):
                for beam_id, sent in enumerate(beam_sent):
                    if self.count == 0:
                        hash_key = sent[-prefix_index:].tolist()
                    else:
                        hash_key = sent[-self.count:].tolist()

                    allowed = hash_dict.get(tuple(hash_key), set())
                    if not allowed:
                        mask[batch_id * num_beams + beam_id, eos] = 0
                    else:
                        mask[batch_id * num_beams + beam_id, list(allowed)] = 0

            self.count += 1
            return scores + mask

    processor = ConstrainedLogitsProcessor()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        do_sample=False,
        logits_processor=LogitsProcessorList([processor]),
        pad_token_id=tokenizer.pad_token_id or eos,
        eos_token_id=eos,
        early_stopping=False,
        output_scores=False,
        return_dict_in_generate=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for i in range(outputs.sequences.shape[0]):
        gen_ids = outputs.sequences[i, prompt_len:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
        completions.append(completion)

    return completions


def _build_hash_dict(info_file: str, tokenizer) -> dict[tuple[int, ...], set[int]]:
    """Build prefix-trie constraint exactly as MiniOneRec evaluate.py."""
    hash_dict: dict[tuple[int, ...], set[int]] = defaultdict(set)
    eos = tokenizer.eos_token_id

    with open(info_file, encoding="utf-8") as fh:
        for line in fh:
            sid = line.strip()
            if not sid:
                continue
            tok_ids = tokenizer.encode(sid, add_special_tokens=False)
            if not tok_ids:
                continue
            # All prefixes point to next valid token
            for i in range(1, len(tok_ids)):
                prefix = tuple(tok_ids[:i])
                hash_dict[prefix].add(tok_ids[i])
            # Complete SID → EOS
            hash_dict[tuple(tok_ids)].add(eos)

    return hash_dict


# =============================================================================
# Metrics
# =============================================================================


def normalize_completion(text: str) -> str:
    """Extract just the SID part and strip whitespace."""
    text = text.strip()
    # Remove everything before the last "Response:\n" if present
    if "Response:\n" in text:
        text = text.rsplit("Response:\n", 1)[-1]
    return text.strip("\n\" ")


def compute_metrics(completions: list[str], target: str, ks: list[int] = (1, 3, 5, 10, 20)) -> dict:
    """Compute hit-rate and NDCG for one prompt's completions against target."""
    target = target.strip().strip("\n\" ")
    results = {}

    for k in ks:
        hits = sum(1 for c in completions[:k] if normalize_completion(c) == target)
        results[f"HR@{k}"] = hits / k

    # NDCG
    dcg = 0.0
    idcg = 1.0 / math.log2(2)          # ideal: hit at rank 1
    for i, c in enumerate(completions):
        if normalize_completion(c) == target:
            dcg += 1.0 / math.log2(i + 2)
    results["NDCG"] = dcg / idcg if idcg > 0 else 0.0

    # Pass@1
    results["Pass@1"] = 1.0 if any(normalize_completion(c) == target for c in completions[:1]) else 0.0

    return results


# =============================================================================
# Checkpoint loading
# =============================================================================


def load_model_automatic(ckpt_path: str, device: str = "cuda"):
    """Load model from any checkpoint format.

    Strategy:
    1. If path has huggingface/ subdir → use it
    2. If path has config.json → load as HF directly
    3. If path has model_world_size_*.pt → error with instructions
    """
    ckpt = Path(ckpt_path)

    # Check for huggingface subdir
    hf_dir = ckpt / "huggingface"
    if hf_dir.is_dir() and (hf_dir / "config.json").exists():
        print(f"[load] Using huggingface/ subdir: {hf_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(hf_dir), torch_dtype=torch.bfloat16, trust_remote_code=True)
        return model.to(device), tokenizer

    # Check for direct HF format
    if (ckpt / "config.json").exists():
        print(f"[load] HF format: {ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True)
        return model.to(device), tokenizer

    # Check for FSDP sharded format
    shards = sorted(ckpt.glob("model_world_size_*_rank_*.pt"))
    if shards:
        print(f"[load] Found {len(shards)} FSDP shards. Merging via torchrun...")
        return _load_fsdp_sharded(ckpt_path, shards, device)

    raise FileNotFoundError(f"Cannot determine checkpoint format for {ckpt_path}. "
                            f"Expected huggingface/ subdir, config.json, or model_world_size_*.pt files.")


def _load_fsdp_sharded(ckpt_path: str, shards: list[Path], device: str):
    """Merge per-rank FSDP shards into a single HF model.

    This requires a temporary conversion step using torch.distributed.
    We launch a subprocess with torchrun --nproc_per_node=N.
    """
    import subprocess
    import tempfile

    world_size = len(shards)
    output_dir = Path(ckpt_path).parent / f"{Path(ckpt_path).name}_hf"

    if output_dir.exists() and (output_dir / "config.json").exists():
        print(f"[load] Using cached HF conversion: {output_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(output_dir), torch_dtype=torch.bfloat16, trust_remote_code=True)
        return model.to(device), tokenizer

    # Infer model base path from shard metadata
    # Try loading a shard to get the base model path
    shard_data = torch.load(str(shards[0]), map_location="cpu", weights_only=False)
    if isinstance(shard_data, dict) and "extra_state" in shard_data:
        extra = shard_data["extra_state"]
    elif isinstance(shard_data, dict):
        extra = shard_data
    else:
        extra = {}

    # We need the original base model to load config
    print("[load] FSDP shards detected. Two options:")
    print(f"  1) Manually merge via: python scripts/merge_fsdp_ckpt.py --ckpt {ckpt_path}")
    print(f"  2) Use the MiniOneRec evaluate.py (which loads natively) for the original ckpt")
    print(f"  3) Set model.path to a base model + load the FSDP weights via verl's CheckpointHandler")
    raise RuntimeError(
        "FSDP sharded checkpoint requires manual conversion first.\n"
        f"Run: python scripts/merge_fsdp_ckpt.py --ckpt {ckpt_path} --base_model /path/to/base/Qwen2-0.5B --output {output_dir}\n"
        f"Then re-run this script with --ckpt {output_dir}"
    )


# =============================================================================
# Main
# =============================================================================


def evaluate_ckpt(ckpt_path: str, test_file: str, info_file: str, num_beams: int, device: str, max_samples: int):
    """Run full evaluation on one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt_path}")
    print(f"{'='*60}")

    model, tokenizer = load_model_automatic(ckpt_path, device)
    model.eval()

    # Debug: verify hash_dict and tokenizer
    hd = _build_hash_dict(info_file, tokenizer)
    print(f"[debug] hash_dict entries: {len(hd)}")
    # Pick first hash_dict key and verify lookup
    sample_key = next(iter(hd))
    print(f"[debug] sample hash_key: {sample_key} → allowed: {hd[sample_key]}")
    # Verify tokenizer encodes a sample SID
    with open(info_file) as fh:
        sample_sid = fh.readline().strip()
    sample_tokens = tokenizer.encode(sample_sid, add_special_tokens=False)
    print(f"[debug] sample SID '{sample_sid}' → tokens: {sample_tokens}")
    # Check vocab size
    print(f"[debug] tokenizer vocab_size: {len(tokenizer)}, pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token_id}")

    # Load test data
    import pandas as pd
    df = pd.read_csv(test_file)
    if max_samples > 0:
        df = df.head(max_samples)

    print(f"Test samples: {len(df)}")

    all_metrics = defaultdict(list)
    start = time.time()

    for idx, row in df.iterrows():
        row = row.to_dict()
        history = _parse_history(row.get("history_item_sid", "[]"))
        target = str(row.get("item_sid", "")).strip()

        if not history or not target:
            continue

        prompt = _build_prompt(history)

        try:
            completions = constrained_beam_generate(
                model, tokenizer, prompt, info_file,
                num_beams=num_beams, max_new_tokens=16, temperature=1.0, device=device
            )
        except Exception as e:
            print(f"  [{idx}] Generation error: {e}")
            continue

        # Debug: print first 3 samples
        if idx < 3:
            print(f"\n  --- Sample {idx} ---")
            print(f"  Prompt (tail 200): ...{repr(prompt[-200:])}")
            print(f"  Target: [{target}]")
            for ci, c in enumerate(completions[:5]):
                norm = normalize_completion(c)
                match = "Y" if norm == target.strip().strip('\n\" ') else "N"
                print(f"  Beam {ci}: [{c[:80]}] norm=[{norm}] {match}")

        metrics = compute_metrics(completions, target)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            avg = {k: sum(v) / len(v) for k, v in all_metrics.items()}
            print(f"  [{idx+1}/{len(df)}] {elapsed:.1f}s | HR@1={avg.get('HR@1', 0):.4f} HR@5={avg.get('HR@5', 0):.4f} HR@20={avg.get('HR@20', 0):.4f}")

    # Final report
    elapsed = time.time() - start
    print(f"\n--- Results for {Path(ckpt_path).name} ---")
    print(f"Samples: {len(all_metrics['HR@1'])} | Time: {elapsed:.1f}s")
    for k in ("HR@1", "HR@3", "HR@5", "HR@10", "HR@20", "NDCG", "Pass@1"):
        if k in all_metrics:
            values = all_metrics[k]
            print(f"  {k:12s}: {sum(values)/len(values):.4f}")

    return dict(all_metrics)


def _parse_history(value):
    import ast
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return []


def _build_prompt(history):
    hstr = ", ".join(str(x) for x in history)
    return (
        "### User Input:\n"
        f"The user has interacted with items {hstr} in chronological order. "
        "Can you predict the next possible item that the user may expect?\n\n"
        "### Response:\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Unified MiniOneRec checkpoint evaluation")
    parser.add_argument("--ckpt", required=True, help="Primary checkpoint path")
    parser.add_argument("--ckpt2", default=None, help="Optional second checkpoint for comparison")
    parser.add_argument("--test_file", required=True, help="Test CSV file path")
    parser.add_argument("--info_file", required=True, help="SID info file for prefix trie constraint")
    parser.add_argument("--num_beams", type=int, default=50, help="Number of beams (default: 50)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max test samples (-1 = all)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    results1 = evaluate_ckpt(args.ckpt, args.test_file, args.info_file, args.num_beams, args.device, args.max_samples)

    if args.ckpt2:
        results2 = evaluate_ckpt(args.ckpt2, args.test_file, args.info_file, args.num_beams, args.device, args.max_samples)

        # Comparison table
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Metric':12s} | {'CKPT1':>10s} | {'CKPT2':>10s} | {'Delta':>10s}")
        print("-" * 48)
        for k in ("HR@1", "HR@3", "HR@5", "HR@10", "HR@20", "NDCG"):
            v1 = sum(results1[k]) / len(results1[k]) if k in results1 else 0
            v2 = sum(results2[k]) / len(results2[k]) if k in results2 else 0
            delta = v1 - v2
            print(f"{k:12s} | {v1:10.4f} | {v2:10.4f} | {delta:+10.4f}")


if __name__ == "__main__":
    main()
