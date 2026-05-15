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
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    device: str = "cuda",
) -> list[str]:
    """Generate `num_beams` constrained completions for a single prompt.

    Mirror of MiniOneRec ``evaluate.py`` + ``ConstrainedLogitsProcessor``.
    """
    from transformers import LogitsProcessor, LogitsProcessorList

    prefix_index = _constraint_prefix_index(tokenizer)
    hash_dict = _build_hash_dict(info_file, tokenizer, prefix_index=prefix_index)
    eos = tokenizer.eos_token_id

    class ConstrainedLogitsProcessor(LogitsProcessor):
        """Exact mirror of MiniOneRec ``LogitProcessor.py:24-72`` with evaluate.py hash_dict."""

        def __init__(self):
            self.count = 0

        @staticmethod
        def _get_hash(x):
            return '-'.join(str(_) for _ in x)

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

                    allowed = hash_dict.get(self._get_hash(hash_key), [])
                    if not allowed:
                        mask[batch_id * num_beams + beam_id, eos] = 0
                    else:
                        mask[batch_id * num_beams + beam_id, allowed] = 0

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


def _constraint_prefix_index(tokenizer) -> int:
    return 4 if "gpt2" in str(type(tokenizer)).lower() else 3


def _build_hash_dict(info_file: str, tokenizer, *, prefix_index: int | None = None) -> dict[str, list[int]]:
    """Build constraint hash dict — exact mirror of MiniOneRec ``evaluate.py:61-113``."""
    if prefix_index is None:
        prefix_index = _constraint_prefix_index(tokenizer)

    with open(info_file, encoding="utf-8") as f:
        info = f.readlines()

    # Parse: line.split('\t')[0].strip() + "\n"  (evaluate.py:64)
    semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]

    # Prefix with "### Response:\n" (evaluate.py:68)
    info_semantic = [f"### Response:\n{_}" for _ in semantic_ids]

    # Tokenize (evaluate.py:78-80)
    prefixID = [tokenizer(_).input_ids for _ in info_semantic]

    # Build hash_dict (evaluate.py:86-98)
    def _hash(x):
        return '-'.join(str(_) for _ in x)

    hash_dict: dict[str, list[int]] = {}
    for ID in prefixID:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = _hash(ID[:i])
            else:
                hash_number = _hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = []
            hash_dict[hash_number].append(ID[i])
        _ = _hash(ID[prefix_index:])

    # Deduplicate values (evaluate.py:116)
    for k in hash_dict:
        hash_dict[k] = sorted(set(hash_dict[k]))

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


def compute_metrics(completions: list[str], target: str, ks: list[int] = (3, 5, 10)) -> dict:
    """Compute HR@k and NDCG@k per MiniOneRec calc.py convention.

    HR@k  = 1.0 if target found in first k positions, else 0.0
    NDCG@k = (1/log2(pos+2)) / (1/log2(2)) if target found in first k, else 0.0

    Both are averaged across all queries by the caller.
    """
    norm_target = target.strip().strip('\n" ')
    results = {}

    # Find first match position
    match_pos = -1
    for i, c in enumerate(completions):
        if normalize_completion(c) == norm_target:
            match_pos = i
            break

    idcg = 1.0 / math.log2(2)                 # ideal DCG: hit at position 0

    for k in ks:
        if k > len(completions):
            continue
        if 0 <= match_pos < k:
            results[f"HR@{k}"] = 1.0
            results[f"NDCG@{k}"] = (1.0 / math.log2(match_pos + 2)) / idcg
        else:
            results[f"HR@{k}"] = 0.0
            results[f"NDCG@{k}"] = 0.0

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

    # Quick sanity check: generate WITHOUT constraint on first prompt
    import pandas as pd
    df = pd.read_csv(test_file)
    test_prompt = _build_prompt(_parse_history(df.iloc[0].to_dict().get("history_item_sid", "[]")))
    inputs = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=True).to(device)
    # Unconstrained greedy
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"[debug] unconstrained greedy gen: [{gen}]")
    # Unconstrained sampling
    out_s = model.generate(**inputs, max_new_tokens=16, do_sample=True, temperature=1.0,
                           pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    gen_s = tokenizer.decode(out_s[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"[debug] unconstrained sample gen: [{gen_s}]")

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
                num_beams=num_beams, max_new_tokens=256, temperature=1.0, device=device
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
            avg_hr3 = avg.get('HR@3', 0)
            avg_hr5 = avg.get('HR@5', 0)
            avg_hr10 = avg.get('HR@10', 0)
            print(f"  [{idx+1}/{len(df)}] {elapsed:.1f}s | HR@3={avg_hr3:.4f} HR@5={avg_hr5:.4f} HR@10={avg_hr10:.4f}")

    # Final report
    elapsed = time.time() - start
    print(f"\n--- Results for {Path(ckpt_path).name} ---")
    print(f"Samples: {len(all_metrics['HR@3'])} | Time: {elapsed:.1f}s")
    for k in ("HR@3", "NDCG@3", "HR@5", "NDCG@5", "HR@10", "NDCG@10"):
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
    parser.add_argument("--output", default=None, help="Save results to JSON file")
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
        for k in ("HR@3", "NDCG@3", "HR@5", "NDCG@5", "HR@10", "NDCG@10"):
            v1 = sum(results1[k]) / len(results1[k]) if k in results1 else 0
            v2 = sum(results2[k]) / len(results2[k]) if k in results2 else 0
            delta = v1 - v2
            print(f"{k:12s} | {v1:10.4f} | {v2:10.4f} | {delta:+10.4f}")

    # Save results to JSON if --output specified
    if args.output:
        import json as _json
        from datetime import datetime as _dt

        def _summarize(r):
            return {k: sum(v) / len(v) if v else 0.0 for k, v in r.items()}

        payload = {
            "timestamp": _dt.now().isoformat(),
            "ckpt1": str(args.ckpt),
            "ckpt1_metrics": _summarize(results1),
            "test_file": str(args.test_file),
            "info_file": str(args.info_file),
            "num_beams": args.num_beams,
            "num_samples": len(results1.get("HR@3", [])),
        }
        if args.ckpt2 and 'results2' in dir():
            payload["ckpt2"] = str(args.ckpt2)
            payload["ckpt2_metrics"] = _summarize(results2)

        with open(args.output, "w", encoding="utf-8") as _fh:
            _json.dump(payload, _fh, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
