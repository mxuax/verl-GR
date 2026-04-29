#!/usr/bin/env python3
"""Evaluate OpenOneRec GRPO actor checkpoints.

This script is intentionally a thin orchestration layer:

1. Resolve a pretrained HuggingFace model directory or a verl actor checkpoint,
   usually ``outputs/<experiment>/ckpt/global_step_*/actor``.
2. Merge FSDP actor shards into a HuggingFace model with
   ``python -m verl.model_merger``.
3. Run self-contained OpenOneRec two-stage inference on ``test.parquet`` and
   compute the same SID metrics used by the OpenOneRec benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
VERL_GR_ROOT = EVAL_DIR.parent
WORKSPACE_ROOT = VERL_GR_ROOT.parent
DEFAULT_MERGED_ROOT = EVAL_DIR / "outputs" / "merged_models"
DEFAULT_RESULT_ROOT = EVAL_DIR / "outputs" / "results"

logger = logging.getLogger("openonerec_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a GRPO actor checkpoint and evaluate it on OpenOneRec test.parquet."
    )

    ckpt = parser.add_argument_group("checkpoint merge")
    ckpt.add_argument(
        "--actor-checkpoint",
        type=Path,
        default=None,
        help="Path to global_step_*/actor. If omitted, the latest actor checkpoint is discovered.",
    )
    ckpt.add_argument(
        "--checkpoint-root",
        type=Path,
        default=VERL_GR_ROOT / "outputs",
        help=(
            "Directory to search when --actor-checkpoint is omitted. Accepts an outputs root, "
            "an experiment dir, a ckpt dir containing global_step_*, or a HuggingFace "
            "model/snapshots dir with .safetensors weights."
        ),
    )
    ckpt.add_argument(
        "--global-step",
        type=int,
        default=None,
        help="Select a specific global_step_N under --checkpoint-root/ckpt.",
    )
    ckpt.add_argument(
        "--merged-model-dir",
        type=Path,
        default=None,
        help="Target HuggingFace model directory. Defaults under eval/outputs/merged_models.",
    )
    ckpt.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Use an existing HuggingFace model and skip checkpoint discovery/merge.",
    )
    ckpt.add_argument(
        "--skip-merge",
        action="store_true",
        help="Do not run the model merger. Requires --model-path or an existing --merged-model-dir.",
    )
    ckpt.add_argument(
        "--force-merge",
        action="store_true",
        help="Recreate --merged-model-dir if it already exists.",
    )
    ckpt.add_argument(
        "--merge-backend",
        default="fsdp",
        choices=("fsdp", "megatron"),
        help="Checkpoint backend passed to verl.model_merger.",
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--test-parquet",
        type=Path,
        default=None,
        help=(
            "Path to the OpenOneRec test parquet. Defaults to the first existing "
            "verl_gr/recipes/openonerec/output/rl_data/test.parquet in this fork or sibling verl-GR."
        ),
    )
    data.add_argument(
        "--test-max-sample",
        type=int,
        default=-1,
        help="Maximum evaluation samples. Use -1 to evaluate all rows.",
    )
    data.add_argument("--seed", type=int, default=0, help="Sampling seed used by the benchmark.")

    infer = parser.add_argument_group("two-stage inference")
    infer.add_argument("--backend", choices=("offline", "serving"), default="offline")
    infer.add_argument("--host", default="127.0.0.1", help="Serving backend host.")
    infer.add_argument("--port", type=int, default=8000, help="Serving backend port.")
    infer.add_argument("--base-url", default=None, help="Serving backend base URL.")
    infer.add_argument("--tokenizer", type=Path, default=None, help="Tokenizer path; defaults to model path.")
    infer.add_argument("--trust-remote-code", action="store_true")
    infer.add_argument("--disable-thinking", action="store_true", help="Run single-stage beam search.")
    infer.add_argument("--num-return-thinking", type=int, default=1)
    infer.add_argument("--max-thinking-tokens", type=int, default=1024)
    infer.add_argument("--thinking-temperature", type=float, default=0.6)
    infer.add_argument("--thinking-top-p", type=float, default=0.95)
    infer.add_argument("--thinking-top-k", type=int, default=50)
    infer.add_argument("--num-beams", type=int, default=32)
    infer.add_argument("--max-new-tokens", type=int, default=3)
    infer.add_argument("--prompt-token", default="<|sid_begin|>")
    infer.add_argument("--batch-size", "--bs", type=int, default=None)
    infer.add_argument("--max-concurrent", type=int, default=64)
    infer.add_argument("--max-retries", type=int, default=3)
    infer.add_argument("--rps", type=float, default=float("inf"))
    infer.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    infer.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    infer.add_argument("--max-model-len", type=int, default=None)
    infer.add_argument("--enforce-eager", action="store_true")

    metrics = parser.add_argument_group("metrics/output")
    metrics.add_argument("--k-values", default="1,32")
    metrics.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_ROOT)
    metrics.add_argument("--result-filename", default=None)
    metrics.add_argument("--save-detailed", action="store_true")
    metrics.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the checkpoint merge command without running merge or evaluation.",
    )
    return parser.parse_args()


def default_test_parquet() -> Path:
    candidates = [
        VERL_GR_ROOT / "verl_gr" / "recipes" / "openonerec" / "output" / "rl_data" / "test.parquet",
        WORKSPACE_ROOT / "verl-GR" / "verl_gr" / "recipes" / "openonerec" / "output" / "rl_data" / "test.parquet",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find a default test parquet. Searched:\n  {searched}")


def is_raw_hf_model_checkpoint(path: Path) -> bool:
    """Return true for an already-merged/pretrained HuggingFace model directory."""
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(path.glob("*.safetensors"))
    )


def discover_raw_hf_model_checkpoint(checkpoint_root: Path) -> Path | None:
    root = checkpoint_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    if is_raw_hf_model_checkpoint(root):
        return root

    snapshot_root = root / "snapshots"
    if not snapshot_root.is_dir():
        return None

    candidates = [
        snapshot_dir
        for snapshot_dir in snapshot_root.iterdir()
        if is_raw_hf_model_checkpoint(snapshot_dir)
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def discover_actor_checkpoint(checkpoint_root: Path, global_step: int | None) -> Path:
    root = checkpoint_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    if root.name == "actor" and root.is_dir():
        return root

    ckpt_dirs = []
    if (root / "latest_checkpointed_iteration.txt").is_file():
        ckpt_dirs.append(root)
    if any(root.glob("global_step_*")):
        ckpt_dirs.append(root)
    ckpt_dirs.extend(path.parent for path in root.glob("**/latest_checkpointed_iteration.txt"))

    seen: set[Path] = set()
    unique_ckpt_dirs = []
    for ckpt_dir in ckpt_dirs:
        resolved = ckpt_dir.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_ckpt_dirs.append(resolved)

    candidates: list[tuple[float, int, Path]] = []
    for ckpt_dir in unique_ckpt_dirs:
        step_dirs = [p for p in ckpt_dir.glob("global_step_*") if p.is_dir()]
        if global_step is not None:
            step_dirs = [ckpt_dir / f"global_step_{global_step}"]
        elif (ckpt_dir / "latest_checkpointed_iteration.txt").is_file():
            latest_text = (ckpt_dir / "latest_checkpointed_iteration.txt").read_text().strip()
            if latest_text:
                step_dirs = [ckpt_dir / f"global_step_{int(latest_text)}"]

        for step_dir in step_dirs:
            actor_dir = step_dir / "actor"
            if not actor_dir.is_dir():
                continue
            step = _parse_step(step_dir.name)
            candidates.append((actor_dir.stat().st_mtime, step, actor_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No actor checkpoints found under {root}. Expected global_step_*/actor."
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _parse_step(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def default_merged_model_dir(actor_checkpoint: Path) -> Path:
    step_name = actor_checkpoint.parent.name
    experiment_name = actor_checkpoint.parents[2].name if len(actor_checkpoint.parents) > 2 else "checkpoint"
    return DEFAULT_MERGED_ROOT / f"{experiment_name}_{step_name}"


def build_pythonpath() -> str:
    parts = [
        str(VERL_GR_ROOT),
        str(WORKSPACE_ROOT / "verl_v0.7.1"),
        os.environ.get("PYTHONPATH", ""),
    ]
    return os.pathsep.join(part for part in parts if part)


def run_command(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(_quote(part) for part in cmd)
    print(f"+ {printable}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = build_pythonpath()
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def merge_actor_checkpoint(
    actor_checkpoint: Path,
    target_dir: Path,
    *,
    backend: str,
    force: bool,
    dry_run: bool,
) -> Path:
    actor_checkpoint = actor_checkpoint.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()

    if not actor_checkpoint.is_dir():
        raise FileNotFoundError(f"Actor checkpoint does not exist: {actor_checkpoint}")
    if target_dir.exists():
        if force:
            if not dry_run:
                shutil.rmtree(target_dir)
        elif any(target_dir.iterdir()):
            print(f"Using existing merged model: {target_dir}", flush=True)
            return target_dir

    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        backend,
        "--local_dir",
        str(actor_checkpoint),
        "--target_dir",
        str(target_dir),
    ]
    run_command(cmd, dry_run=dry_run)
    return target_dir


@dataclass
class Sample:
    sample_id: str
    prompt: str
    prompt_len: int
    groundtruth: str
    metadata: dict[str, Any] | None = None


@dataclass
class RequestOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""
    all_generated_texts: list[str] = field(default_factory=list)


@dataclass
class StageMetrics:
    completed: int = 0
    failed: int = 0
    total_input: int = 0
    total_output: int = 0
    duration: float = 0.0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    total_token_throughput: float = 0.0


def load_samples(
    test_parquet: Path,
    tokenizer,
    *,
    max_samples: int,
    enable_thinking: bool,
    seed: int,
) -> list[Sample]:
    import pandas as pd

    df = pd.read_parquet(test_parquet)
    if "messages" not in df.columns:
        raise ValueError(f"OpenOneRec parquet must contain a 'messages' column: {test_parquet}")

    rows = list(df.iterrows())
    random.Random(seed).shuffle(rows)
    limit = len(rows) if max_samples is None or max_samples <= 0 else min(max_samples, len(rows))

    samples: list[Sample] = []
    for _, row in rows:
        if len(samples) >= limit:
            break
        item = row.to_dict()
        messages = _decode_json_like(item.get("messages"))
        if not isinstance(messages, list):
            continue
        messages = _convert_messages_format(messages)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        metadata = _decode_json_like(item.get("metadata"))
        groundtruth = ""
        if isinstance(metadata, dict) and metadata.get("answer") is not None:
            groundtruth = str(metadata["answer"]).strip()
        elif messages:
            groundtruth = str(messages[-1].get("content", "")).strip()
        samples.append(
            Sample(
                sample_id=str(len(samples)),
                prompt=prompt,
                prompt_len=len(tokenizer(prompt).input_ids),
                groundtruth=groundtruth,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
    return samples


def _decode_json_like(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _convert_messages_format(messages: list[Any]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        converted.append({"role": message.get("role", ""), "content": str(content or "")})
    return converted


def extract_ids_from_answer(answer: str) -> set[str]:
    return set(extract_ids_from_answer_ordered(answer))


def extract_ids_from_answer_ordered(answer: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for part in answer.split("<|sid_begin|>"):
        if "<|sid_end|>" not in part:
            continue
        sid = part.split("<|sid_end|>", 1)[0].strip()
        if sid and sid not in seen:
            ordered.append(sid)
            seen.add(sid)
    return ordered


def extract_first_id_from_answer(answer: str) -> str:
    ids = extract_ids_from_answer_ordered(answer)
    return ids[0] if ids else ""


def extract_id_from_generation(generation: str) -> str:
    generation = generation.strip()
    if "</think>" in generation:
        generation = generation.split("</think>")[-1].strip()
    if "<|sid_begin|>" in generation:
        for part in generation.split("<|sid_begin|>"):
            if "<|sid_end|>" in part:
                sid = part.split("<|sid_end|>", 1)[0].strip()
                if sid:
                    return sid
            elif part.strip():
                return part.strip()
    return generation


def evaluate_sid(samples: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    total = len(samples)
    evaluated = 0
    skipped_no_gt = 0
    skipped_no_gen = 0
    pass_counts = {k: 0 for k in k_values}
    pos1_counts = {k: 0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    per_sample: dict[str, dict[str, Any]] = {}
    debug_info: dict[str, Any] = {
        "passed_samples": [],
        "failed_samples": [],
        "no_generation_samples": [],
    }

    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        groundtruth = str(sample.get("groundtruth", ""))
        beams = list(sample.get("beams", []))
        gt_ids = extract_ids_from_answer(groundtruth)
        first_gt = extract_first_id_from_answer(groundtruth)
        if not gt_ids:
            skipped_no_gt += 1
            continue
        if not beams:
            skipped_no_gen += 1
            debug_info["no_generation_samples"].append(
                {"sample_id": sample_id, "ground_truth_sids": extract_ids_from_answer_ordered(groundtruth)}
            )
            continue

        predicted = [extract_id_from_generation(beam) for beam in beams]
        sample_metrics: dict[str, Any] = {}
        pass_results: dict[str, bool] = {}
        pos1_results: dict[str, bool] = {}
        for k in k_values:
            top_k = predicted[:k]
            p = any(sid in gt_ids for sid in top_k)
            p1 = any(sid == first_gt for sid in top_k)
            recall = len(set(top_k) & gt_ids) / len(gt_ids)
            sample_metrics[f"pass@{k}"] = p
            sample_metrics[f"position1_pass@{k}"] = p1
            sample_metrics[f"recall@{k}"] = recall
            pass_results[f"pass@{k}"] = p
            pos1_results[f"position1_pass@{k}"] = p1
            pass_counts[k] += int(p)
            pos1_counts[k] += int(p1)
            recall_sums[k] += recall

        per_sample[sample_id] = sample_metrics
        evaluated += 1
        debug_item = {
            "sample_id": sample_id,
            "ground_truth_sids": extract_ids_from_answer_ordered(groundtruth),
            "first_ground_truth_sid": first_gt,
            "top_10_generations": predicted[:10],
            "pass_results": pass_results,
            "position1_pass_results": pos1_results,
        }
        if any(pass_results.values()):
            debug_info["passed_samples"].append(debug_item)
        else:
            debug_info["failed_samples"].append(debug_item)

    denom = evaluated if evaluated > 0 else 1
    metrics: dict[str, Any] = {
        "total_samples": total,
        "evaluated_samples": evaluated,
        "skipped_no_groundtruth": skipped_no_gt,
        "skipped_no_generation": skipped_no_gen,
    }
    for k in k_values:
        metrics[f"pass@{k}"] = pass_counts[k] / denom
        metrics[f"position1_pass@{k}"] = pos1_counts[k] / denom
        metrics[f"recall@{k}"] = recall_sums[k] / denom
    debug_info["statistics"] = {
        "total_samples": total,
        "passed_samples_count": len(debug_info["passed_samples"]),
        "failed_samples_count": len(debug_info["failed_samples"]),
        "no_generation_samples_count": len(debug_info["no_generation_samples"]),
    }
    debug_info["metrics"] = dict(metrics)
    metrics["per_sample"] = per_sample
    metrics["debug_info"] = debug_info
    return metrics


def calculate_stage_metrics(outputs: list[RequestOutput], duration: float) -> StageMetrics:
    completed = sum(1 for output in outputs if output.success)
    total_input = sum(output.prompt_len for output in outputs if output.success)
    total_output = sum(output.output_tokens for output in outputs if output.success)
    return StageMetrics(
        completed=completed,
        failed=len(outputs) - completed,
        total_input=total_input,
        total_output=total_output,
        duration=duration,
        request_throughput=completed / duration if duration > 0 else 0.0,
        output_throughput=total_output / duration if duration > 0 else 0.0,
        total_token_throughput=(total_input + total_output) / duration if duration > 0 else 0.0,
    )


def run_stage1_offline(llm, samples: list[Sample], args: argparse.Namespace) -> tuple[list[RequestOutput], float]:
    from vllm import SamplingParams  # type: ignore[reportMissingImports]

    params = SamplingParams(
        n=args.num_return_thinking,
        max_tokens=args.max_thinking_tokens,
        temperature=args.thinking_temperature,
        top_p=args.thinking_top_p,
        top_k=args.thinking_top_k,
        stop=["</think>"],
    )
    batch_size = args.batch_size or len(samples)
    outputs: list[RequestOutput] = []
    started = time.perf_counter()
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        for vllm_out, sample in zip(llm.generate([s.prompt for s in batch], params), batch):
            out = RequestOutput(prompt_len=sample.prompt_len)
            if vllm_out.outputs:
                completion = vllm_out.outputs[0]
                out.generated_text = completion.text
                out.output_tokens = len(completion.token_ids)
                out.success = True
            else:
                out.error = "No output from llm.generate()."
            outputs.append(out)
    elapsed = time.perf_counter() - started
    for out in outputs:
        out.latency = elapsed
        out.ttft = elapsed / max(len(outputs), 1)
    return outputs, elapsed


def run_stage2_offline(
    llm,
    tokenizer,
    prompts: list[str],
    prompt_lens: list[int],
    args: argparse.Namespace,
) -> tuple[list[RequestOutput], float]:
    from vllm.sampling_params import BeamSearchParams  # type: ignore[reportMissingImports]

    params = BeamSearchParams(beam_width=args.num_beams, max_tokens=args.max_new_tokens)
    prompt_dicts = [{"prompt": prompt} for prompt in prompts]
    batch_size = args.batch_size or len(prompt_dicts)
    outputs: list[RequestOutput] = []
    started = time.perf_counter()
    for start in range(0, len(prompt_dicts), batch_size):
        batch = prompt_dicts[start : start + batch_size]
        batch_lens = prompt_lens[start : start + batch_size]
        for vllm_out, prompt_len in zip(llm.beam_search(batch, params), batch_lens):
            out = RequestOutput(prompt_len=prompt_len)
            if vllm_out.sequences:
                out.all_generated_texts = [
                    tokenizer.decode(seq.tokens[prompt_len:], skip_special_tokens=True)
                    for seq in vllm_out.sequences
                ]
                out.generated_text = out.all_generated_texts[0] if out.all_generated_texts else ""
                out.output_tokens = sum(max(0, len(seq.tokens) - prompt_len) for seq in vllm_out.sequences)
                out.success = True
            else:
                out.error = "No sequences from llm.beam_search()."
            outputs.append(out)
    elapsed = time.perf_counter() - started
    for out in outputs:
        out.latency = elapsed
        out.ttft = elapsed / max(len(outputs), 1)
    return outputs, elapsed


async def run_stage1_serving(samples: list[Sample], args: argparse.Namespace, model: Path) -> tuple[list[RequestOutput], float]:
    import aiohttp

    url = f"{args.base_url or f'http://{args.host}:{args.port}'}/v1/completions"
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60), trust_env=True) as session:
        started = time.perf_counter()
        outputs = await asyncio.gather(
            *[
                _send_stage1_serving(session, semaphore, url, sample, args, model)
                for sample in samples
            ]
        )
        return list(outputs), time.perf_counter() - started


async def run_stage2_serving(
    prompts: list[str],
    prompt_lens: list[int],
    args: argparse.Namespace,
    model: Path,
) -> tuple[list[RequestOutput], float]:
    import aiohttp

    url = f"{args.base_url or f'http://{args.host}:{args.port}'}/v1/completions"
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60), trust_env=True) as session:
        started = time.perf_counter()
        outputs = await asyncio.gather(
            *[
                _send_stage2_serving(session, semaphore, url, prompt, prompt_len, args, model)
                for prompt, prompt_len in zip(prompts, prompt_lens)
            ]
        )
        return list(outputs), time.perf_counter() - started


async def _send_stage1_serving(session, semaphore, url: str, sample: Sample, args: argparse.Namespace, model: Path) -> RequestOutput:
    payload = {
        "model": str(model),
        "prompt": sample.prompt,
        "max_tokens": args.max_thinking_tokens,
        "temperature": args.thinking_temperature,
        "top_p": args.thinking_top_p,
        "n": args.num_return_thinking,
        "stop": ["</think>"],
        "stream": False,
    }
    if args.thinking_top_k > 0:
        payload["top_k"] = args.thinking_top_k
    return await _post_completion(
        session,
        semaphore,
        url,
        payload,
        sample.prompt_len,
        all_choices=False,
        max_retries=args.max_retries,
    )


async def _send_stage2_serving(
    session,
    semaphore,
    url: str,
    prompt: str,
    prompt_len: int,
    args: argparse.Namespace,
    model: Path,
) -> RequestOutput:
    payload = {
        "model": str(model),
        "prompt": prompt,
        "max_tokens": args.max_new_tokens,
        "n": args.num_beams,
        "use_beam_search": True,
        "temperature": 0.0,
        "stream": False,
    }
    return await _post_completion(
        session,
        semaphore,
        url,
        payload,
        prompt_len,
        all_choices=True,
        max_retries=args.max_retries,
    )


async def _post_completion(
    session,
    semaphore,
    url: str,
    payload: dict[str, Any],
    prompt_len: int,
    *,
    all_choices: bool,
    max_retries: int,
) -> RequestOutput:
    async with semaphore:
        last_output = RequestOutput(prompt_len=prompt_len)
        for attempt in range(max_retries + 1):
            output = RequestOutput(prompt_len=prompt_len)
            started = time.perf_counter()
            try:
                async with session.post(url=url, json=payload) as response:
                    output.latency = time.perf_counter() - started
                    output.ttft = output.latency
                    if response.status != 200:
                        body = await response.text()
                        output.error = f"HTTP {response.status}: {body[:500]}"
                        return output
                    body = await response.json()
                    choices = sorted(body.get("choices", []), key=lambda choice: choice.get("index", 0))
                    usage = body.get("usage", {})
                    if not choices:
                        output.error = "No choices in completion response."
                        return output
                    output.generated_text = choices[0].get("text", "")
                    output.all_generated_texts = (
                        [choice.get("text", "") for choice in choices] if all_choices else [output.generated_text]
                    )
                    output.output_tokens = int(usage.get("completion_tokens", 0))
                    output.success = True
                    return output
            except Exception:
                output.error = "".join(traceback.format_exception(*sys.exc_info()))
                last_output = output
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt)
        return last_output


def run_self_contained_eval(args: argparse.Namespace, model_path: Path, test_parquet: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer.resolve() if args.tokenizer else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    max_samples = args.test_max_sample if args.test_max_sample and args.test_max_sample > 0 else -1
    samples = load_samples(
        test_parquet,
        tokenizer,
        max_samples=max_samples,
        enable_thinking=not args.disable_thinking,
        seed=args.seed,
    )
    logger.info("Loaded %s evaluation samples from %s", len(samples), test_parquet)

    if args.backend == "offline":
        from vllm import LLM  # type: ignore[reportMissingImports]

        engine_kwargs: dict[str, Any] = {
            "model": str(model_path),
            "tokenizer": str(tokenizer_path),
            "trust_remote_code": args.trust_remote_code,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "max_logprobs": 2 * args.num_beams,
        }
        if args.max_model_len is not None:
            engine_kwargs["max_model_len"] = args.max_model_len
        llm = LLM(**engine_kwargs)
        stage1_outputs, stage1_time, stage2_outputs, origin_indices, stage2_time = _run_offline_eval(
            llm, tokenizer, samples, args
        )
    else:
        stage1_outputs, stage1_time, stage2_outputs, origin_indices, stage2_time = asyncio.run(
            _run_serving_eval(tokenizer, samples, args, model_path)
        )

    stage2_all_texts = [
        output.all_generated_texts if output.success else [] for output in stage2_outputs
    ]
    eval_samples = [
        {
            "sample_id": samples[orig_idx].sample_id,
            "groundtruth": samples[orig_idx].groundtruth,
            "beams": stage2_all_texts[s2_idx] if s2_idx < len(stage2_all_texts) else [],
        }
        for s2_idx, orig_idx in enumerate(origin_indices)
    ]
    eval_metrics_sid = evaluate_sid(eval_samples, [int(k.strip()) for k in args.k_values.split(",") if k.strip()])
    eval_per_sample = eval_metrics_sid.pop("per_sample", {})
    debug_info_sid = eval_metrics_sid.pop("debug_info", None)
    stage1_metrics = calculate_stage_metrics(stage1_outputs, stage1_time) if stage1_outputs else None
    stage2_metrics = calculate_stage_metrics(stage2_outputs, stage2_time)
    _log_eval_summary(eval_metrics_sid, stage1_metrics, stage2_metrics, args)
    _save_eval_results(
        args=args,
        model_path=model_path,
        samples=samples,
        eval_metrics=eval_metrics_sid,
        eval_per_sample=eval_per_sample,
        debug_info=debug_info_sid,
        stage1_metrics=stage1_metrics,
        stage2_metrics=stage2_metrics,
        total_duration=stage1_time + stage2_time,
    )


def _run_offline_eval(llm, tokenizer, samples: list[Sample], args: argparse.Namespace):
    if args.disable_thinking:
        prompts = [sample.prompt + args.prompt_token for sample in samples]
        prompt_lens = [len(tokenizer(prompt).input_ids) for prompt in prompts]
        stage2_outputs, stage2_time = run_stage2_offline(llm, tokenizer, prompts, prompt_lens, args)
        return [], 0.0, stage2_outputs, list(range(len(samples))), stage2_time

    stage1_outputs, stage1_time = run_stage1_offline(llm, samples, args)
    prompts, prompt_lens, origin_indices = _build_stage2_prompts(tokenizer, samples, stage1_outputs, args.prompt_token)
    stage2_outputs, stage2_time = run_stage2_offline(llm, tokenizer, prompts, prompt_lens, args)
    return stage1_outputs, stage1_time, stage2_outputs, origin_indices, stage2_time


async def _run_serving_eval(tokenizer, samples: list[Sample], args: argparse.Namespace, model_path: Path):
    if args.disable_thinking:
        prompts = [sample.prompt + args.prompt_token for sample in samples]
        prompt_lens = [len(tokenizer(prompt).input_ids) for prompt in prompts]
        stage2_outputs, stage2_time = await run_stage2_serving(prompts, prompt_lens, args, model_path)
        return [], 0.0, stage2_outputs, list(range(len(samples))), stage2_time

    stage1_outputs, stage1_time = await run_stage1_serving(samples, args, model_path)
    prompts, prompt_lens, origin_indices = _build_stage2_prompts(tokenizer, samples, stage1_outputs, args.prompt_token)
    stage2_outputs, stage2_time = await run_stage2_serving(prompts, prompt_lens, args, model_path)
    return stage1_outputs, stage1_time, stage2_outputs, origin_indices, stage2_time


def _build_stage2_prompts(tokenizer, samples: list[Sample], stage1_outputs: list[RequestOutput], prompt_token: str):
    prompts: list[str] = []
    prompt_lens: list[int] = []
    origin_indices: list[int] = []
    continuation = "</think>\n" + prompt_token
    for idx, (sample, output) in enumerate(zip(samples, stage1_outputs)):
        if not output.success or not output.generated_text:
            continue
        prompt = sample.prompt + output.generated_text + continuation
        prompts.append(prompt)
        prompt_lens.append(len(tokenizer(prompt).input_ids))
        origin_indices.append(idx)
    return prompts, prompt_lens, origin_indices


def _log_eval_summary(
    eval_metrics: dict[str, Any],
    stage1_metrics: StageMetrics | None,
    stage2_metrics: StageMetrics,
    args: argparse.Namespace,
) -> None:
    if stage1_metrics is not None:
        logger.info("Stage 1 completed: %s, duration: %.2fs", stage1_metrics.completed, stage1_metrics.duration)
    logger.info("Stage 2 completed: %s, duration: %.2fs", stage2_metrics.completed, stage2_metrics.duration)
    for k in [int(k.strip()) for k in args.k_values.split(",") if k.strip()]:
        logger.info("pass@%s: %.4f", k, eval_metrics.get(f"pass@{k}", 0.0))
        logger.info("position1_pass@%s: %.4f", k, eval_metrics.get(f"position1_pass@{k}", 0.0))
        logger.info("recall@%s: %.4f", k, eval_metrics.get(f"recall@{k}", 0.0))
    logger.info(
        "Evaluated / Total: %s/%s",
        eval_metrics.get("evaluated_samples", 0),
        eval_metrics.get("total_samples", 0),
    )


def _save_eval_results(
    *,
    args: argparse.Namespace,
    model_path: Path,
    samples: list[Sample],
    eval_metrics: dict[str, Any],
    eval_per_sample: dict[str, dict[str, Any]],
    debug_info: dict[str, Any] | None,
    stage1_metrics: StageMetrics | None,
    stage2_metrics: StageMetrics,
    total_duration: float,
) -> None:
    args.result_dir.mkdir(parents=True, exist_ok=True)
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = args.result_filename or f"openonerec-eval-{model_path.name}-{current_dt}.json"
    result_path = args.result_dir / filename
    result: dict[str, Any] = {
        "backend": args.backend,
        "model": str(model_path),
        "num_prompts": len(samples),
        "enable_thinking": not args.disable_thinking,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "k_values": [int(k.strip()) for k in args.k_values.split(",") if k.strip()],
        "stage2": stage2_metrics.__dict__,
        "total_duration": total_duration,
        "evaluation": eval_metrics,
    }
    if stage1_metrics is not None:
        result["stage1"] = stage1_metrics.__dict__
    if args.save_detailed:
        result["evaluation_per_sample"] = eval_per_sample
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", result_path)

    if args.save_detailed and debug_info is not None:
        debug_path = args.result_dir / f"{current_dt}-debug.json"
        with debug_path.open("w", encoding="utf-8") as handle:
            json.dump(debug_info, handle, indent=2, ensure_ascii=False)
        logger.info("Debug info saved to: %s", debug_path)


def _quote(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return repr(value)
    return value


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    test_parquet = args.test_parquet or default_test_parquet()
    print(f"Test parquet: {test_parquet}", flush=True)

    if args.model_path is not None:
        model_path = args.model_path.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
    else:
        raw_model_checkpoint = None
        if args.actor_checkpoint is None and args.global_step is None:
            raw_model_checkpoint = discover_raw_hf_model_checkpoint(args.checkpoint_root)

        if raw_model_checkpoint is not None:
            model_path = raw_model_checkpoint
            print(f"Raw HuggingFace model checkpoint: {model_path}", flush=True)
        else:
            actor_checkpoint = args.actor_checkpoint or discover_actor_checkpoint(
                args.checkpoint_root, args.global_step
            )
            print(f"Actor checkpoint: {actor_checkpoint}", flush=True)
            merged_dir = args.merged_model_dir or default_merged_model_dir(actor_checkpoint)
            if args.skip_merge:
                model_path = merged_dir.expanduser().resolve()
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"--skip-merge requested but merged model does not exist: {model_path}"
                    )
            else:
                model_path = merge_actor_checkpoint(
                    actor_checkpoint,
                    merged_dir,
                    backend=args.merge_backend,
                    force=args.force_merge,
                    dry_run=args.dry_run,
                )

    print(f"Evaluation model: {model_path}", flush=True)
    if args.dry_run:
        print("Dry run: skipping self-contained inference/evaluation.", flush=True)
        return
    run_self_contained_eval(args, model_path, test_parquet.expanduser().resolve())


if __name__ == "__main__":
    main()
