#!/usr/bin/env python3
"""H22: compare shuffled training record order vs original MiniOneRec rl.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

VERL_GR_ROOT = Path(__file__).resolve().parents[2]
MINIONEREC_ROOT = VERL_GR_ROOT.parent / "MiniOneRec"
if str(VERL_GR_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_GR_ROOT))
if str(MINIONEREC_ROOT) not in sys.path:
    sys.path.insert(0, str(MINIONEREC_ROOT))

from data import RLSeqTitle2SidDataset, RLTitle2SidDataset, SidDataset  # noqa: E402
from datasets import Dataset, concatenate_datasets  # noqa: E402
from verl_gr.recipes.minionerec.minionerec_dataset import MiniOneRecDataset  # noqa: E402


def build_orig_records(train_csv: str, item_meta: str, sid_index: str, category: str) -> list[dict]:
    train_datasets = [
        SidDataset(train_csv, category=category),
        RLTitle2SidDataset(item_file=item_meta, index_file=sid_index, category=category),
        RLSeqTitle2SidDataset(train_csv, sample=10000, seed=0, category=category),
    ]
    merged = {k: [elm[k] for elm in train_datasets[0]] for k in train_datasets[0][0].keys()}
    for ds in train_datasets[1:]:
        for k in merged:
            merged[k].extend([elm[k] for elm in ds])
    dataset = Dataset.from_dict(merged).shuffle(seed=42)
    records = []
    for row in dataset:
        prompt = row["prompt"]
        target = str(row.get("completion", row.get("response", ""))).strip()
        task = "unknown"
        if "title sequence" in prompt:
            task = "seq_title2sid"
        elif "title:" in prompt.lower() and "description" not in prompt.lower():
            task = "title2sid"
        elif "description" in prompt.lower():
            task = "description2sid"
        elif "historical interactions" in prompt.lower() or "history" in prompt.lower():
            task = "sid"
        records.append({"task": task, "prompt": prompt, "target": target})
    return records


def build_verl_records(train_csv: str, item_meta: str, sid_index: str, seed: int, seq_seed: int) -> list[dict]:
    cfg = OmegaConf.create(
        {
            "cache_dir": "~/.cache/verl/rlhf",
            "prompt_key": "prompt",
            "max_prompt_length": 2560,
            "truncation": "left",
            "filter_overlong_prompts": False,
            "shuffle": True,
            "seed": seed,
            "category": "Industrial_and_Scientific",
            "include_alignment_tasks": True,
            "include_alignment_tasks_for_val": False,
            "sid_index_path": sid_index,
            "item_meta_path": item_meta,
            "seq_title_sample": 10000,
            "seq_title_sample_seed": seq_seed,
            "val_files": [str(MINIONEREC_ROOT / "data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv")],
        }
    )
    tok = AutoTokenizer.from_pretrained(str(MINIONEREC_ROOT / "output_dir/xxx/checkpoint-390"))
    tok.pad_token = tok.eos_token
    ds = MiniOneRecDataset([train_csv], tok, cfg)
    records = []
    for i in range(len(ds.dataframe)):
        row = ds.dataframe[i]
        extra = row.get("extra_info") or {}
        records.append(
            {
                "task": str(extra.get("task", "unknown")),
                "prompt": str(row["prompt"]),
                "target": str((row.get("reward_model") or {}).get("ground_truth", "")).strip(),
            }
        )
    return records


def task_distribution(records: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r["task"]] = out.get(r["task"], 0) + 1
    return out


def compare_records(orig: list[dict], verl: list[dict], n: int) -> dict:
    limit = min(n, len(orig), len(verl))
    mismatches = []
    for i in range(limit):
        o, v = orig[i], verl[i]
        if o["prompt"] != v["prompt"] or o["target"] != v["target"]:
            mismatches.append(
                {
                    "index": i,
                    "orig": o,
                    "verl": v,
                }
            )
    return {
        "orig_len": len(orig),
        "verl_len": len(verl),
        "compared": limit,
        "mismatch_count": len(mismatches),
        "orig_task_dist": task_distribution(orig),
        "verl_task_dist": task_distribution(verl),
        "first_mismatches": mismatches[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(MINIONEREC_ROOT / "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"))
    parser.add_argument("--item-meta", default=str(MINIONEREC_ROOT / "data/Amazon/index/Industrial_and_Scientific.item.json"))
    parser.add_argument("--sid-index", default=str(MINIONEREC_ROOT / "data/Amazon/index/Industrial_and_Scientific.index.json"))
    parser.add_argument("--category", default="industrial and scientific items")
    parser.add_argument("--compare-n", type=int, default=160)
    parser.add_argument("--out", default=str(VERL_GR_ROOT / "logs/convergence/h22_data_order.json"))
    args = parser.parse_args()

    orig = build_orig_records(args.train_csv, args.item_meta, args.sid_index, args.category)
    verl = build_verl_records(args.train_csv, args.item_meta, args.sid_index, seed=42, seq_seed=0)
    result = compare_records(orig, verl, args.compare_n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["mismatch_count"] == 0 and result["orig_len"] == result["verl_len"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
