#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 7b: sample a fixed-size high-risk Bench2Drive VQA training set."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from common import DEFAULT_DATAROOT, mkdir, output_root


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scenario_prefix(scene_name: str) -> str:
    return scene_name.split("_Town")[0] if "_Town" in scene_name else scene_name.split("_")[0]


def answer(entry: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(entry["conversations"][1]["value"])
    except (KeyError, IndexError, json.JSONDecodeError):
        return {}


def current_score(entry: Dict[str, Any]) -> int:
    value = answer(entry).get("current_risk", {}).get("risk_score")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 99


def is_high_risk(entry: Dict[str, Any], max_risk_score: int) -> bool:
    return current_score(entry) <= max_risk_score


def balanced_sample(entries: Sequence[Dict[str, Any]], target_samples: int, seed: int) -> List[Dict[str, Any]]:
    if len(entries) <= target_samples:
        return list(entries)
    rng = random.Random(seed)
    by_prefix: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_prefix[scenario_prefix(str(entry.get("scene", "")))].append(entry)
    for rows in by_prefix.values():
        rows.sort(key=lambda row: (current_score(row), row.get("clip_id", ""), row.get("id", "")))
        strongest = rows[: max(1, len(rows) // 2)]
        rest = rows[max(1, len(rows) // 2) :]
        rng.shuffle(strongest)
        rng.shuffle(rest)
        rows[:] = strongest + rest

    selected = []
    prefixes = sorted(by_prefix)
    while len(selected) < target_samples and prefixes:
        remaining = []
        for prefix in prefixes:
            rows = by_prefix[prefix]
            if rows:
                selected.append(rows.pop(0))
                if len(selected) >= target_samples:
                    break
            if rows:
                remaining.append(prefix)
        prefixes = remaining
    selected.sort(key=lambda row: (row.get("scene", ""), row.get("clip_id", ""), row.get("id", "")))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None, help="Default: <dataroot>/nurisk_style")
    parser.add_argument("--output-dir", default=None, help="Default: <input-dir>/high_risk_vqa_10000")
    parser.add_argument("--target-samples", type=int, default=10000)
    parser.add_argument("--max-risk-score", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = output_root(args.dataroot, args.input_dir)
    outdir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / f"high_risk_vqa_{args.target_samples}"
    source = load_json(root / "qwen_future_vqa_dataset.json")
    entries = [entry for entry in source.get("entries", []) if is_high_risk(entry, args.max_risk_score)]
    selected = balanced_sample(entries, args.target_samples, args.seed)

    rng = random.Random(args.seed)
    clip_ids = sorted({entry.get("clip_id") for entry in selected})
    rng.shuffle(clip_ids)
    train_clip_count = int(round(len(clip_ids) * args.train_ratio))
    train_clips = set(clip_ids[:train_clip_count])
    train = [entry for entry in selected if entry.get("clip_id") in train_clips]
    validation = [entry for entry in selected if entry.get("clip_id") not in train_clips]

    output = {
        "metadata": {
            "source": str(root / "qwen_future_vqa_dataset.json"),
            "target_samples": args.target_samples,
            "selected_samples": len(selected),
            "candidate_high_risk_samples": len(entries),
            "max_risk_score": args.max_risk_score,
            "train_samples": len(train),
            "validation_samples": len(validation),
            "selected_by_prefix": dict(Counter(scenario_prefix(str(entry.get("scene", ""))) for entry in selected)),
            "selected_by_current_risk_score": dict(Counter(str(current_score(entry)) for entry in selected)),
        },
        "entries": selected,
    }
    write_json(outdir / "qwen_future_vqa_dataset.json", output)
    write_jsonl(outdir / "qwen_future_vqa_dataset.jsonl", selected)
    write_json(outdir / "train.json", train)
    write_json(outdir / "validation.json", validation)
    write_json(outdir / "dataset_stats.json", output["metadata"])
    print("Stage 7b done.")
    print(f"Candidate high-risk samples: {len(entries)}")
    print(f"Selected samples: {len(selected)}")
    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Output dir: {outdir}")


if __name__ == "__main__":
    main()
