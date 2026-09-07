#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export a 5% clip-level local training subset with repo-relative layout."""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


DATASETS = {
    "nuscenes_full": {
        "jsonl": Path("data/sets/nuscenes_full/nurisk_style/qwen_future_vqa_dataset.jsonl"),
        "video_root": Path("data/sets/nuscenes_full/video_clip_dataset"),
    },
    "waymo": {
        "jsonl": Path("data/sets/waymo/nurisk_style/qwen_future_vqa_dataset.jsonl"),
        "video_root": Path("data/sets/waymo/video_clip_dataset"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="exports/local_5pct_nuscenes-devkit")
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--include-model", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    output_root = (repo_root / args.output_root).resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    summary = {
        "fraction": args.fraction,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "datasets": {},
    }
    rng = random.Random(args.seed)
    for dataset_name, cfg in DATASETS.items():
        dataset_summary = export_dataset(repo_root, output_root, dataset_name, cfg, args, rng)
        summary["datasets"][dataset_name] = dataset_summary

    copy_training_code(repo_root, output_root)
    if args.include_model:
        copy_tree_filtered(repo_root / "models" / "Qwen3-VL-2B-Instruct", output_root / "models" / "Qwen3-VL-2B-Instruct")
    write_json(output_root / "EXPORT_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Export root: {output_root}")


def export_dataset(
    repo_root: Path,
    output_root: Path,
    dataset_name: str,
    cfg: Dict[str, Path],
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, Any]:
    jsonl_path = repo_root / cfg["jsonl"]
    video_root = repo_root / cfg["video_root"]
    clip_ids = collect_clip_ids(jsonl_path)
    selected_clip_ids = select_clip_ids(clip_ids, args.fraction, rng)
    train_clip_ids, validation_clip_ids = split_clip_ids(selected_clip_ids, args.train_ratio, rng)

    train_entries, validation_entries, selected_videos = collect_selected_entries(
        jsonl_path,
        train_clip_ids,
        validation_clip_ids,
    )

    output_dataset_root = output_root / "data" / "sets" / dataset_name
    splits_dir = output_dataset_root / "nurisk_style" / "dataset_splits"
    write_json(splits_dir / "train.json", train_entries)
    write_json(splits_dir / "validation.json", validation_entries)

    all_entries = train_entries + validation_entries
    write_json(output_dataset_root / "nurisk_style" / "qwen_future_vqa_dataset.json", {
        "metadata": {
            "description": f"Local 5% clip-level subset for {dataset_name}",
            "total_conversations": len(all_entries),
            "selected_clips": len(selected_clip_ids),
            "source_jsonl": str(cfg["jsonl"]),
        },
        "entries": all_entries,
    })
    write_jsonl(output_dataset_root / "nurisk_style" / "qwen_future_vqa_dataset.jsonl", all_entries)

    copied_bytes = copy_videos(video_root, output_dataset_root / "video_clip_dataset", selected_videos)
    stats = {
        "dataset": dataset_name,
        "source_clips": len(clip_ids),
        "selected_clips": len(selected_clip_ids),
        "train_clips": len(train_clip_ids),
        "validation_clips": len(validation_clip_ids),
        "train_samples": len(train_entries),
        "validation_samples": len(validation_entries),
        "selected_video_files": len(selected_videos),
        "selected_video_bytes": copied_bytes,
        "selected_video_gb": round(copied_bytes / 1e9, 3),
    }
    write_json(splits_dir / "dataset_stats.json", stats)
    return stats


def collect_clip_ids(jsonl_path: Path) -> List[str]:
    clip_ids: Set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            clip_ids.add(str(entry["clip_id"]))
    return sorted(clip_ids)


def select_clip_ids(clip_ids: Sequence[str], fraction: float, rng: random.Random) -> Set[str]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    count = max(1, int(round(len(clip_ids) * fraction)))
    return set(rng.sample(list(clip_ids), count))


def split_clip_ids(clip_ids: Set[str], train_ratio: float, rng: random.Random) -> Tuple[Set[str], Set[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1)")
    shuffled = list(clip_ids)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return set(shuffled[:split]), set(shuffled[split:])


def collect_selected_entries(
    jsonl_path: Path,
    train_clip_ids: Set[str],
    validation_clip_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str]]:
    selected_clip_ids = train_clip_ids | validation_clip_ids
    train_entries: List[Dict[str, Any]] = []
    validation_entries: List[Dict[str, Any]] = []
    selected_videos: Set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            clip_id = str(entry["clip_id"])
            if clip_id not in selected_clip_ids:
                continue
            for video in entry.get("video", []):
                selected_videos.add(str(video))
            if clip_id in train_clip_ids:
                train_entries.append(entry)
            else:
                validation_entries.append(entry)
    return train_entries, validation_entries, selected_videos


def copy_videos(source_video_root: Path, target_video_root: Path, videos: Iterable[str]) -> int:
    copied_bytes = 0
    for rel_video in sorted(videos):
        source = Path(rel_video)
        if not source.is_absolute():
            source = source_video_root / source
        if not source.is_file():
            raise FileNotFoundError(f"Missing video: {source}")
        target = target_video_root / rel_video
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_bytes += target.stat().st_size
    return copied_bytes


def copy_training_code(repo_root: Path, output_root: Path) -> None:
    def ignore_training(dir_path: str, names: Sequence[str]) -> Set[str]:
        ignored = {"__pycache__", ".pytest_cache"}
        path = Path(dir_path)
        if path.name in {"training"}:
            ignored.update({"outputs", "logs"})
        return ignored & set(names)

    shutil.copytree(repo_root / "training", output_root / "training", ignore=ignore_training)


def copy_tree_filtered(source: Path, target: Path) -> None:
    def ignore_common(_: str, names: Sequence[str]) -> Set[str]:
        return {"__pycache__", ".git", ".cache"} & set(names)

    shutil.copytree(source, target, ignore=ignore_common)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
