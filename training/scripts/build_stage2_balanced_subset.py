#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a small risk-balanced Stage-2 VQA subset without loading full JSONs.

The output keeps nuScenes and Waymo in separate JSON files so each dataset can
keep its own ``vqa_root`` in the training config.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Tuple


SOURCES = {
    "nuscenes": {
        "train_json": Path("data/sets/nuscenes_full/nurisk_style/dataset_splits/train.json"),
        "validation_json": Path("data/sets/nuscenes_full/nurisk_style/dataset_splits/validation.json"),
        "vqa_root": Path("data/sets/nuscenes_full/nurisk_style"),
        "video_root": Path("data/sets/nuscenes_full/video_clip_dataset"),
    },
    "waymo": {
        "train_json": Path("data/sets/waymo/nurisk_style/dataset_splits/train.json"),
        "validation_json": Path("data/sets/waymo/nurisk_style/dataset_splits/validation.json"),
        "vqa_root": Path("data/sets/waymo/nurisk_style"),
        "video_root": Path("data/sets/waymo/video_clip_dataset"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/sets/stage2_balanced_15k/nurisk_style/dataset_splits",
        help="Directory for sampled train/validation JSON files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-total", type=int, default=15000)
    parser.add_argument("--train-high-risk", type=int, default=5000)
    parser.add_argument("--validation-total", type=int, default=200)
    parser.add_argument("--validation-high-risk", type=int, default=80)
    parser.add_argument(
        "--min-views",
        type=int,
        default=5,
        help="Minimum number of camera video paths required. Use 6 to exclude current 5-view Waymo samples.",
    )
    parser.add_argument(
        "--min-video-bytes",
        type=int,
        default=1024,
        help="When selecting candidates, skip samples whose any video file is smaller than this size.",
    )
    parser.add_argument(
        "--reservoir-multiplier",
        type=int,
        default=2,
        help="Keep extra candidates per source/risk bucket so Waymo bad samples can be replaced.",
    )
    parser.add_argument("--progress-every", type=int, default=50000, help="Print scan progress every N entries.")
    parser.add_argument(
        "--validate-selected-decode",
        action="store_true",
        help="Slow but stricter: when selecting candidates, try to decode videos with torchvision and replace failures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sampled JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.json")) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains JSON files. Use --overwrite to replace them.")

    rng = random.Random(args.seed)
    specs = make_bucket_specs(args)
    all_outputs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    all_stats: Dict[str, Any] = {
        "seed": args.seed,
        "train_total_target": args.train_total,
        "train_high_risk_target": args.train_high_risk,
        "train_normal_target": args.train_total - args.train_high_risk,
        "validation_total_target": args.validation_total,
        "validation_high_risk_target": args.validation_high_risk,
        "validation_normal_target": args.validation_total - args.validation_high_risk,
        "high_risk_rule": "min(current_risk_score, future_worst_risk_score) <= 2",
        "quality_filter": {
            "min_views": args.min_views,
            "min_video_bytes": args.min_video_bytes,
            "validate_selected_decode": bool(args.validate_selected_decode),
        },
        "sources": {},
    }

    for split in ("train", "validation"):
        split_outputs, split_stats = build_split(repo_root, split, specs[split], args, rng)
        all_outputs[split] = split_outputs
        all_stats[f"{split}_summary"] = split_stats

    for source in SOURCES:
        train_entries = all_outputs["train"].get(source, [])
        validation_entries = all_outputs["validation"].get(source, [])
        write_json(output_dir / f"train_{source}.json", train_entries)
        write_json(output_dir / f"validation_{source}.json", validation_entries)
        all_stats["sources"][source] = {
            "train_samples": len(train_entries),
            "validation_samples": len(validation_entries),
            "train_risk_distribution": risk_counter(train_entries),
            "validation_risk_distribution": risk_counter(validation_entries),
        }

    write_json(output_dir / "dataset_stats.json", all_stats)
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    print(f"Sampled split directory: {output_dir}")


def make_bucket_specs(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, int]]]:
    return {
        "train": even_source_targets(args.train_high_risk, args.train_total - args.train_high_risk),
        "validation": even_source_targets(
            args.validation_high_risk,
            args.validation_total - args.validation_high_risk,
        ),
    }


def even_source_targets(high_total: int, normal_total: int) -> Dict[str, Dict[str, int]]:
    sources = list(SOURCES)
    high_base, high_rem = divmod(high_total, len(sources))
    normal_base, normal_rem = divmod(normal_total, len(sources))
    targets: Dict[str, Dict[str, int]] = {}
    for index, source in enumerate(sources):
        targets[source] = {
            "high": high_base + (1 if index < high_rem else 0),
            "normal": normal_base + (1 if index < normal_rem else 0),
        }
    return targets


def build_split(
    repo_root: Path,
    split: str,
    target_by_source: Dict[str, Dict[str, int]],
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    candidates: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        source: {"high": [], "normal": []} for source in SOURCES
    }
    seen_counts: MutableMapping[Tuple[str, str], int] = defaultdict(int)
    skipped: MutableMapping[str, Counter] = defaultdict(Counter)

    for source, cfg in SOURCES.items():
        path = repo_root / cfg[f"{split}_json"]
        capacities = {
            group: max(target_by_source[source][group] * args.reservoir_multiplier, target_by_source[source][group])
            for group in ("high", "normal")
        }
        scanned = 0
        for entry in iter_json_array(path):
            scanned += 1
            if args.progress_every > 0 and scanned % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "scan_progress",
                            "split": split,
                            "source": source,
                            "scanned": scanned,
                            "high_seen": seen_counts[(source, "high")],
                            "normal_seen": seen_counts[(source, "normal")],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            group, reason = classify_entry(entry)
            if group is None:
                skipped[source][reason or "unclassified"] += 1
                continue
            reservoir_add(candidates[source][group], entry, capacities[group], seen_counts[(source, group)], rng)
            seen_counts[(source, group)] += 1
        print(
            json.dumps(
                {
                    "event": "scan_done",
                    "split": split,
                    "source": source,
                    "scanned": scanned,
                    "high_seen": seen_counts[(source, "high")],
                    "normal_seen": seen_counts[(source, "normal")],
                    "candidate_pool_high": len(candidates[source]["high"]),
                    "candidate_pool_normal": len(candidates[source]["normal"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    outputs: Dict[str, List[Dict[str, Any]]] = {source: [] for source in SOURCES}
    summary: Dict[str, Any] = {"targets": target_by_source, "seen_valid_candidates": {}, "skipped": {}}
    for source in SOURCES:
        summary["seen_valid_candidates"][source] = {
            group: seen_counts[(source, group)] for group in ("high", "normal")
        }
        summary["skipped"][source] = dict(skipped[source])

    selected_ids: set[str] = set()
    for group in ("high", "normal"):
        for source in SOURCES:
            take = target_by_source[source][group]
            selected = take_validated(
                candidates[source][group],
                take,
                repo_root / SOURCES[source]["video_root"],
                args,
                rng,
                selected_ids,
            )
            outputs[source].extend(selected)

        target_group_total = sum(target_by_source[source][group] for source in SOURCES)
        current_group_total = sum(1 for entries in outputs.values() for entry in entries if risk_group(entry) == group)
        deficit = target_group_total - current_group_total
        if deficit > 0:
            filler_pool: List[Tuple[str, Dict[str, Any]]] = []
            for source in SOURCES:
                for entry in candidates[source][group]:
                    entry_id = str(entry.get("id", ""))
                    if entry_id not in selected_ids:
                        filler_pool.append((source, entry))
            rng.shuffle(filler_pool)
            for source, entry in filler_pool:
                if deficit <= 0:
                    break
                ok, _ = has_valid_video_files(
                    entry,
                    repo_root / SOURCES[source]["video_root"],
                    args.min_views,
                    args.min_video_bytes,
                )
                if not ok:
                    continue
                if args.validate_selected_decode and not can_decode_all_videos(entry, repo_root / SOURCES[source]["video_root"]):
                    continue
                selected_ids.add(str(entry.get("id", "")))
                outputs[source].append(entry)
                deficit -= 1

    for source in SOURCES:
        rng.shuffle(outputs[source])
    summary["actual"] = {
        source: {
            "samples": len(entries),
            "risk_distribution": risk_counter(entries),
            "high_risk_samples": sum(1 for entry in entries if risk_group(entry) == "high"),
            "normal_samples": sum(1 for entry in entries if risk_group(entry) == "normal"),
        }
        for source, entries in outputs.items()
    }
    summary["actual_total"] = sum(len(entries) for entries in outputs.values())
    return outputs, summary


def reservoir_add(
    reservoir: List[Dict[str, Any]],
    entry: Dict[str, Any],
    capacity: int,
    seen_before: int,
    rng: random.Random,
) -> None:
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(entry)
        return
    index = rng.randint(0, seen_before)
    if index < capacity:
        reservoir[index] = entry


def take_validated(
    pool: List[Dict[str, Any]],
    count: int,
    video_root: Path,
    args: argparse.Namespace,
    rng: random.Random,
    selected_ids: set[str],
) -> List[Dict[str, Any]]:
    rng.shuffle(pool)
    selected: List[Dict[str, Any]] = []
    for entry in pool:
        if len(selected) >= count:
            break
        entry_id = str(entry.get("id", ""))
        if entry_id in selected_ids:
            continue
        ok, _ = has_valid_video_files(entry, video_root, args.min_views, args.min_video_bytes)
        if not ok:
            continue
        if args.validate_selected_decode and not can_decode_all_videos(entry, video_root):
            continue
        selected.append(entry)
        selected_ids.add(entry_id)
    return selected


def iter_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk and not buffer[position:].strip():
                break
            buffer = buffer[position:] + chunk
            position = 0
            if not started:
                start = buffer.find("[")
                if start < 0:
                    if not chunk:
                        raise ValueError(f"No JSON array found in {path}")
                    continue
                position = start + 1
                started = True
            while True:
                while position < len(buffer) and buffer[position] in " \r\n\t,":
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    obj, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict):
                    yield obj
                position = end
            if not chunk:
                break


def classify_entry(entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    answer = parse_answer(entry)
    if not isinstance(answer, dict):
        return None, "bad_answer_json"
    scores = [
        get_path(answer, ("current_risk", "risk_score")),
        first_path(answer, ("future_worst_risk", "risk_score"), ("predicted_future_worst_risk", "risk_score")),
    ]
    numeric_scores = [int(score) for score in scores if is_int_like(score)]
    if not numeric_scores:
        return None, "missing_risk_score"
    return ("high" if min(numeric_scores) <= 2 else "normal"), None


def risk_group(entry: Dict[str, Any]) -> str:
    group, _ = classify_entry(entry)
    return group or "unknown"


def parse_answer(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conversations = entry.get("conversations", [])
    if len(conversations) < 2:
        return None
    try:
        answer = json.loads(conversations[1].get("value", ""))
    except json.JSONDecodeError:
        return None
    return answer if isinstance(answer, dict) else None


def risk_counter(entries: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter = Counter()
    for entry in entries:
        answer = parse_answer(entry)
        if not isinstance(answer, dict):
            counter["missing"] += 1
            continue
        score = first_path(answer, ("future_worst_risk", "risk_score"), ("predicted_future_worst_risk", "risk_score"))
        if not is_int_like(score):
            score = get_path(answer, ("current_risk", "risk_score"))
        counter[str(score) if score is not None else "missing"] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def has_valid_video_files(
    entry: Dict[str, Any],
    video_root: Path,
    min_views: int,
    min_video_bytes: int,
) -> Tuple[bool, Optional[str]]:
    videos = entry.get("video", [])
    if not isinstance(videos, list) or len(videos) < min_views:
        return False, f"fewer_than_{min_views}_views"
    for video in videos:
        path = resolve_video_path(str(video), video_root)
        if not path.is_file():
            return False, "missing_video_file"
        if path.stat().st_size < min_video_bytes:
            return False, "too_small_video_file"
    return True, None


def can_decode_all_videos(entry: Dict[str, Any], video_root: Path) -> bool:
    try:
        import torchvision
    except ImportError:
        raise ImportError("--validate-selected-decode requires torchvision")
    for video in entry.get("video", []):
        path = resolve_video_path(str(video), video_root)
        try:
            frames, _, _ = torchvision.io.read_video(str(path), pts_unit="sec", output_format="TCHW")
        except Exception:
            return False
        if frames.numel() == 0:
            return False
    return True


def resolve_video_path(video: str, video_root: Path) -> Path:
    video_path = Path(video)
    if video_path.is_absolute():
        return video_path
    return video_root / video_path


def get_path(obj: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    current: Any = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_path(obj: Dict[str, Any], *paths: Tuple[str, ...]) -> Any:
    for path in paths:
        value = get_path(obj, path)
        if value is not None:
            return value
    return None


def is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
