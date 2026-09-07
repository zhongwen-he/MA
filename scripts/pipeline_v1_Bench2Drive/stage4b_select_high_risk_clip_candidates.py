#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 4b: select high-risk Bench2Drive clip candidates before video encoding.

This stage reads Stage 4 risk JSON files and writes a compact clip-selection
manifest. Stage 5a can then generate videos only for the selected clips.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import DEFAULT_DATAROOT, DEFAULT_KEYFRAME_INTERVAL_SECONDS, format_time_key, mkdir, output_root


DEFAULT_HIGH_RISK_PREFIXES = {
    "Accident",
    "AccidentTwoWays",
    "BlockedIntersection",
    "ConstructionObstacle",
    "ConstructionObstacleTwoWays",
    "ControlLoss",
    "CrossingBicycleFlow",
    "DynamicObjectCrossing",
    "HardBreakRoute",
    "HazardAtSideLane",
    "HazardAtSideLaneTwoWays",
    "HighwayCutIn",
    "InvadingTurn",
    "OppositeVehicleRunningRedLight",
    "OppositeVehicleTakingPriority",
    "ParkedObstacle",
    "ParkedObstacleTwoWays",
    "ParkingCrossingPedestrian",
    "ParkingCutIn",
    "PedestrianCrossing",
    "StaticCutIn",
    "VehicleOpensDoorTwoWays",
    "VehicleTurningRoutePedestrian",
    "YieldToEmergencyVehicle",
}


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


def parse_prefixes(value: str) -> set:
    if value.strip().lower() in {"default", "high_risk", "high-risk"}:
        return set(DEFAULT_HIGH_RISK_PREFIXES)
    if value.strip().lower() == "all":
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_time_key(key: str, interval_seconds: float) -> Optional[int]:
    try:
        text = key.strip()
        if text.startswith("At ") and text.endswith(" seconds"):
            seconds = float(text[len("At ") : -len(" seconds")])
            return int(round(seconds / interval_seconds))
    except ValueError:
        return None
    return None


def agent_score(agent_data: Dict[str, Any]) -> Optional[int]:
    score = agent_data.get("Risk Analysis", {}).get("Overall Risk Score")
    try:
        return int(score)
    except (TypeError, ValueError):
        return None


def scene_num_frames(scene_dir: Path) -> int:
    ego_path = scene_dir / "ego_trajectory.csv"
    if not ego_path.exists():
        return 0
    with open(ego_path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def risk_candidates_for_scene(
    scene_dir: Path,
    clip_len: int,
    clip_stride: int,
    interval_seconds: float,
    max_risk_score: int,
    min_target_agents: int,
) -> List[Dict[str, Any]]:
    scene_name = scene_dir.name
    risk_path = scene_dir / "risk_scores_output_enhanced.json"
    if not risk_path.exists():
        return []
    risk_data = load_json(risk_path)
    num_frames = scene_num_frames(scene_dir)
    if num_frames < clip_len:
        return []

    rows = []
    for start in range(0, num_frames - clip_len + 1, clip_stride):
        end = start + clip_len - 1
        frame_agents = risk_data.get(format_time_key(end, interval_seconds), {})
        selected_agents = []
        score_counts = Counter()
        for agent_id, agent_data in frame_agents.items():
            score = agent_score(agent_data)
            if score is None:
                continue
            score_counts[str(score)] += 1
            if score <= max_risk_score:
                selected_agents.append(
                    {
                        "agent_id": agent_id,
                        "risk_score": score,
                        "risk_level": agent_data.get("Risk Analysis", {}).get("Risk Level"),
                        "relative_direction": agent_data.get("Relative Direction"),
                        "distance_to_collision": agent_data.get("Distance to Collision"),
                    }
                )
        if len(selected_agents) < min_target_agents:
            continue
        selected_agents.sort(key=lambda row: (row["risk_score"], row["agent_id"]))
        rows.append(
            {
                "scene_name": scene_name,
                "scenario_prefix": scenario_prefix(scene_name),
                "clip_id": f"{scene_name}_{start:04d}_{end:04d}",
                "start_frame_index": start,
                "end_frame_index": end,
                "clip_len": clip_len,
                "num_high_risk_agents": len(selected_agents),
                "min_risk_score": selected_agents[0]["risk_score"],
                "risk_score_counts": dict(score_counts),
                "target_agents": selected_agents,
                "selection_reason": f"reference frame has >= {min_target_agents} agent(s) with risk_score <= {max_risk_score}",
            }
        )
    return rows


def balanced_sample(
    candidates: Sequence[Dict[str, Any]],
    target_vqa_samples: int,
    estimated_samples_per_clip: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if target_vqa_samples <= 0:
        return list(candidates)
    target_clips = max(1, int(math.ceil(target_vqa_samples / max(estimated_samples_per_clip, 0.1))))
    if len(candidates) <= target_clips:
        return list(candidates)

    rng = random.Random(seed)
    by_prefix: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_prefix[row["scenario_prefix"]].append(row)
    for rows in by_prefix.values():
        rows.sort(key=lambda row: (row["min_risk_score"], -row["num_high_risk_agents"], row["clip_id"]))
        top = rows[: max(1, len(rows) // 2)]
        rest = rows[max(1, len(rows) // 2) :]
        rng.shuffle(top)
        rng.shuffle(rest)
        rows[:] = top + rest

    selected = []
    prefixes = sorted(by_prefix)
    while len(selected) < target_clips and prefixes:
        remaining = []
        for prefix in prefixes:
            rows = by_prefix[prefix]
            if rows:
                selected.append(rows.pop(0))
                if len(selected) >= target_clips:
                    break
            if rows:
                remaining.append(prefix)
        prefixes = remaining
    selected.sort(key=lambda row: (row["scene_name"], row["start_frame_index"]))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None, help="Default: <dataroot>/nurisk_style")
    parser.add_argument("--output-dir", default=None, help="Default: <input-dir>/high_risk_selection")
    parser.add_argument("--scenario-prefixes", default="default", help="default, all, or comma-separated prefixes")
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--max-risk-score", type=int, default=2)
    parser.add_argument("--min-target-agents", type=int, default=1)
    parser.add_argument("--target-vqa-samples", type=int, default=10000)
    parser.add_argument("--estimated-samples-per-clip", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keyframe-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between Bench2Drive frames. Default is 0.5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = output_root(args.dataroot, args.input_dir)
    outdir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "high_risk_selection"
    prefixes = parse_prefixes(args.scenario_prefixes)

    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir() and (not prefixes or scenario_prefix(path.name) in prefixes))
    candidates = []
    for scene_dir in scene_dirs:
        candidates.extend(
            risk_candidates_for_scene(
                scene_dir=scene_dir,
                clip_len=args.clip_len,
                clip_stride=args.clip_stride,
                interval_seconds=args.keyframe_interval_seconds,
                max_risk_score=args.max_risk_score,
                min_target_agents=args.min_target_agents,
            )
        )

    selected = balanced_sample(
        candidates,
        target_vqa_samples=args.target_vqa_samples,
        estimated_samples_per_clip=args.estimated_samples_per_clip,
        seed=args.seed,
    )

    manifest_path = outdir / "selected_clips.jsonl"
    stats_path = outdir / "selection_stats.json"
    write_jsonl(manifest_path, selected)
    stats = {
        "source": str(root),
        "scenario_prefixes": sorted(prefixes) if prefixes else "all",
        "scene_dirs_considered": len(scene_dirs),
        "candidate_clips": len(candidates),
        "selected_clips": len(selected),
        "target_vqa_samples": args.target_vqa_samples,
        "estimated_samples_per_clip": args.estimated_samples_per_clip,
        "estimated_vqa_samples": int(round(len(selected) * args.estimated_samples_per_clip)),
        "clip_len": args.clip_len,
        "clip_stride": args.clip_stride,
        "max_risk_score": args.max_risk_score,
        "min_target_agents": args.min_target_agents,
        "selected_by_prefix": dict(Counter(row["scenario_prefix"] for row in selected)),
        "selected_by_min_risk_score": dict(Counter(str(row["min_risk_score"]) for row in selected)),
        "manifest": str(manifest_path),
    }
    write_json(stats_path, stats)
    print("Stage 4b done.")
    print(f"Scene dirs considered: {len(scene_dirs)}")
    print(f"Candidate clips: {len(candidates)}")
    print(f"Selected clips: {len(selected)}")
    print(f"Estimated VQA samples: {stats['estimated_vqa_samples']}")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
