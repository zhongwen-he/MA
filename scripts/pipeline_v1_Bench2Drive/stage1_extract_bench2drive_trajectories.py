#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 1: extract NuRisk-style trajectories from Bench2Drive frame annotations.

Bench2Drive stores one compressed JSON annotation per frame under:

    <dataroot>/raw_camera_anno/<scene>/anno/*.json.gz

This adapter exports the same CSV schema used by the nuScenes and Waymo
pipelines so Stage 2-8 can be reused.
"""

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import (
    DEFAULT_DATAROOT,
    DEFAULT_KEYFRAME_INTERVAL_SECONDS,
    mkdir,
    open_writer,
    output_root,
    require_absent,
)


EGO_FIELDS = [
    "timestep",
    "x_position",
    "y_position",
    "orientation",
    "velocity",
    "acceleration",
    "scene_name",
    "sample_token",
    "timestamp",
    "ego_pose_token",
]

DYNAMIC_OBSTACLE_FIELDS = [
    "timestep",
    "obstacle_id",
    "x_position",
    "y_position",
    "orientation",
    "velocity",
    "acceleration",
    "scene_name",
    "sample_token",
    "timestamp",
    "annotation_token",
    "category_name",
]

CATEGORY_MAP = {
    "vehicle": "vehicle.car",
    "ego_vehicle": "ego_vehicle",
    "walker": "pedestrian",
    "pedestrian": "pedestrian",
    "traffic_light": "traffic_light",
    "traffic_sign": "traffic_sign",
}


def load_frame(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def frame_index(path: Path) -> int:
    return int(path.name.split(".")[0])


def timestamp_for_frame(index: int, interval_seconds: float) -> int:
    return int(round(index * interval_seconds * 1_000_000))


def bench_yaw_to_pipeline_radians(rotation: Any) -> float:
    if not isinstance(rotation, Sequence) or len(rotation) < 3:
        return 0.0
    return math.radians(safe_float(rotation[2])) + math.pi / 2.0


def category_name(box: Dict[str, Any]) -> str:
    raw_class = str(box.get("class") or "object").strip()
    if raw_class == "vehicle":
        base_type = str(box.get("base_type") or "").strip().lower()
        type_id = str(box.get("type_id") or "").strip().lower()
        if "motorcycle" in type_id or base_type == "motorcycle":
            return "vehicle.motorcycle"
        if "bike" in type_id or "bicycle" in type_id or base_type == "bicycle":
            return "vehicle.bicycle"
        if "truck" in type_id or base_type == "truck":
            return "vehicle.truck"
        if "bus" in type_id or base_type == "bus":
            return "vehicle.bus"
    return CATEGORY_MAP.get(raw_class, raw_class.lower())


def build_ego_rows(
    scene_name: str,
    frames: Sequence[Tuple[Path, Dict[str, Any]]],
    interval_seconds: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for timestep, (path, data) in enumerate(frames):
        frame_id = frame_index(path)
        timestamp = timestamp_for_frame(timestep, interval_seconds)
        rows.append(
            {
                "timestep": timestep,
                "x_position": safe_float(data.get("x")),
                "y_position": safe_float(data.get("y")),
                "orientation": safe_float(data.get("theta")),
                "velocity": safe_float(data.get("speed")),
                "acceleration": 0.0,
                "scene_name": scene_name,
                "sample_token": f"{scene_name};{frame_id:05d}",
                "timestamp": timestamp,
                "ego_pose_token": f"{scene_name};ego_pose;{frame_id:05d}",
            }
        )

    for index, row in enumerate(rows):
        if len(rows) <= 2 or index == 0 or index == len(rows) - 1:
            row["acceleration"] = 0.0
            continue
        before, after = rows[index - 1], rows[index + 1]
        dt = (float(after["timestamp"]) - float(before["timestamp"])) / 1_000_000.0
        row["acceleration"] = (
            (safe_float(after["velocity"]) - safe_float(before["velocity"])) / dt
            if dt > 0
            else 0.0
        )
    return rows


def build_obstacle_rows(
    scene_name: str,
    frames: Sequence[Tuple[Path, Dict[str, Any]]],
    interval_seconds: float,
    include_static: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for timestep, (path, data) in enumerate(frames):
        frame_id = frame_index(path)
        timestamp = timestamp_for_frame(timestep, interval_seconds)
        sample_token = f"{scene_name};{frame_id:05d}"
        for box in data.get("bounding_boxes", []):
            if box.get("class") == "ego_vehicle":
                continue
            if not include_static and box.get("state") == "static":
                continue
            obstacle_id = str(box.get("id") or "")
            location = box.get("location") or box.get("center") or []
            if not obstacle_id or len(location) < 2:
                continue
            rows.append(
                {
                    "timestep": timestep,
                    "obstacle_id": obstacle_id,
                    "x_position": safe_float(location[0]),
                    "y_position": safe_float(location[1]),
                    "orientation": bench_yaw_to_pipeline_radians(box.get("rotation")),
                    "velocity": safe_float(box.get("speed")),
                    "acceleration": 0.0,
                    "scene_name": scene_name,
                    "sample_token": sample_token,
                    "timestamp": timestamp,
                    "annotation_token": f"{scene_name};{frame_id:05d};{obstacle_id}",
                    "category_name": category_name(box),
                }
            )
    fill_obstacle_motion(rows, interval_seconds)
    rows.sort(key=lambda item: (int(item["timestep"]), str(item["obstacle_id"])))
    return rows


def fill_obstacle_motion(rows: List[Dict[str, Any]], interval_seconds: float) -> None:
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["obstacle_id"])].append(row)

    for track in by_id.values():
        track.sort(key=lambda item: int(item["timestep"]))
        for index, row in enumerate(track):
            if len(track) == 1:
                continue
            if 0 < index < len(track) - 1:
                before, after = track[index - 1], track[index + 1]
            elif index == 0:
                before, after = track[index], track[index + 1]
            else:
                before, after = track[index - 1], track[index]
            dt = (int(after["timestep"]) - int(before["timestep"])) * interval_seconds
            if dt <= 0:
                continue
            dx = safe_float(after["x_position"]) - safe_float(before["x_position"])
            dy = safe_float(after["y_position"]) - safe_float(before["y_position"])
            estimated_speed = math.hypot(dx, dy) / dt
            if safe_float(row.get("velocity")) <= 0.01 and estimated_speed > 0.01:
                row["velocity"] = estimated_speed
            if estimated_speed > 0.2:
                row["orientation"] = math.atan2(dy, dx)

        speeds = [safe_float(row.get("velocity")) for row in track]
        for index, row in enumerate(track):
            if len(track) <= 2 or index == 0 or index == len(track) - 1:
                row["acceleration"] = 0.0
                continue
            before, after = track[index - 1], track[index + 1]
            dt = (int(after["timestep"]) - int(before["timestep"])) * interval_seconds
            row["acceleration"] = (speeds[index + 1] - speeds[index - 1]) / dt if dt > 0 else 0.0


def write_rows(path: Path, fields: Iterable[str], rows: Sequence[Dict[str, Any]]) -> None:
    f, writer = open_writer(path, fields)
    try:
        writer.writerows(rows)
    finally:
        f.close()


def list_scenes(raw_root: Path) -> List[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing Bench2Drive raw directory: {raw_root}")
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def process_scene(
    scene_path: Path,
    output_dir: Path,
    interval_seconds: float,
    overwrite: bool,
    max_keyframes: Optional[int],
    include_static: bool,
) -> Tuple[int, int]:
    scene_name = scene_path.name
    anno_dir = scene_path / "anno"
    frame_paths = sorted(anno_dir.glob("*.json.gz"), key=frame_index)
    if max_keyframes is not None:
        frame_paths = frame_paths[:max_keyframes]
    if not frame_paths:
        raise ValueError(f"No Bench2Drive annotation frames found in {anno_dir}")

    scene_dir = output_dir / scene_name
    outputs = [scene_dir / "ego_trajectory.csv", scene_dir / "dynamic_obstacles.csv"]
    require_absent(outputs, overwrite)
    mkdir(scene_dir)

    frames = [(path, load_frame(path)) for path in frame_paths]
    ego_rows = build_ego_rows(scene_name, frames, interval_seconds)
    obstacle_rows = build_obstacle_rows(scene_name, frames, interval_seconds, include_static)
    write_rows(scene_dir / "ego_trajectory.csv", EGO_FIELDS, ego_rows)
    write_rows(scene_dir / "dynamic_obstacles.csv", DYNAMIC_OBSTACLE_FIELDS, obstacle_rows)

    with open(scene_dir / "bench2drive_stage1_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene_name": scene_name,
                "source": "Bench2Drive-V0.0.4",
                "raw_scene_path": str(scene_path),
                "sample_interval_seconds": interval_seconds,
                "num_ego_rows": len(ego_rows),
                "num_obstacle_rows": len(obstacle_rows),
                "include_static": include_static,
                "coordinate_frame": "Bench2Drive/CARLA world",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    return len(ego_rows), len(obstacle_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--raw-subdir", default="raw_camera_anno")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between exported Bench2Drive frames. Default is 0.5.",
    )
    parser.add_argument(
        "--dynamic-only",
        action="store_true",
        help="Drop static actors from dynamic_obstacles.csv. Default keeps all non-ego annotated agents.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    raw_root = dataroot / args.raw_subdir
    root = output_root(str(dataroot), args.output_dir)

    if args.scene_name:
        selected_scenes = [raw_root / args.scene_name]
    else:
        selected_scenes = list_scenes(raw_root)
    if args.max_scenes is not None:
        selected_scenes = selected_scenes[: args.max_scenes]
    if not selected_scenes:
        raise ValueError("No Bench2Drive scenes selected.")

    if args.scene_name is None and root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {root}. Use --overwrite.")
    mkdir(root)

    ego_count = 0
    obstacle_count = 0
    for scene_path in selected_scenes:
        scene_ego_count, scene_obstacle_count = process_scene(
            scene_path=scene_path,
            output_dir=root,
            interval_seconds=args.sample_interval_seconds,
            overwrite=args.overwrite,
            max_keyframes=args.max_keyframes,
            include_static=not args.dynamic_only,
        )
        ego_count += scene_ego_count
        obstacle_count += scene_obstacle_count

    print(f"Bench2Drive Stage 1 done: {ego_count} ego rows, {obstacle_count} obstacle rows -> {root}")


if __name__ == "__main__":
    main()
