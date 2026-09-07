#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 1: extract NuRisk-style trajectories from Waymo v2 parquet data.

The downstream Stage 2-6 scripts operate on the same CSV schema used by the
nuScenes pipeline. This adapter reads Waymo Perception v2 modular parquet
components and writes:

    <dataroot>/nurisk_style/<segment>/ego_trajectory.csv
    <dataroot>/nurisk_style/<segment>/dynamic_obstacles.csv

Only sampled keyframes are exported. The default 0.5 second interval matches the
2Hz nuScenes keyframe cadence used by the existing pipeline.
"""

import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

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

WAYMO_TYPE_NAMES = {
    0: "unknown",
    1: "vehicle",
    2: "pedestrian",
    3: "sign",
    4: "cyclist",
}


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Waymo component parquet: {path}")
    return pd.read_parquet(path)


def matrix4(values: Any) -> List[float]:
    if values is None:
        raise ValueError("Missing 4x4 transform")
    data = list(values)
    if len(data) != 16:
        raise ValueError(f"Expected 16 transform values, got {len(data)}")
    return [float(v) for v in data]


def transform_point(transform: Sequence[float], x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
    return (
        transform[0] * x + transform[1] * y + transform[2] * z + transform[3],
        transform[4] * x + transform[5] * y + transform[6] * z + transform[7],
        transform[8] * x + transform[9] * y + transform[10] * z + transform[11],
    )


def transform_vector(transform: Sequence[float], x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
    return (
        transform[0] * x + transform[1] * y + transform[2] * z,
        transform[4] * x + transform[5] * y + transform[6] * z,
        transform[8] * x + transform[9] * y + transform[10] * z,
    )


def yaw_from_world_from_vehicle(transform: Sequence[float]) -> float:
    return math.atan2(transform[4], transform[0])


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def segment_name_from_path(path: Path) -> str:
    return path.stem


def list_segments(component_dir: Path) -> List[str]:
    return sorted(path.stem for path in component_dir.glob("*.parquet"))


def select_keyframe_timestamps(
    timestamps: Sequence[int],
    interval_seconds: float,
    max_keyframes: Optional[int] = None,
) -> List[int]:
    ordered = sorted({int(ts) for ts in timestamps})
    if not ordered:
        return []

    interval_us = int(round(interval_seconds * 1_000_000))
    tolerance_us = max(1, interval_us // 2)
    selected: List[int] = []
    target = ordered[0]
    last = ordered[-1]
    while target <= last + tolerance_us:
        idx = bisect_left(ordered, target)
        candidates = []
        if idx < len(ordered):
            candidates.append(ordered[idx])
        if idx > 0:
            candidates.append(ordered[idx - 1])
        if not candidates:
            break
        nearest = min(candidates, key=lambda ts: abs(ts - target))
        if abs(nearest - target) <= tolerance_us and (not selected or selected[-1] != nearest):
            selected.append(nearest)
            if max_keyframes is not None and len(selected) >= max_keyframes:
                break
        target += interval_us
    return selected


def fill_motion(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    speeds: List[float] = []
    for index, row in enumerate(rows):
        if len(rows) == 1:
            speeds.append(0.0)
            continue
        if 0 < index < len(rows) - 1:
            before, after = rows[index - 1], rows[index + 1]
        elif index == 0:
            before, after = rows[index], rows[index + 1]
        else:
            before, after = rows[index - 1], rows[index]
        dt = (float(after["timestamp"]) - float(before["timestamp"])) / 1_000_000.0
        distance = math.hypot(
            float(after["x_position"]) - float(before["x_position"]),
            float(after["y_position"]) - float(before["y_position"]),
        )
        speeds.append(distance / dt if dt > 0 else 0.0)

    for index, row in enumerate(rows):
        row["velocity"] = round(speeds[index], 6)
        if len(rows) <= 2 or index == 0 or index == len(rows) - 1:
            row["acceleration"] = 0.0
            continue
        before, after = rows[index - 1], rows[index + 1]
        dt = (float(after["timestamp"]) - float(before["timestamp"])) / 1_000_000.0
        row["acceleration"] = round((speeds[index + 1] - speeds[index - 1]) / dt, 6) if dt > 0 else 0.0


def build_ego_rows(
    scene_name: str,
    vehicle_pose: pd.DataFrame,
    sampled_timestamps: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[int, Sequence[float]]]:
    pose_by_timestamp = {
        int(row["key.frame_timestamp_micros"]): matrix4(row["[VehiclePoseComponent].world_from_vehicle.transform"])
        for _, row in vehicle_pose.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    sampled_transforms: Dict[int, Sequence[float]] = {}
    for timestep, timestamp in enumerate(sampled_timestamps):
        transform = pose_by_timestamp[timestamp]
        sampled_transforms[timestamp] = transform
        rows.append(
            {
                "timestep": timestep,
                "x_position": transform[3],
                "y_position": transform[7],
                "orientation": yaw_from_world_from_vehicle(transform),
                "velocity": 0.0,
                "acceleration": 0.0,
                "scene_name": scene_name,
                "sample_token": f"{scene_name};{timestamp}",
                "timestamp": timestamp,
                "ego_pose_token": f"{scene_name};vehicle_pose;{timestamp}",
            }
        )
    fill_motion(rows)
    return rows, sampled_transforms


def infer_obstacle_heading(
    transform: Sequence[float],
    row: pd.Series,
    vehicle_yaw: float,
) -> Tuple[float, float]:
    vx_vehicle = safe_float(row.get("[LiDARBoxComponent].speed.x"))
    vy_vehicle = safe_float(row.get("[LiDARBoxComponent].speed.y"))
    vx_world, vy_world, _ = transform_vector(transform, vx_vehicle, vy_vehicle, 0.0)
    speed = math.hypot(vx_world, vy_world)
    if speed > 0.2:
        return math.atan2(vy_world, vx_world), speed
    heading_vehicle = safe_float(row.get("[LiDARBoxComponent].box.heading"))
    return normalize_angle(vehicle_yaw + heading_vehicle), speed


def obstacle_acceleration(transform: Sequence[float], row: pd.Series, heading: float) -> float:
    ax_vehicle = safe_float(row.get("[LiDARBoxComponent].acceleration.x"))
    ay_vehicle = safe_float(row.get("[LiDARBoxComponent].acceleration.y"))
    ax_world, ay_world, _ = transform_vector(transform, ax_vehicle, ay_vehicle, 0.0)
    return ax_world * math.cos(heading) + ay_world * math.sin(heading)


def build_obstacle_rows(
    scene_name: str,
    lidar_box: pd.DataFrame,
    sampled_timestamps: Sequence[int],
    sampled_transforms: Dict[int, Sequence[float]],
) -> List[Dict[str, Any]]:
    sampled_set = set(int(ts) for ts in sampled_timestamps)
    timestep_by_timestamp = {int(ts): idx for idx, ts in enumerate(sampled_timestamps)}
    rows: List[Dict[str, Any]] = []

    for _, row in lidar_box.iterrows():
        timestamp = int(row["key.frame_timestamp_micros"])
        if timestamp not in sampled_set:
            continue
        object_id = str(row["key.laser_object_id"])
        transform = sampled_transforms[timestamp]
        vehicle_yaw = yaw_from_world_from_vehicle(transform)
        x_vehicle = safe_float(row.get("[LiDARBoxComponent].box.center.x"))
        y_vehicle = safe_float(row.get("[LiDARBoxComponent].box.center.y"))
        z_vehicle = safe_float(row.get("[LiDARBoxComponent].box.center.z"))
        x_world, y_world, _ = transform_point(transform, x_vehicle, y_vehicle, z_vehicle)
        heading, speed = infer_obstacle_heading(transform, row, vehicle_yaw)
        obj_type = int(safe_float(row.get("[LiDARBoxComponent].type"), 0.0))
        timestep = timestep_by_timestamp[timestamp]
        rows.append(
            {
                "timestep": timestep,
                "obstacle_id": object_id,
                "x_position": x_world,
                "y_position": y_world,
                "orientation": heading,
                "velocity": speed,
                "acceleration": obstacle_acceleration(transform, row, heading),
                "scene_name": scene_name,
                "sample_token": f"{scene_name};{timestamp}",
                "timestamp": timestamp,
                "annotation_token": f"{scene_name};{timestamp};{object_id}",
                "category_name": WAYMO_TYPE_NAMES.get(obj_type, "unknown"),
            }
        )
    rows.sort(key=lambda item: (int(item["timestep"]), str(item["obstacle_id"])))
    return rows


def write_rows(path: Path, fields: Iterable[str], rows: Sequence[Dict[str, Any]]) -> None:
    f, writer = open_writer(path, fields)
    try:
        writer.writerows(rows)
    finally:
        f.close()


def process_segment(
    dataroot: Path,
    split: str,
    segment_name: str,
    output_dir: Path,
    interval_seconds: float,
    overwrite: bool,
    max_keyframes: Optional[int],
) -> Tuple[int, int]:
    scene_dir = output_dir / segment_name
    outputs = [scene_dir / "ego_trajectory.csv", scene_dir / "dynamic_obstacles.csv"]
    require_absent(outputs, overwrite)
    mkdir(scene_dir)

    vehicle_pose = read_parquet(dataroot / split / "vehicle_pose" / f"{segment_name}.parquet")
    lidar_box = read_parquet(dataroot / split / "lidar_box" / f"{segment_name}.parquet")
    sampled_timestamps = select_keyframe_timestamps(
        vehicle_pose["key.frame_timestamp_micros"].tolist(),
        interval_seconds=interval_seconds,
        max_keyframes=max_keyframes,
    )
    if not sampled_timestamps:
        raise ValueError(f"No sampled timestamps selected for segment {segment_name}")

    ego_rows, sampled_transforms = build_ego_rows(segment_name, vehicle_pose, sampled_timestamps)
    obstacle_rows = build_obstacle_rows(segment_name, lidar_box, sampled_timestamps, sampled_transforms)
    write_rows(scene_dir / "ego_trajectory.csv", EGO_FIELDS, ego_rows)
    write_rows(scene_dir / "dynamic_obstacles.csv", DYNAMIC_OBSTACLE_FIELDS, obstacle_rows)

    with open(scene_dir / "waymo_stage1_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "segment_name": segment_name,
                "split": split,
                "source": "waymo_open_dataset_v_2_0_1",
                "sample_interval_seconds": interval_seconds,
                "sampled_timestamps": sampled_timestamps,
                "num_ego_rows": len(ego_rows),
                "num_obstacle_rows": len(obstacle_rows),
                "coordinate_frame": "world",
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
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--segment-name", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between sampled Waymo frames. Default is 0.5 for 2Hz.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    root = output_root(str(dataroot), args.output_dir)
    lidar_box_dir = dataroot / args.split / "lidar_box"
    vehicle_pose_dir = dataroot / args.split / "vehicle_pose"
    if not lidar_box_dir.exists() or not vehicle_pose_dir.exists():
        raise FileNotFoundError(
            f"Missing Waymo components under {dataroot / args.split}; "
            "expected lidar_box/ and vehicle_pose/."
        )

    if args.segment_name:
        selected_segments = [args.segment_name]
    else:
        lidar_segments = set(list_segments(lidar_box_dir))
        pose_segments = set(list_segments(vehicle_pose_dir))
        selected_segments = sorted(lidar_segments & pose_segments)
    if args.max_scenes is not None:
        selected_segments = selected_segments[: args.max_scenes]
    if not selected_segments:
        raise ValueError("No Waymo segments selected.")

    if args.segment_name is None and root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {root}. Use --overwrite.")
    mkdir(root)

    ego_count = 0
    obstacle_count = 0
    for segment_name in selected_segments:
        scene_ego_count, scene_obstacle_count = process_segment(
            dataroot=dataroot,
            split=args.split,
            segment_name=segment_name,
            output_dir=root,
            interval_seconds=args.sample_interval_seconds,
            overwrite=args.overwrite,
            max_keyframes=args.max_keyframes,
        )
        ego_count += scene_ego_count
        obstacle_count += scene_obstacle_count

    print(f"Waymo Stage 1 done: {ego_count} ego rows, {obstacle_count} obstacle rows -> {root}")


if __name__ == "__main__":
    main()
