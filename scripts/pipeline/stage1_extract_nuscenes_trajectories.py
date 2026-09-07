#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 1: extract NuRisk-style trajectories directly from raw nuScenes data.

This stage reads the nuScenes metadata under `<dataroot>/<version>` and follows
each scene's `sample` chain. A nuScenes `sample` is the keyframe-level record;
sweeps are not used.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common import (
    DEFAULT_DATAROOT,
    ensure_csv_field_size,
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


def load_table(version_dir: Path, name: str) -> List[Dict[str, Any]]:
    path = version_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing nuScenes metadata file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def by_token(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["token"]: row for row in rows}


def quaternion_yaw(rotation: List[float]) -> float:
    w, x, y, z = rotation
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def seconds(timestamp_us: Any) -> float:
    return float(timestamp_us) / 1_000_000.0


def distance_xy(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax, ay = a["translation"][0], a["translation"][1]
    bx, by = b["translation"][0], b["translation"][1]
    return math.hypot(bx - ax, by - ay)


def scalar_speed(
    current: Dict[str, Any],
    token_to_record: Dict[str, Dict[str, Any]],
    sample_by_token: Dict[str, Dict[str, Any]],
) -> float:
    prev_token = current.get("prev", "")
    next_token = current.get("next", "")
    if prev_token and next_token:
        prev_record = token_to_record[prev_token]
        next_record = token_to_record[next_token]
        dt = seconds(sample_by_token[next_record["sample_token"]]["timestamp"]) - seconds(
            sample_by_token[prev_record["sample_token"]]["timestamp"]
        )
        return distance_xy(prev_record, next_record) / dt if dt > 0 else 0.0
    if prev_token:
        prev_record = token_to_record[prev_token]
        dt = seconds(current["timestamp"]) - seconds(sample_by_token[prev_record["sample_token"]]["timestamp"])
        return distance_xy(prev_record, current) / dt if dt > 0 else 0.0
    if next_token:
        next_record = token_to_record[next_token]
        dt = seconds(sample_by_token[next_record["sample_token"]]["timestamp"]) - seconds(current["timestamp"])
        return distance_xy(current, next_record) / dt if dt > 0 else 0.0
    return 0.0


def annotation_speed(
    ann: Dict[str, Any],
    ann_by_token: Dict[str, Dict[str, Any]],
    sample_by_token: Dict[str, Dict[str, Any]],
) -> float:
    prev_token = ann.get("prev", "")
    next_token = ann.get("next", "")
    current_timestamp = sample_by_token[ann["sample_token"]]["timestamp"]
    if prev_token and next_token:
        prev_ann = ann_by_token[prev_token]
        next_ann = ann_by_token[next_token]
        dt = seconds(sample_by_token[next_ann["sample_token"]]["timestamp"]) - seconds(
            sample_by_token[prev_ann["sample_token"]]["timestamp"]
        )
        return distance_xy(prev_ann, next_ann) / dt if dt > 0 else 0.0
    if prev_token:
        prev_ann = ann_by_token[prev_token]
        dt = seconds(current_timestamp) - seconds(sample_by_token[prev_ann["sample_token"]]["timestamp"])
        return distance_xy(prev_ann, ann) / dt if dt > 0 else 0.0
    if next_token:
        next_ann = ann_by_token[next_token]
        dt = seconds(sample_by_token[next_ann["sample_token"]]["timestamp"]) - seconds(current_timestamp)
        return distance_xy(ann, next_ann) / dt if dt > 0 else 0.0
    return 0.0


def annotation_acceleration(
    ann: Dict[str, Any],
    ann_by_token: Dict[str, Dict[str, Any]],
    sample_by_token: Dict[str, Dict[str, Any]],
) -> float:
    prev_token = ann.get("prev", "")
    next_token = ann.get("next", "")
    if prev_token and next_token:
        prev_ann = ann_by_token[prev_token]
        next_ann = ann_by_token[next_token]
        prev_speed = annotation_speed(prev_ann, ann_by_token, sample_by_token)
        next_speed = annotation_speed(next_ann, ann_by_token, sample_by_token)
        dt = seconds(sample_by_token[next_ann["sample_token"]]["timestamp"]) - seconds(
            sample_by_token[prev_ann["sample_token"]]["timestamp"]
        )
        return (next_speed - prev_speed) / dt if dt > 0 else 0.0
    return 0.0


def scene_samples(scene: Dict[str, Any], sample_by_token: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = sample_by_token[token]
        samples.append(sample)
        if token == scene["last_sample_token"]:
            break
        token = sample["next"]
    return samples


def choose_keyframe_sample_data(
    sample_token: str,
    sample_data_by_sample_channel: Dict[Tuple[str, str], Dict[str, Any]],
    sample_data_by_sample: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    preferred = sample_data_by_sample_channel.get((sample_token, "CAM_FRONT"))
    if preferred and preferred.get("is_key_frame") and preferred.get("filename", "").startswith("samples/"):
        return preferred
    for row in sample_data_by_sample.get(sample_token, []):
        if row.get("is_key_frame") and row.get("filename", "").startswith("samples/"):
            return row
    raise ValueError(f"No keyframe sample_data under samples/ for sample {sample_token}")


def build_ego_rows(
    scene_name: str,
    samples: List[Dict[str, Any]],
    ego_pose_by_token: Dict[str, Dict[str, Any]],
    sample_data_by_sample_channel: Dict[Tuple[str, str], Dict[str, Any]],
    sample_data_by_sample: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows = []
    for timestep, sample in enumerate(samples):
        sample_data = choose_keyframe_sample_data(
            sample["token"], sample_data_by_sample_channel, sample_data_by_sample
        )
        ego_pose = ego_pose_by_token[sample_data["ego_pose_token"]]
        rows.append(
            {
                "timestep": timestep,
                "x_position": ego_pose["translation"][0],
                "y_position": ego_pose["translation"][1],
                "orientation": quaternion_yaw(ego_pose["rotation"]),
                "velocity": 0.0,
                "acceleration": 0.0,
                "scene_name": scene_name,
                "sample_token": sample["token"],
                "timestamp": sample["timestamp"],
                "ego_pose_token": sample_data["ego_pose_token"],
                "_translation": ego_pose["translation"],
            }
        )
    fill_ego_motion(rows)
    for row in rows:
        row.pop("_translation", None)
    return rows


def fill_ego_motion(rows: List[Dict[str, Any]]) -> None:
    speeds = []
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
        dt = seconds(after["timestamp"]) - seconds(before["timestamp"])
        dist = math.hypot(
            after["_translation"][0] - before["_translation"][0],
            after["_translation"][1] - before["_translation"][1],
        )
        speeds.append(dist / dt if dt > 0 else 0.0)

    for index, row in enumerate(rows):
        row["velocity"] = speeds[index]
        if len(rows) <= 2:
            row["acceleration"] = 0.0
        elif 0 < index < len(rows) - 1:
            before, after = rows[index - 1], rows[index + 1]
            dt = seconds(after["timestamp"]) - seconds(before["timestamp"])
            row["acceleration"] = (speeds[index + 1] - speeds[index - 1]) / dt if dt > 0 else 0.0
        else:
            row["acceleration"] = 0.0


def build_obstacle_rows(
    scene_name: str,
    samples: List[Dict[str, Any]],
    annotations_by_sample: Dict[str, List[Dict[str, Any]]],
    ann_by_token: Dict[str, Dict[str, Any]],
    sample_by_token: Dict[str, Dict[str, Any]],
    instance_by_token: Dict[str, Dict[str, Any]],
    category_by_token: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for timestep, sample in enumerate(samples):
        for ann in annotations_by_sample.get(sample["token"], []):
            instance = instance_by_token.get(ann["instance_token"], {})
            category = category_by_token.get(instance.get("category_token", ""), {})
            rows.append(
                {
                    "timestep": timestep,
                    "obstacle_id": ann["instance_token"],
                    "x_position": ann["translation"][0],
                    "y_position": ann["translation"][1],
                    "orientation": quaternion_yaw(ann["rotation"]),
                    "velocity": annotation_speed(ann, ann_by_token, sample_by_token),
                    "acceleration": annotation_acceleration(ann, ann_by_token, sample_by_token),
                    "scene_name": scene_name,
                    "sample_token": sample["token"],
                    "timestamp": sample["timestamp"],
                    "annotation_token": ann["token"],
                    "category_name": ann.get("category_name") or category.get("name", ""),
                }
            )
    return rows


def write_rows(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    f, writer = open_writer(path, fields)
    try:
        writer.writerows(rows)
    finally:
        f.close()


def build_sample_data_indexes(
    sample_data: List[Dict[str, Any]],
    calibrated_sensor_by_token: Dict[str, Dict[str, Any]],
    sensor_by_token: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[str, str], Dict[str, Any]]]:
    by_sample: Dict[str, List[Dict[str, Any]]] = {}
    by_sample_channel: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in sample_data:
        by_sample.setdefault(row["sample_token"], []).append(row)
        calibrated = calibrated_sensor_by_token.get(row["calibrated_sensor_token"], {})
        sensor = sensor_by_token.get(calibrated.get("sensor_token", ""), {})
        channel = sensor.get("channel")
        if channel:
            by_sample_channel[(row["sample_token"], channel)] = row
    return by_sample, by_sample_channel


def process_scene(
    scene: Dict[str, Any],
    output_dir: Path,
    indexes: Dict[str, Any],
    overwrite: bool,
) -> Tuple[int, int]:
    scene_name = scene["name"]
    scene_dir = output_dir / scene_name
    outputs = [scene_dir / "ego_trajectory.csv", scene_dir / "dynamic_obstacles.csv"]
    require_absent(outputs, overwrite)
    mkdir(scene_dir)

    samples = scene_samples(scene, indexes["sample_by_token"])
    ego_rows = build_ego_rows(
        scene_name,
        samples,
        indexes["ego_pose_by_token"],
        indexes["sample_data_by_sample_channel"],
        indexes["sample_data_by_sample"],
    )
    obstacle_rows = build_obstacle_rows(
        scene_name,
        samples,
        indexes["annotations_by_sample"],
        indexes["ann_by_token"],
        indexes["sample_by_token"],
        indexes["instance_by_token"],
        indexes["category_by_token"],
    )
    write_rows(scene_dir / "ego_trajectory.csv", EGO_FIELDS, ego_rows)
    write_rows(scene_dir / "dynamic_obstacles.csv", DYNAMIC_OBSTACLE_FIELDS, obstacle_rows)
    return len(ego_rows), len(obstacle_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_csv_field_size()
    dataroot = Path(args.dataroot).expanduser().resolve()
    version_dir = dataroot / args.version
    root = output_root(str(dataroot), args.output_dir)

    scenes = load_table(version_dir, "scene")
    samples = load_table(version_dir, "sample")
    sample_data = load_table(version_dir, "sample_data")
    ego_pose = load_table(version_dir, "ego_pose")
    sample_annotation = load_table(version_dir, "sample_annotation")
    instance = load_table(version_dir, "instance")
    category = load_table(version_dir, "category")
    calibrated_sensor = load_table(version_dir, "calibrated_sensor")
    sensor = load_table(version_dir, "sensor")

    sample_by_token = by_token(samples)
    ann_by_token = by_token(sample_annotation)
    annotations_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for ann in sample_annotation:
        annotations_by_sample.setdefault(ann["sample_token"], []).append(ann)

    sample_data_by_sample, sample_data_by_sample_channel = build_sample_data_indexes(
        sample_data, by_token(calibrated_sensor), by_token(sensor)
    )

    selected_scenes = [scene for scene in scenes if args.scene_name is None or scene["name"] == args.scene_name]
    if args.max_scenes is not None:
        selected_scenes = selected_scenes[: args.max_scenes]
    if not selected_scenes:
        raise ValueError("No scenes selected. Check --scene-name or --max-scenes.")

    if args.scene_name is None and root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {root}. Use --overwrite.")
    mkdir(root)

    indexes = {
        "sample_by_token": sample_by_token,
        "ego_pose_by_token": by_token(ego_pose),
        "sample_data_by_sample": sample_data_by_sample,
        "sample_data_by_sample_channel": sample_data_by_sample_channel,
        "annotations_by_sample": annotations_by_sample,
        "ann_by_token": ann_by_token,
        "instance_by_token": by_token(instance),
        "category_by_token": by_token(category),
    }

    ego_count = 0
    obstacle_count = 0
    for scene in selected_scenes:
        scene_ego_count, scene_obstacle_count = process_scene(scene, root, indexes, args.overwrite)
        ego_count += scene_ego_count
        obstacle_count += scene_obstacle_count

    print(f"Stage 1 done: {ego_count} ego rows, {obstacle_count} obstacle rows -> {root}")


if __name__ == "__main__":
    main()
