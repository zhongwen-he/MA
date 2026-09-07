#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 2: compute NuRisk-style full relative_metrics.csv files."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from common import (
    DEFAULT_DATAROOT,
    DEFAULT_EGO_LENGTH,
    DEFAULT_EGO_WIDTH,
    DEFAULT_KEYFRAME_INTERVAL_SECONDS,
    calculate_adjusted_relative_distances,
    calculate_relative_acceleration,
    calculate_relative_distances,
    calculate_relative_velocity,
    calculate_time_to_collision,
    ensure_csv_field_size,
    format_time_key,
    identify_relative_direction,
    load_ego_by_timestep,
    open_writer,
    output_root,
    parse_float,
    read_scene_dirs,
    require_absent,
    round_value,
)


RELATIVE_FIELDS = [
    "timestep",
    "obstacle_id",
    "relative_direction",
    "d_long",
    "d_lat",
    "adjusted_d_long",
    "adjusted_d_lat",
    "v_rel_long",
    "v_rel_lat",
    "a_rel_long",
    "a_rel_lat",
    "ttc_long",
    "ttc_lat",
    "motion_description",
    "scene_name",
    "sample_token",
    "timestamp",
    "annotation_token",
    "category_name",
]


def build_relative_row(
    ego: Dict[str, str],
    obs: Dict[str, str],
    ego_length: float,
    ego_width: float,
    digits: int,
) -> Dict[str, Any]:
    x_ego = parse_float(ego.get("x_position"))
    y_ego = parse_float(ego.get("y_position"))
    theta_ego = parse_float(ego.get("orientation"))
    v_ego = parse_float(ego.get("velocity"))
    a_ego = parse_float(ego.get("acceleration"))

    x_obs = parse_float(obs.get("x_position"))
    y_obs = parse_float(obs.get("y_position"))
    theta_obs = parse_float(obs.get("orientation"))
    v_obs = parse_float(obs.get("velocity"))
    a_obs = parse_float(obs.get("acceleration"))

    d_long, d_lat = calculate_relative_distances(x_ego, y_ego, theta_ego, x_obs, y_obs)
    adjusted_d_long, adjusted_d_lat = calculate_adjusted_relative_distances(
        d_long, d_lat, ego_length, ego_width
    )
    v_rel_long, v_rel_lat = calculate_relative_velocity(v_ego, theta_ego, v_obs, theta_obs)
    a_rel_long, a_rel_lat = calculate_relative_acceleration(a_ego, theta_ego, a_obs, theta_obs)
    relative_direction = identify_relative_direction(d_long, d_lat, ego_length, ego_width)
    ttc_long, ttc_lat, motion_description = calculate_time_to_collision(
        adjusted_d_long, adjusted_d_lat, v_rel_long, v_rel_lat, relative_direction
    )

    return {
        "timestep": obs.get("timestep", ""),
        "obstacle_id": obs.get("obstacle_id", ""),
        "relative_direction": relative_direction,
        "d_long": round_value(d_long, digits),
        "d_lat": round_value(d_lat, digits),
        "adjusted_d_long": round_value(adjusted_d_long, digits),
        "adjusted_d_lat": round_value(adjusted_d_lat, digits),
        "v_rel_long": round_value(v_rel_long, digits),
        "v_rel_lat": round_value(v_rel_lat, digits),
        "a_rel_long": round_value(a_rel_long, digits),
        "a_rel_lat": round_value(a_rel_lat, digits),
        "ttc_long": round_value(ttc_long, digits),
        "ttc_lat": round_value(ttc_lat, digits),
        "motion_description": motion_description,
        "scene_name": obs.get("scene_name", ""),
        "sample_token": obs.get("sample_token", ""),
        "timestamp": obs.get("timestamp", ""),
        "annotation_token": obs.get("annotation_token", ""),
        "category_name": obs.get("category_name", ""),
    }


def process_scene(
    scene_dir: Path,
    ego_length: float,
    ego_width: float,
    digits: int,
    overwrite: bool,
    keyframe_interval_seconds: float,
) -> int:
    ego_path = scene_dir / "ego_trajectory.csv"
    obstacles_path = scene_dir / "dynamic_obstacles.csv"
    output_path = scene_dir / "relative_metrics.csv"
    json_path = scene_dir / "output.json"
    if not ego_path.exists() or not obstacles_path.exists():
        print(f"Skipping {scene_dir.name}: missing Stage 1 files")
        return 0
    require_absent([output_path, json_path], overwrite)

    ego_by_timestep = load_ego_by_timestep(ego_path)
    output_json: Dict[str, Dict[str, Dict[str, Any]]] = {}
    count = 0
    out_f, writer = open_writer(output_path, RELATIVE_FIELDS)
    try:
        with open(obstacles_path, "r", encoding="utf-8", newline="") as f:
            for obs in csv.DictReader(f):
                ego = ego_by_timestep.get(obs.get("timestep", ""))
                if ego is None:
                    continue
                relative = build_relative_row(ego, obs, ego_length, ego_width, digits)
                writer.writerow(relative)
                add_output_json_row(output_json, relative, keyframe_interval_seconds)
                count += 1
    finally:
        out_f.close()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
        f.write("\n")
    return count


def json_number(value: Any) -> Any:
    if value == "inf":
        return "Infinity"
    if value == "-inf":
        return "-Infinity"
    return float(value)


def add_output_json_row(
    output_json: Dict[str, Dict[str, Dict[str, Any]]],
    row: Dict[str, Any],
    keyframe_interval_seconds: float,
) -> None:
    time_key = format_time_key(row["timestep"], keyframe_interval_seconds)
    obstacle_key = f"Obstacle {row['obstacle_id']}"
    output_json.setdefault(time_key, {})[obstacle_key] = {
        "Relative Direction": row["relative_direction"],
        "Distance to Collision": {
            "Longitudinal": abs(float(row["adjusted_d_long"])),
            "Lateral": abs(float(row["adjusted_d_lat"])),
        },
        "Time to Collision": {
            "Longitudinal": json_number(row["ttc_long"]),
            "Lateral": json_number(row["ttc_lat"]),
        },
        "Motion Description": row["motion_description"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--ego-length", type=float, default=DEFAULT_EGO_LENGTH)
    parser.add_argument("--ego-width", type=float, default=DEFAULT_EGO_WIDTH)
    parser.add_argument(
        "--keyframe-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between Bench2Drive frames. Default is 0.5 for 2Hz keyframes.",
    )
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_csv_field_size()
    root = output_root(args.dataroot, args.input_dir)
    total = 0
    for scene_dir in read_scene_dirs(root, args.scene_name):
        total += process_scene(
            scene_dir,
            args.ego_length,
            args.ego_width,
            args.digits,
            args.overwrite,
            args.keyframe_interval_seconds,
        )
    print(f"Stage 2 done: {total} relative metric rows -> {root}")


if __name__ == "__main__":
    main()
