#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter close agents and export NuRisk-style close relative metrics for nuScenes.

This script mirrors NuRisk's two-step close-agent preprocessing:
  1. Keep obstacles inside a rectangular ego-centric window.
  2. Export the relative physical metrics only for those close obstacles.

Input:
  data/sets/nuscenes_full/physical_csv/relative_agent_states.csv
  data/sets/nuscenes_full/physical_csv/scenes_physical.csv

Output:
  data/sets/nuscenes_full/physical_csv/close_dynamic_obstacles.csv
  data/sets/nuscenes_full/physical_csv/close_relative_metrics.csv
  data/sets/nuscenes_full/physical_csv/close_agent_schema.json
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


DEFAULT_DATAROOT = "/home/dellpro2/zhongwen/nuscenes-devkit/data/sets/nuscenes_full"
DEFAULT_EGO_LENGTH = 4.508
DEFAULT_EGO_WIDTH = 1.610

CLOSE_DYNAMIC_FIELDS = [
    "scene_name",
    "scene_token",
    "split",
    "frame_index",
    "sample_token",
    "obstacle_id",
    "annotation_token",
    "category_name",
    "longitudinal_distance",
    "lateral_distance",
    "relative_distance_m",
    "bearing_rad",
    "length_range_min",
    "length_range_max",
    "width_range_min",
    "width_range_max",
]

CLOSE_METRIC_FIELDS = [
    "scene_name",
    "scene_token",
    "split",
    "frame_index",
    "sample_token",
    "obstacle_id",
    "annotation_token",
    "category_name",
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
    "relative_distance_m",
    "bearing_rad",
]


def parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def parse_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def apply_zero_epsilon(value: Optional[float], zero_eps: float) -> Optional[float]:
    if value is None:
        return None
    if abs(value) < zero_eps:
        return 0.0
    return value


def round_value(value: Optional[float], digits: int) -> Any:
    if value is None:
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    rounded = round(value, digits)
    return 0.0 if rounded == -0.0 else rounded


def load_scene_splits(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["scene_name"]: row.get("split", "") for row in reader}


def is_close_agent(
    d_long: Optional[float],
    d_lat: Optional[float],
    length_range: Tuple[float, float],
    width_range: Tuple[float, float],
) -> bool:
    if d_long is None or d_lat is None:
        return False
    return (
        length_range[0] <= d_long <= length_range[1]
        and width_range[0] <= d_lat <= width_range[1]
    )


def calculate_adjusted_relative_distances(
    d_long: float,
    d_lat: float,
    ego_length: float,
    ego_width: float,
) -> Tuple[float, float]:
    adjusted_d_long = (
        d_long - ego_length
        if d_long > ego_length
        else (d_long + ego_length if d_long < -ego_length else 0.0)
    )
    adjusted_d_lat = (
        d_lat - ego_width
        if d_lat > ego_width
        else (d_lat + ego_width if d_lat < -ego_width else 0.0)
    )
    return adjusted_d_long, adjusted_d_lat


def identify_relative_direction(
    d_long: float,
    d_lat: float,
    ego_length: float,
    ego_width: float,
) -> str:
    if d_long > ego_length:
        if d_lat > ego_width:
            return "Front-left"
        if d_lat < -ego_width:
            return "Front-right"
        return "Front"
    if d_long < -ego_length:
        if d_lat > ego_width:
            return "Rear-left"
        if d_lat < -ego_width:
            return "Rear-right"
        return "Behind"

    if d_lat > ego_width:
        return "Left"
    if d_lat < -ego_width:
        return "Right"

    return "Collision"


def calculate_time_to_collision(
    adjusted_d_long: float,
    adjusted_d_lat: float,
    v_rel_long: float,
    v_rel_lat: float,
    relative_direction: str,
) -> Tuple[float, float, str]:
    motion_description = ""

    if relative_direction in ["Front", "Front-left", "Front-right"]:
        if v_rel_long > 0:
            ttc_long = float("inf")
            motion_description = "Obstacle is moving away longitudinally."
        elif v_rel_long < 0:
            ttc_long = adjusted_d_long / abs(v_rel_long)
            motion_description = "Obstacle is driving toward the ego car longitudinally."
        else:
            ttc_long = float("inf")
            motion_description = "No longitudinal relative motion."
    elif relative_direction in ["Behind", "Rear-left", "Rear-right"]:
        if v_rel_long > 0:
            ttc_long = abs(adjusted_d_long) / v_rel_long
            motion_description = "Obstacle is driving toward the ego car from behind."
        elif v_rel_long < 0:
            ttc_long = float("inf")
            motion_description = "Obstacle is moving away longitudinally."
        else:
            ttc_long = float("inf")
            motion_description = "No longitudinal relative motion."
    else:
        ttc_long = 0.0
        motion_description = "Exact longitudinal alignment or co."

    if relative_direction in ["Left", "Front-left", "Rear-left"]:
        if v_rel_lat > 0:
            ttc_lat = float("inf")
            motion_description += " Obstacle is moving away laterally to the left."
        elif v_rel_lat < 0:
            ttc_lat = abs(adjusted_d_lat / v_rel_lat)
            motion_description += " Obstacle is driving toward the ego car laterally from the left."
        else:
            ttc_lat = float("inf")
            motion_description += " No lateral relative motion."
    elif relative_direction in ["Right", "Front-right", "Rear-right"]:
        if v_rel_lat > 0:
            ttc_lat = abs(adjusted_d_lat / v_rel_lat)
            motion_description += " Obstacle is driving toward the ego car laterally from the right."
        elif v_rel_lat < 0:
            ttc_lat = float("inf")
            motion_description += " Obstacle is moving away laterally to the right."
        else:
            ttc_lat = float("inf")
            motion_description += " No lateral relative motion."
    else:
        ttc_lat = 0.0
        motion_description += " Exact lateral alignment or unknown case."

    return ttc_long, ttc_lat, motion_description


def open_writer(path: Path, fieldnames: Iterable[str]) -> Tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    return f, writer


def build_rows(
    row: Dict[str, str],
    split: str,
    length_range: Tuple[float, float],
    width_range: Tuple[float, float],
    ego_length: float,
    ego_width: float,
    digits: int,
    zero_eps: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    d_long = parse_float(row.get("longitudinal_distance_m"))
    d_lat = parse_float(row.get("lateral_distance_m"))
    if d_long is None:
        d_long = parse_float(row.get("relative_x_forward_m"))
    if d_lat is None:
        d_lat = parse_float(row.get("relative_y_left_m"))

    v_rel_long = apply_zero_epsilon(parse_float(row.get("relative_vx_forward_mps")) or 0.0, zero_eps)
    v_rel_lat = apply_zero_epsilon(parse_float(row.get("relative_vy_left_mps")) or 0.0, zero_eps)
    a_rel_long = apply_zero_epsilon(parse_float(row.get("relative_ax_forward_mps2")) or 0.0, zero_eps)
    a_rel_lat = apply_zero_epsilon(parse_float(row.get("relative_ay_left_mps2")) or 0.0, zero_eps)
    distance = parse_float(row.get("relative_distance_m"))
    bearing = parse_float(row.get("bearing_rad"))

    adjusted_d_long, adjusted_d_lat = calculate_adjusted_relative_distances(
        d_long, d_lat, ego_length, ego_width
    )
    relative_direction = identify_relative_direction(d_long, d_lat, ego_length, ego_width)
    ttc_long, ttc_lat, motion_description = calculate_time_to_collision(
        adjusted_d_long, adjusted_d_lat, v_rel_long, v_rel_lat, relative_direction
    )

    common = {
        "scene_name": row.get("scene_name", ""),
        "scene_token": row.get("scene_token", ""),
        "split": split,
        "frame_index": parse_int(row.get("frame_index")),
        "sample_token": row.get("sample_token", ""),
        "obstacle_id": row.get("agent_instance_token", ""),
        "annotation_token": row.get("annotation_token", ""),
        "category_name": row.get("category_name", ""),
    }

    close_row = {
        **common,
        "longitudinal_distance": round_value(d_long, digits),
        "lateral_distance": round_value(d_lat, digits),
        "relative_distance_m": round_value(distance, digits),
        "bearing_rad": round_value(bearing, 6),
        "length_range_min": length_range[0],
        "length_range_max": length_range[1],
        "width_range_min": width_range[0],
        "width_range_max": width_range[1],
    }

    metric_row = {
        **common,
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
        "relative_distance_m": round_value(distance, digits),
        "bearing_rad": round_value(bearing, 6),
    }
    return close_row, metric_row


def write_schema(
    path: Path,
    relative_csv: Path,
    close_dynamic_csv: Path,
    close_metrics_csv: Path,
    length_range: Tuple[float, float],
    width_range: Tuple[float, float],
    ego_length: float,
    ego_width: float,
    zero_eps: float,
    counts: Dict[str, int],
) -> None:
    schema = {
        "description": "Close-agent physical CSV generated with NuRisk-style rectangular filtering and relative metrics.",
        "inputs": {
            "relative_agent_states_csv": str(relative_csv),
        },
        "outputs": {
            "close_dynamic_obstacles_csv": str(close_dynamic_csv),
            "close_relative_metrics_csv": str(close_metrics_csv),
        },
        "filter": {
            "coordinate_frame": "ego-centric",
            "longitudinal_axis": "x_forward_m",
            "lateral_axis": "y_left_m",
            "length_range_m": list(length_range),
            "width_range_m": list(width_range),
        },
        "ego_footprint_for_adjusted_distance": {
            "length_m": ego_length,
            "width_m": ego_width,
        },
        "numeric_tolerance": {
            "zero_eps_for_relative_velocity_and_acceleration": zero_eps,
        },
        "counts": counts,
        "close_dynamic_obstacles_fields": CLOSE_DYNAMIC_FIELDS,
        "close_relative_metrics_fields": CLOSE_METRIC_FIELDS,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter close nuScenes agents and export NuRisk-style physical CSV."
    )
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--physical-csv-dir", default=None)
    parser.add_argument("--relative-csv", default=None)
    parser.add_argument("--scene-name", default=None, help="Optional single scene_name filter.")
    parser.add_argument("--split", default=None, choices=["train", "val"], help="Optional split filter.")
    parser.add_argument("--length-min", type=float, default=-30.0)
    parser.add_argument("--length-max", type=float, default=40.0)
    parser.add_argument("--width-min", type=float, default=-30.0)
    parser.add_argument("--width-max", type=float, default=30.0)
    parser.add_argument("--ego-length", type=float, default=DEFAULT_EGO_LENGTH)
    parser.add_argument("--ego-width", type=float, default=DEFAULT_EGO_WIDTH)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument(
        "--zero-eps",
        type=float,
        default=1e-6,
        help="Treat smaller absolute relative velocity/acceleration values as zero.",
    )
    parser.add_argument("--max-output-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot)
    physical_csv_dir = Path(args.physical_csv_dir) if args.physical_csv_dir else dataroot / "physical_csv"
    relative_csv = Path(args.relative_csv) if args.relative_csv else physical_csv_dir / "relative_agent_states.csv"
    scenes_csv = physical_csv_dir / "scenes_physical.csv"
    close_dynamic_csv = physical_csv_dir / "close_dynamic_obstacles.csv"
    close_metrics_csv = physical_csv_dir / "close_relative_metrics.csv"
    schema_path = physical_csv_dir / "close_agent_schema.json"

    if not relative_csv.exists():
        raise FileNotFoundError(f"Missing relative CSV: {relative_csv}")
    if not args.overwrite:
        existing = [p for p in [close_dynamic_csv, close_metrics_csv, schema_path] if p.exists()]
        if existing:
            paths = ", ".join(str(p) for p in existing)
            raise FileExistsError(f"Output already exists: {paths}. Use --overwrite to replace.")

    length_range = (args.length_min, args.length_max)
    width_range = (args.width_min, args.width_max)
    scene_splits = load_scene_splits(scenes_csv)

    total_rows = 0
    close_rows = 0
    scenes_seen = set()
    close_scenes = set()

    csv.field_size_limit(sys.maxsize)
    close_f, close_writer = open_writer(close_dynamic_csv, CLOSE_DYNAMIC_FIELDS)
    metric_f, metric_writer = open_writer(close_metrics_csv, CLOSE_METRIC_FIELDS)
    try:
        with open(relative_csv, "r", encoding="utf-8", newline="") as in_f:
            reader = csv.DictReader(in_f)
            for row in tqdm(reader, desc="Filtering close agents", unit="rows"):
                scene_name = row.get("scene_name", "")
                split = scene_splits.get(scene_name, "")
                if args.scene_name and scene_name != args.scene_name:
                    continue
                if args.split and split != args.split:
                    continue

                total_rows += 1
                scenes_seen.add(scene_name)
                d_long = parse_float(row.get("longitudinal_distance_m"))
                d_lat = parse_float(row.get("lateral_distance_m"))
                if d_long is None:
                    d_long = parse_float(row.get("relative_x_forward_m"))
                if d_lat is None:
                    d_lat = parse_float(row.get("relative_y_left_m"))
                if not is_close_agent(d_long, d_lat, length_range, width_range):
                    continue

                close_row, metric_row = build_rows(
                    row,
                    split,
                    length_range,
                    width_range,
                    args.ego_length,
                    args.ego_width,
                    args.digits,
                    args.zero_eps,
                )
                close_writer.writerow(close_row)
                metric_writer.writerow(metric_row)
                close_rows += 1
                close_scenes.add(scene_name)

                if args.max_output_rows is not None and close_rows >= args.max_output_rows:
                    break
    finally:
        close_f.close()
        metric_f.close()

    counts = {
        "candidate_rows_read_after_filters": total_rows,
        "close_rows_written": close_rows,
        "scenes_seen_after_filters": len(scenes_seen),
        "scenes_with_close_agents": len(close_scenes),
    }
    write_schema(
        schema_path,
        relative_csv,
        close_dynamic_csv,
        close_metrics_csv,
        length_range,
        width_range,
        args.ego_length,
        args.ego_width,
        args.zero_eps,
        counts,
    )

    print("Done.")
    print(f"Read rows after scene/split filters: {total_rows}")
    print(f"Close rows written: {close_rows}")
    print(f"Close dynamic obstacles: {close_dynamic_csv}")
    print(f"Close relative metrics: {close_metrics_csv}")
    print(f"Schema: {schema_path}")


if __name__ == "__main__":
    main()
