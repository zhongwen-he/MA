#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for the NuRisk-style Waymo preprocessing scripts."""

import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


DEFAULT_DATAROOT = "/home/dellpro2/zhongwen/nuscenes-devkit/data/sets/waymo_test"
DEFAULT_OUTPUT_SUBDIR = "nurisk_style"
DEFAULT_EGO_LENGTH = 4.508
DEFAULT_EGO_WIDTH = 1.610
DEFAULT_LENGTH_RANGE = (-30.0, 40.0)
DEFAULT_WIDTH_RANGE = (-30.0, 30.0)
DEFAULT_KEYFRAME_INTERVAL_SECONDS = 0.5


def output_root(dataroot: str, output_dir: Optional[str]) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return Path(dataroot).expanduser().resolve() / DEFAULT_OUTPUT_SUBDIR


def ensure_csv_field_size() -> None:
    csv.field_size_limit(sys.maxsize)


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def format_time_key(
    frame_index: Any,
    keyframe_interval_seconds: float = DEFAULT_KEYFRAME_INTERVAL_SECONDS,
) -> str:
    seconds = parse_float(frame_index) * keyframe_interval_seconds
    return f"At {seconds:.1f} seconds"


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def round_value(value: float, digits: int) -> Any:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    rounded = round(value, digits)
    return 0.0 if rounded == -0.0 else rounded


def require_absent(paths: Sequence[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Output already exists: {existing}. Use --overwrite to replace.")


def open_writer(path: Path, fields: Iterable[str]) -> Tuple[Any, csv.DictWriter]:
    mkdir(path.parent)
    f = open(path, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    return f, writer


def read_scene_dirs(root: Path, scene_name: Optional[str]) -> Sequence[Path]:
    if scene_name:
        scene_dir = root / scene_name
        if not scene_dir.exists():
            raise FileNotFoundError(f"Missing scene directory: {scene_dir}")
        return [scene_dir]
    return sorted(path for path in root.iterdir() if path.is_dir())


def calculate_relative_distances(
    x_ego: float,
    y_ego: float,
    theta_ego: float,
    x_obs: float,
    y_obs: float,
) -> Tuple[float, float]:
    delta_x = x_obs - x_ego
    delta_y = y_obs - y_ego
    d_long = delta_x * math.cos(-theta_ego) - delta_y * math.sin(-theta_ego)
    d_lat = delta_x * math.sin(-theta_ego) + delta_y * math.cos(-theta_ego)
    return d_long, d_lat


def calculate_adjusted_relative_distances(
    d_long: float,
    d_lat: float,
    ego_length: float = DEFAULT_EGO_LENGTH,
    ego_width: float = DEFAULT_EGO_WIDTH,
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


def calculate_relative_velocity(
    v_ego: float,
    theta_ego: float,
    v_obs: float,
    theta_obs: float,
) -> Tuple[float, float]:
    v_obs_long = v_obs * math.cos(theta_obs - theta_ego)
    v_obs_lat = v_obs * math.sin(theta_obs - theta_ego)
    v_rel_long = v_obs_long - v_ego
    v_rel_lat = v_obs_lat
    return v_rel_long, v_rel_lat


def calculate_relative_acceleration(
    a_ego: float,
    theta_ego: float,
    a_obs: float,
    theta_obs: float,
) -> Tuple[float, float]:
    a_obs_long = a_obs * math.cos(theta_obs)
    a_obs_lat = a_obs * math.sin(theta_obs)
    a_rel_long = a_obs_long - a_ego * math.cos(theta_ego)
    a_rel_lat = a_obs_lat - a_ego * math.sin(theta_ego)
    return a_rel_long, a_rel_lat


def identify_relative_direction(
    d_long: float,
    d_lat: float,
    ego_length: float = DEFAULT_EGO_LENGTH,
    ego_width: float = DEFAULT_EGO_WIDTH,
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


def is_close_agent(
    d_long: float,
    d_lat: float,
    length_range: Tuple[float, float] = DEFAULT_LENGTH_RANGE,
    width_range: Tuple[float, float] = DEFAULT_WIDTH_RANGE,
) -> bool:
    return (
        length_range[0] <= d_long <= length_range[1]
        and width_range[0] <= d_lat <= width_range[1]
    )


def make_key(timestep: Any, obstacle_id: Any) -> str:
    return f"{timestep}::{obstacle_id}"


def load_close_keys(path: Path) -> set:
    keys = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            keys.add(make_key(row["timestep"], row["obstacle_id"]))
    return keys


def load_ego_by_timestep(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {row["timestep"]: row for row in csv.DictReader(f)}
