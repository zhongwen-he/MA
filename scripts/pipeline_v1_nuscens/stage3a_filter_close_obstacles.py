#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 3a: filter NuRisk-style close dynamic obstacles."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Tuple

from common import (
    DEFAULT_DATAROOT,
    DEFAULT_LENGTH_RANGE,
    DEFAULT_WIDTH_RANGE,
    calculate_relative_distances,
    ensure_csv_field_size,
    is_close_agent,
    load_ego_by_timestep,
    open_writer,
    output_root,
    parse_float,
    read_scene_dirs,
    require_absent,
    round_value,
)


CLOSE_OBSTACLE_FIELDS = [
    "timestep",
    "obstacle_id",
    "x_position",
    "y_position",
    "orientation",
    "velocity",
    "acceleration",
    "longitudinal_distance",
    "lateral_distance",
    "scene_name",
    "sample_token",
    "timestamp",
    "annotation_token",
    "category_name",
]


def build_close_row(
    ego: Dict[str, str],
    obs: Dict[str, str],
    d_long: float,
    d_lat: float,
    digits: int,
) -> Dict[str, Any]:
    return {
        **obs,
        "longitudinal_distance": round_value(d_long, digits),
        "lateral_distance": round_value(d_lat, digits),
    }


def process_scene(
    scene_dir: Path,
    length_range: Tuple[float, float],
    width_range: Tuple[float, float],
    digits: int,
    overwrite: bool,
) -> int:
    ego_path = scene_dir / "ego_trajectory.csv"
    obstacles_path = scene_dir / "dynamic_obstacles.csv"
    output_path = scene_dir / "close_dynamic_obstacles.csv"
    if not ego_path.exists() or not obstacles_path.exists():
        print(f"Skipping {scene_dir.name}: missing Stage 1 files")
        return 0
    require_absent([output_path], overwrite)

    ego_by_timestep = load_ego_by_timestep(ego_path)
    count = 0
    out_f, writer = open_writer(output_path, CLOSE_OBSTACLE_FIELDS)
    try:
        with open(obstacles_path, "r", encoding="utf-8", newline="") as f:
            for obs in csv.DictReader(f):
                ego = ego_by_timestep.get(obs.get("timestep", ""))
                if ego is None:
                    continue
                d_long, d_lat = calculate_relative_distances(
                    parse_float(ego.get("x_position")),
                    parse_float(ego.get("y_position")),
                    parse_float(ego.get("orientation")),
                    parse_float(obs.get("x_position")),
                    parse_float(obs.get("y_position")),
                )
                if not is_close_agent(d_long, d_lat, length_range, width_range):
                    continue
                writer.writerow(build_close_row(ego, obs, d_long, d_lat, digits))
                count += 1
    finally:
        out_f.close()
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--length-min", type=float, default=DEFAULT_LENGTH_RANGE[0])
    parser.add_argument("--length-max", type=float, default=DEFAULT_LENGTH_RANGE[1])
    parser.add_argument("--width-min", type=float, default=DEFAULT_WIDTH_RANGE[0])
    parser.add_argument("--width-max", type=float, default=DEFAULT_WIDTH_RANGE[1])
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_csv_field_size()
    root = output_root(args.dataroot, args.input_dir)
    length_range = (args.length_min, args.length_max)
    width_range = (args.width_min, args.width_max)
    total = 0
    for scene_dir in read_scene_dirs(root, args.scene_name):
        total += process_scene(scene_dir, length_range, width_range, args.digits, args.overwrite)
    print(f"Stage 3a done: {total} close obstacle rows -> {root}")


if __name__ == "__main__":
    main()
