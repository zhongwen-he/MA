#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 3b: extract relative metrics for close dynamic obstacles."""

import argparse
import csv
from pathlib import Path

from common import (
    DEFAULT_DATAROOT,
    ensure_csv_field_size,
    load_close_keys,
    make_key,
    open_writer,
    output_root,
    read_scene_dirs,
    require_absent,
)
from stage2_compute_relative_metrics import RELATIVE_FIELDS


def process_scene(scene_dir: Path, overwrite: bool) -> int:
    close_path = scene_dir / "close_dynamic_obstacles.csv"
    relative_path = scene_dir / "relative_metrics.csv"
    output_path = scene_dir / "close_relative_metrics.csv"
    if not close_path.exists() or not relative_path.exists():
        print(f"Skipping {scene_dir.name}: missing Stage 2 or Stage 3a files")
        return 0
    require_absent([output_path], overwrite)

    close_keys = load_close_keys(close_path)
    count = 0
    out_f, writer = open_writer(output_path, RELATIVE_FIELDS)
    try:
        with open(relative_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if make_key(row["timestep"], row["obstacle_id"]) not in close_keys:
                    continue
                writer.writerow(row)
                count += 1
    finally:
        out_f.close()
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_csv_field_size()
    root = output_root(args.dataroot, args.input_dir)
    total = 0
    for scene_dir in read_scene_dirs(root, args.scene_name):
        total += process_scene(scene_dir, args.overwrite)
    print(f"Stage 3b done: {total} close relative metric rows -> {root}")


if __name__ == "__main__":
    main()
