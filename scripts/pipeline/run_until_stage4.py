#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the NuRisk-style nuScenes workflow through Stage 4."""

import argparse
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_DATAROOT, DEFAULT_KEYFRAME_INTERVAL_SECONDS, output_root


STAGES = [
    "stage1_extract_nuscenes_trajectories.py",
    "stage2_compute_relative_metrics.py",
    "stage3a_filter_close_obstacles.py",
    "stage3b_extract_close_relative_metrics.py",
    "stage4_compute_risk_scores_enhanced.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--keyframe-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between nuScenes keyframes. Default is 0.5 for 2Hz keyframes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = output_root(args.dataroot, args.output_dir)

    for stage in STAGES:
        cmd = [
            sys.executable,
            str(script_dir / stage),
            "--dataroot",
            args.dataroot,
        ]
        if args.scene_name:
            cmd += ["--scene-name", args.scene_name]
        if args.max_scenes is not None and stage == STAGES[0]:
            cmd += ["--max-scenes", str(args.max_scenes)]
        if args.overwrite:
            cmd += ["--overwrite"]
        if stage == STAGES[0]:
            cmd += ["--version", args.version]
            if args.output_dir:
                cmd += ["--output-dir", args.output_dir]
        else:
            cmd += ["--input-dir", str(root)]
        if stage in {
            "stage2_compute_relative_metrics.py",
            "stage4_compute_risk_scores_enhanced.py",
        }:
            cmd += ["--keyframe-interval-seconds", str(args.keyframe_interval_seconds)]

        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
