#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the Waymo NuRisk-style workflow through Stage 4."""

import argparse
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_DATAROOT, DEFAULT_KEYFRAME_INTERVAL_SECONDS, output_root


STAGES = [
    "stage1_extract_waymo_trajectories.py",
    "stage2_compute_relative_metrics.py",
    "stage3a_filter_close_obstacles.py",
    "stage3b_extract_close_relative_metrics.py",
    "stage4_compute_risk_scores_enhanced.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--segment-name", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument(
        "--keyframe-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between sampled Waymo frames. Default is 0.5 for 2Hz.",
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
        if args.overwrite:
            cmd += ["--overwrite"]
        if stage == STAGES[0]:
            cmd += ["--split", args.split]
            if args.output_dir:
                cmd += ["--output-dir", args.output_dir]
            if args.segment_name:
                cmd += ["--segment-name", args.segment_name]
            if args.max_scenes is not None:
                cmd += ["--max-scenes", str(args.max_scenes)]
            if args.max_keyframes is not None:
                cmd += ["--max-keyframes", str(args.max_keyframes)]
            cmd += ["--sample-interval-seconds", str(args.keyframe_interval_seconds)]
        else:
            cmd += ["--input-dir", str(root)]
            if args.segment_name:
                cmd += ["--scene-name", args.segment_name]
        if stage in {
            "stage2_compute_relative_metrics.py",
            "stage4_compute_risk_scores_enhanced.py",
        }:
            cmd += ["--keyframe-interval-seconds", str(args.keyframe_interval_seconds)]

        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
