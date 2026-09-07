#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run Stage 5a and Stage 5b."""

import argparse
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_DATAROOT, DEFAULT_KEYFRAME_INTERVAL_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene-names", default=None)
    parser.add_argument("--clip-dir", default=None, help="Default: <dataroot>/annotated_video_clip_dataset")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument(
        "--keyframe-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between nuScenes keyframes. Default is 0.5 for 2Hz keyframes.",
    )
    parser.add_argument("--overwrite-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    stage5a = [
        sys.executable,
        str(script_dir / "stage5a_build_video_clips.py"),
        "--dataroot",
        args.dataroot,
        "--version",
        args.version,
        "--channels",
        args.channels,
        "--fps",
        str(args.fps),
        "--clip-len",
        str(args.clip_len),
        "--clip-stride",
        str(args.clip_stride),
    ]
    if args.scene_names:
        stage5a += ["--scene-names", args.scene_names]
    if args.clip_dir:
        stage5a += ["--outdir", args.clip_dir]
    if args.max_scenes is not None:
        stage5a += ["--max-scenes", str(args.max_scenes)]
    if args.overwrite_videos:
        stage5a += ["--overwrite-videos"]

    stage5b = [
        sys.executable,
        str(script_dir / "stage5b_align_video_clip_groundtruth.py"),
        "--dataroot",
        args.dataroot,
        "--keyframe-interval-seconds",
        str(args.keyframe_interval_seconds),
    ]
    if args.scene_names and "," not in args.scene_names:
        stage5b += ["--scene-name", args.scene_names]
    if args.clip_dir:
        stage5b += ["--clip-dir", args.clip_dir]

    print("Running:", " ".join(stage5a), flush=True)
    subprocess.run(stage5a, check=True)
    print("Running:", " ".join(stage5b), flush=True)
    subprocess.run(stage5b, check=True)


if __name__ == "__main__":
    main()
