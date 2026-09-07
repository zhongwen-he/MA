#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Repair corrupt Waymo Stage 5a video clips in place.

The normal Stage 5a script skips existing mp4 files when --overwrite-videos is
not set, but it cannot distinguish a valid mp4 from a corrupt one. This repair
script scans existing clips, quarantines only clips that fail decoding, and then
calls Stage 5a process_segment for the affected segments. Existing good clips
remain untouched and global metadata/split index files are not rewritten.
"""

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from common import DEFAULT_DATAROOT, DEFAULT_KEYFRAME_INTERVAL_SECONDS, mkdir
from stage5a_build_waymo_video_clips import (
    DEFAULT_CLIP_SUBDIR,
    parse_channels,
    process_segment,
)


@dataclass(frozen=True)
class VideoStatus:
    path: str
    ok: bool
    error: str


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ffmpeg_check(path: str, timeout_seconds: int) -> VideoStatus:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return VideoStatus(path=path, ok=False, error=f"timeout after {timeout_seconds}s: {exc}")
    except Exception as exc:
        return VideoStatus(path=path, ok=False, error=f"{type(exc).__name__}: {exc}")

    if proc.returncode == 0:
        return VideoStatus(path=path, ok=True, error="")
    error = " ".join(proc.stderr.strip().split())
    return VideoStatus(path=path, ok=False, error=error[:1000])


def iter_video_paths(videos_root: Path, segments: Optional[Sequence[str]]) -> List[Path]:
    if segments:
        paths: List[Path] = []
        for segment in segments:
            paths.extend((videos_root / segment).rglob("*.mp4"))
        return sorted(paths)
    return sorted(videos_root.rglob("*.mp4"))


def path_metadata(path: Path, videos_root: Path) -> Tuple[str, str]:
    parts = path.relative_to(videos_root).parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected Waymo video path under {videos_root}: {path}")
    return parts[0], parts[1]


def scan_bad_videos(
    videos_root: Path,
    report_path: Path,
    workers: int,
    timeout_seconds: int,
    segments: Optional[Sequence[str]],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    paths = iter_video_paths(videos_root, segments)
    if limit is not None:
        paths = paths[:limit]
    print(f"Scanning videos: {len(paths)}", flush=True)

    bad_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(ffmpeg_check, str(path), timeout_seconds) for path in paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking mp4 decode"):
            status = future.result()
            if status.ok:
                continue
            path = Path(status.path)
            segment, channel = path_metadata(path, videos_root)
            bad_rows.append(
                {
                    "path": status.path,
                    "segment": segment,
                    "channel": channel,
                    "error": status.error,
                }
            )

    bad_rows.sort(key=lambda row: row["path"])
    write_jsonl(report_path, bad_rows)
    return bad_rows


def quarantine_bad_videos(
    bad_rows: Sequence[Dict[str, Any]],
    clip_root: Path,
    quarantine_root: Path,
    dry_run: bool,
) -> List[Path]:
    moved_paths: List[Path] = []
    for row in tqdm(bad_rows, desc="Quarantining bad mp4"):
        src = Path(row["path"])
        if not src.exists():
            continue
        rel = src.relative_to(clip_root)
        dst = quarantine_root / rel
        moved_paths.append(src)
        if dry_run:
            continue
        mkdir(dst.parent)
        shutil.move(str(src), str(dst))
    return moved_paths


def repair_segments(
    dataroot: Path,
    split: str,
    clip_root: Path,
    segments: Sequence[str],
    channels: Sequence[str],
    fps: float,
    clip_len: int,
    clip_stride: int,
    interval_seconds: float,
    max_keyframes: Optional[int],
    dry_run: bool,
) -> None:
    for segment in tqdm(segments, desc="Repairing affected segments"):
        if dry_run:
            continue
        process_segment(
            dataroot=dataroot,
            split=split,
            segment_name=segment,
            outdir=clip_root,
            channels=channels,
            fps=fps,
            clip_len=clip_len,
            clip_stride=clip_stride,
            interval_seconds=interval_seconds,
            overwrite_videos=False,
            max_keyframes=max_keyframes,
        )


def verify_paths(paths: Sequence[Path], workers: int, timeout_seconds: int) -> List[Dict[str, Any]]:
    if not paths:
        return []
    bad_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(ffmpeg_check, str(path), timeout_seconds) for path in paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying repaired mp4"):
            status = future.result()
            if not status.ok:
                bad_rows.append({"path": status.path, "error": status.error})
    bad_rows.sort(key=lambda row: row["path"])
    return bad_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--split", default="training")
    parser.add_argument("--clip-dir", default=None, help="Default: <dataroot>/video_clip_dataset")
    parser.add_argument("--channels", default="all")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--sample-interval-seconds", type=float, default=DEFAULT_KEYFRAME_INTERVAL_SECONDS)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--segment-names", default=None, help="Optional comma-separated segment names")
    parser.add_argument("--scan-limit", type=int, default=None, help="Debug only: check first N mp4 paths")
    parser.add_argument("--bad-list", default=None, help="Reuse an existing JSONL bad-video report")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    clip_root = Path(args.clip_dir).expanduser().resolve() if args.clip_dir else dataroot / DEFAULT_CLIP_SUBDIR
    videos_root = clip_root / "videos"
    if not videos_root.exists():
        raise FileNotFoundError(f"Missing Waymo videos directory: {videos_root}")

    channels = parse_channels(args.channels)
    segments = [name.strip() for name in args.segment_names.split(",") if name.strip()] if args.segment_names else None
    run_id = time.strftime("%Y%m%d_%H%M%S")
    report_dir = clip_root / "repair_reports"
    report_path = report_dir / f"bad_videos_{run_id}.jsonl"
    quarantine_root = clip_root / "repair_quarantine" / run_id

    if args.bad_list:
        bad_rows = read_jsonl(Path(args.bad_list).expanduser().resolve())
        report_path = Path(args.bad_list).expanduser().resolve()
    else:
        bad_rows = scan_bad_videos(
            videos_root=videos_root,
            report_path=report_path,
            workers=args.workers,
            timeout_seconds=args.timeout_seconds,
            segments=segments,
            limit=args.scan_limit,
        )

    affected_segments = sorted({row["segment"] for row in bad_rows})
    print(json.dumps(
        {
            "bad_videos": len(bad_rows),
            "affected_segments": len(affected_segments),
            "bad_list": str(report_path),
            "quarantine_root": str(quarantine_root),
            "dry_run": args.dry_run,
        },
        ensure_ascii=False,
    ), flush=True)

    moved_paths = quarantine_bad_videos(bad_rows, clip_root, quarantine_root, args.dry_run)
    repair_segments(
        dataroot=dataroot,
        split=args.split,
        clip_root=clip_root,
        segments=affected_segments,
        channels=channels,
        fps=args.fps,
        clip_len=args.clip_len,
        clip_stride=args.clip_stride,
        interval_seconds=args.sample_interval_seconds,
        max_keyframes=args.max_keyframes,
        dry_run=args.dry_run,
    )

    if not args.no_verify and not args.dry_run:
        verify_bad_rows = verify_paths(moved_paths, args.workers, args.timeout_seconds)
        verify_path = report_dir / f"still_bad_after_repair_{run_id}.jsonl"
        write_jsonl(verify_path, verify_bad_rows)
        print(json.dumps(
            {
                "verified_paths": len(moved_paths),
                "still_bad_after_repair": len(verify_bad_rows),
                "verify_report": str(verify_path),
            },
            ensure_ascii=False,
        ), flush=True)

    print("Waymo Stage 5a repair done.", flush=True)


if __name__ == "__main__":
    main()
