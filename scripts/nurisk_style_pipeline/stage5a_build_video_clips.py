#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 5a: build 5-keyframe sliding-window camera video clips."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import DEFAULT_DATAROOT, mkdir


CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

MINI_TRAIN = {
    "scene-0061",
    "scene-0553",
    "scene-0655",
    "scene-0757",
    "scene-0796",
    "scene-1077",
    "scene-1094",
    "scene-1100",
}
MINI_VAL = {"scene-0103", "scene-0916"}


def load_table(version_dir: Path, name: str) -> List[Dict[str, Any]]:
    path = version_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing nuScenes metadata file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_txt(path: Path, lines: Sequence[str]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def by_token(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["token"]: row for row in rows}


def rel_to_root(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def parse_channels(channels_arg: str) -> List[str]:
    if channels_arg.strip().lower() == "all":
        return CAMERA_CHANNELS
    channels = [channel.strip() for channel in channels_arg.split(",") if channel.strip()]
    invalid = [channel for channel in channels if channel not in CAMERA_CHANNELS]
    if invalid:
        raise ValueError(f"Invalid camera channels: {invalid}. Valid channels: {CAMERA_CHANNELS}")
    return channels


def get_scene_split(scene_name: str, version: str) -> str:
    if version.endswith("mini"):
        if scene_name in MINI_TRAIN:
            return "mini_train"
        if scene_name in MINI_VAL:
            return "mini_val"
    return "unknown"


def collect_scene_samples(scene: Dict[str, Any], sample_by_token: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = sample_by_token[token]
        samples.append(sample)
        if token == scene["last_sample_token"]:
            break
        token = sample["next"]
    return samples


def build_sample_data_indexes(
    sample_data: List[Dict[str, Any]],
    calibrated_sensor_by_token: Dict[str, Dict[str, Any]],
    sensor_by_token: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    by_sample_channel = {}
    for row in sample_data:
        if not row.get("is_key_frame") or not row.get("filename", "").startswith("samples/"):
            continue
        calibrated = calibrated_sensor_by_token.get(row["calibrated_sensor_token"], {})
        sensor = sensor_by_token.get(calibrated.get("sensor_token", ""), {})
        channel = sensor.get("channel")
        if channel:
            by_sample_channel[(row["sample_token"], channel)] = row
    return by_sample_channel


def build_frame_records(
    samples: List[Dict[str, Any]],
    sample_data_by_sample_channel: Dict[Tuple[str, str], Dict[str, Any]],
    channels: Sequence[str],
) -> List[Dict[str, Any]]:
    frames = []
    for frame_index, sample in enumerate(samples):
        frame = {
            "frame_index": frame_index,
            "sample_token": sample["token"],
            "timestamp": sample["timestamp"],
            "prev_sample_token": sample["prev"],
            "next_sample_token": sample["next"],
            "annotation_tokens": sample.get("anns", []),
            "num_annotations": len(sample.get("anns", [])),
            "cameras": {},
        }
        for channel in channels:
            row = sample_data_by_sample_channel.get((sample["token"], channel))
            if row is None or not row.get("is_key_frame") or not row.get("filename", "").startswith("samples/"):
                frame["cameras"][channel] = None
            else:
                frame["cameras"][channel] = {
                    "sample_data_token": row["token"],
                    "filename": row["filename"],
                    "timestamp": row["timestamp"],
                    "is_key_frame": row["is_key_frame"],
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "ego_pose_token": row.get("ego_pose_token"),
                    "calibrated_sensor_token": row.get("calibrated_sensor_token"),
                }
        frames.append(frame)
    return frames


def build_clip_starts(num_frames: int, clip_len: int, clip_stride: int) -> List[int]:
    if clip_len <= 0:
        raise ValueError("--clip-len must be positive")
    if clip_stride <= 0:
        raise ValueError("--clip-stride must be positive")
    if num_frames < clip_len:
        return []
    return list(range(0, num_frames - clip_len + 1, clip_stride))


def encode_clip_with_ffmpeg(
    dataroot: Path,
    frame_paths: Sequence[str],
    output_path: Path,
    fps: float,
    overwrite: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for Stage 5a but was not found in PATH")

    mkdir(output_path.parent)
    with tempfile.TemporaryDirectory(prefix="nuscenes_clip_") as tmp:
        tmp_path = Path(tmp)
        for index, rel_frame_path in enumerate(frame_paths):
            src = dataroot / rel_frame_path
            if not src.exists():
                raise FileNotFoundError(f"Missing camera frame: {src}")
            suffix = src.suffix.lower() or ".jpg"
            link = tmp_path / f"frame_{index:06d}{suffix}"
            link.symlink_to(src)

        suffix = Path(frame_paths[0]).suffix.lower() or ".jpg"
        pattern = str(tmp_path / f"frame_%06d{suffix}")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-frames:v",
            str(len(frame_paths)),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(command, check=True)
    return True


def process_scene(
    scene: Dict[str, Any],
    sample_by_token: Dict[str, Dict[str, Any]],
    sample_data_by_sample_channel: Dict[Tuple[str, str], Dict[str, Any]],
    dataroot: Path,
    outdir: Path,
    version: str,
    channels: Sequence[str],
    fps: float,
    clip_len: int,
    clip_stride: int,
    overwrite_videos: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scene_name = scene["name"]
    scene_split = get_scene_split(scene_name, version)
    samples = collect_scene_samples(scene, sample_by_token)
    frames = build_frame_records(samples, sample_data_by_sample_channel, channels)
    starts = build_clip_starts(len(frames), clip_len, clip_stride)

    scene_clip_records = []
    clip_entries = []
    for start in starts:
        end = start + clip_len - 1
        selected_frames = frames[start : end + 1]
        clip_id = f"{scene_name}_{start:04d}_{end:04d}"
        videos: Dict[str, str] = {}
        channel_records = {}

        for channel in channels:
            channel_frames = [frame["cameras"].get(channel) for frame in selected_frames]
            if any(frame is None for frame in channel_frames):
                continue
            frame_paths = [frame["filename"] for frame in channel_frames if frame is not None]
            clip_filename = f"{scene_name}_{channel}_{start:04d}_{end:04d}.mp4"
            clip_path = outdir / "videos" / scene_name / channel / clip_filename
            wrote = encode_clip_with_ffmpeg(dataroot, frame_paths, clip_path, fps, overwrite_videos)
            clip_rel_path = rel_to_root(clip_path, outdir)
            videos[channel] = clip_rel_path
            channel_records[channel] = {
                "camera_channel": channel,
                "clip_path": clip_rel_path,
                "wrote_video": wrote,
                "sample_data_tokens": [frame["sample_data_token"] for frame in channel_frames if frame is not None],
                "original_filenames": frame_paths,
            }
            clip_entries.append(
                {
                    "clip_id": clip_id,
                    "scene_name": scene_name,
                    "scene_token": scene["token"],
                    "split": scene_split,
                    "camera_channel": channel,
                    "clip_path": clip_rel_path,
                    "start_frame_index": start,
                    "end_frame_index": end,
                    "num_frames": len(selected_frames),
                    "fps": fps,
                    "duration_seconds": (selected_frames[-1]["timestamp"] - selected_frames[0]["timestamp"]) / 1e6,
                    "sample_tokens": [frame["sample_token"] for frame in selected_frames],
                    "sample_data_tokens": channel_records[channel]["sample_data_tokens"],
                    "timestamps": [frame["timestamp"] for frame in selected_frames],
                    "original_filenames": frame_paths,
                }
            )

        scene_clip_records.append(
            {
                "clip_id": clip_id,
                "scene_name": scene_name,
                "scene_token": scene["token"],
                "split": scene_split,
                "start_frame_index": start,
                "end_frame_index": end,
                "num_frames": len(selected_frames),
                "fps": fps,
                "duration_seconds": (selected_frames[-1]["timestamp"] - selected_frames[0]["timestamp"]) / 1e6,
                "timestamp_start": selected_frames[0]["timestamp"],
                "timestamp_end": selected_frames[-1]["timestamp"],
                "sample_tokens": [frame["sample_token"] for frame in selected_frames],
                "timestamps": [frame["timestamp"] for frame in selected_frames],
                "videos": videos,
                "channels": channel_records,
            }
        )

    scene_manifest_path = outdir / "metadata" / f"{scene_name}_clips.json"
    scene_manifest = {
        "scene_name": scene_name,
        "scene_token": scene["token"],
        "split": scene_split,
        "description": scene.get("description", ""),
        "num_keyframes": len(frames),
        "num_clips": len(scene_clip_records),
        "clip_len": clip_len,
        "clip_stride": clip_stride,
        "fps": fps,
        "channels": list(channels),
        "clips": scene_clip_records,
    }
    write_json(scene_manifest_path, scene_manifest)

    scene_record = {
        "scene_name": scene_name,
        "scene_token": scene["token"],
        "split": scene_split,
        "description": scene.get("description", ""),
        "num_keyframes": len(frames),
        "num_clips": len(scene_clip_records),
        "first_sample_token": scene.get("first_sample_token"),
        "last_sample_token": scene.get("last_sample_token"),
        "clip_manifest": rel_to_root(scene_manifest_path, outdir),
    }
    return scene_record, clip_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--outdir", default=None, help="Default: <dataroot>/video_clip_dataset")
    parser.add_argument("--scene-names", default=None, help="Optional comma-separated scene names")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--overwrite-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    version_dir = dataroot / args.version
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else dataroot / "video_clip_dataset"
    channels = parse_channels(args.channels)

    scenes = load_table(version_dir, "scene")
    samples = load_table(version_dir, "sample")
    sample_data = load_table(version_dir, "sample_data")
    calibrated_sensor = load_table(version_dir, "calibrated_sensor")
    sensor = load_table(version_dir, "sensor")

    scene_filter = None
    if args.scene_names:
        scene_filter = {name.strip() for name in args.scene_names.split(",") if name.strip()}

    selected_scenes = [scene for scene in scenes if scene_filter is None or scene["name"] in scene_filter]
    if args.max_scenes is not None:
        selected_scenes = selected_scenes[: args.max_scenes]
    if not selected_scenes:
        raise ValueError("No scenes selected. Check --scene-names or --max-scenes.")

    sample_by_token = by_token(samples)
    sample_data_by_sample_channel = build_sample_data_indexes(sample_data, by_token(calibrated_sensor), by_token(sensor))

    mkdir(outdir / "videos")
    mkdir(outdir / "metadata")
    mkdir(outdir / "splits")

    all_scene_records = []
    all_clip_entries = []
    for scene in selected_scenes:
        scene_record, clip_entries = process_scene(
            scene=scene,
            sample_by_token=sample_by_token,
            sample_data_by_sample_channel=sample_data_by_sample_channel,
            dataroot=dataroot,
            outdir=outdir,
            version=args.version,
            channels=channels,
            fps=args.fps,
            clip_len=args.clip_len,
            clip_stride=args.clip_stride,
            overwrite_videos=args.overwrite_videos,
        )
        all_scene_records.append(scene_record)
        all_clip_entries.extend(clip_entries)

    write_json(outdir / "metadata" / "scenes.json", all_scene_records)
    write_jsonl(outdir / "metadata" / "clips.jsonl", all_clip_entries)

    train_scene_names = [row["scene_name"] for row in all_scene_records if row["split"].endswith("train")]
    val_scene_names = [row["scene_name"] for row in all_scene_records if row["split"].endswith("val")]
    write_txt(outdir / "splits" / "train_scenes.txt", train_scene_names)
    write_txt(outdir / "splits" / "val_scenes.txt", val_scene_names)
    write_jsonl(outdir / "splits" / "train_clips.jsonl", [row for row in all_clip_entries if row["split"].endswith("train")])
    write_jsonl(outdir / "splits" / "val_clips.jsonl", [row for row in all_clip_entries if row["split"].endswith("val")])

    print("Stage 5a done.")
    print(f"Scenes exported: {len(all_scene_records)}")
    print(f"Camera clip entries: {len(all_clip_entries)}")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
