#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 5a: build raw multi-camera video clips from Waymo v2 parquet data.

This writes the same clip manifest shape consumed by Stage 5b, while using
Waymo camera images, camera boxes, and camera-to-lidar associations instead of
nuScenes metadata.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from agent_aliases import build_scene_agent_alias_map, is_vehicle_category, risk_agent_id
from common import DEFAULT_DATAROOT, DEFAULT_KEYFRAME_INTERVAL_SECONDS, mkdir
from stage1_extract_waymo_trajectories import (
    WAYMO_TYPE_NAMES,
    list_segments,
    read_parquet,
    select_keyframe_timestamps,
)


DEFAULT_CLIP_SUBDIR = "video_clip_dataset"

WAYMO_CAMERA_NAMES = {
    1: "WAYMO_FRONT",
    2: "WAYMO_FRONT_LEFT",
    3: "WAYMO_FRONT_RIGHT",
    4: "WAYMO_SIDE_LEFT",
    5: "WAYMO_SIDE_RIGHT",
}
WAYMO_CAMERA_IDS = {value: key for key, value in WAYMO_CAMERA_NAMES.items()}
CAMERA_CHANNELS = list(WAYMO_CAMERA_IDS.keys())


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


def rel_to_root(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def parse_channels(channels_arg: str) -> List[str]:
    if channels_arg.strip().lower() == "all":
        return CAMERA_CHANNELS
    channels = [channel.strip() for channel in channels_arg.split(",") if channel.strip()]
    invalid = [channel for channel in channels if channel not in WAYMO_CAMERA_IDS]
    if invalid:
        raise ValueError(f"Invalid Waymo camera channels: {invalid}. Valid channels: {CAMERA_CHANNELS}")
    return channels


def build_clip_starts(num_frames: int, clip_len: int, clip_stride: int) -> List[int]:
    if clip_len <= 0:
        raise ValueError("--clip-len must be positive")
    if clip_stride <= 0:
        raise ValueError("--clip-stride must be positive")
    if num_frames < clip_len:
        return []
    return list(range(0, num_frames - clip_len + 1, clip_stride))


def encode_clip_with_ffmpeg(
    frame_paths: Sequence[Path],
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
    pattern = str(frame_paths[0].parent / "frame_%06d.jpg")
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


def camera_image_lookup(camera_image: pd.DataFrame) -> Dict[Tuple[int, int], bytes]:
    lookup: Dict[Tuple[int, int], bytes] = {}
    for _, row in camera_image.iterrows():
        timestamp = int(row["key.frame_timestamp_micros"])
        camera_id = int(row["key.camera_name"])
        lookup[(timestamp, camera_id)] = bytes(row["[CameraImageComponent].image"])
    return lookup


def calibration_lookup(camera_calibration: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    for _, row in camera_calibration.iterrows():
        camera_id = int(row["key.camera_name"])
        extrinsic = row.get("[CameraCalibrationComponent].extrinsic.transform")
        rows[camera_id] = {
            "width": int(row.get("[CameraCalibrationComponent].width") or 0),
            "height": int(row.get("[CameraCalibrationComponent].height") or 0),
            "intrinsic": {
                "f_u": row.get("[CameraCalibrationComponent].intrinsic.f_u"),
                "f_v": row.get("[CameraCalibrationComponent].intrinsic.f_v"),
                "c_u": row.get("[CameraCalibrationComponent].intrinsic.c_u"),
                "c_v": row.get("[CameraCalibrationComponent].intrinsic.c_v"),
                "k1": row.get("[CameraCalibrationComponent].intrinsic.k1"),
                "k2": row.get("[CameraCalibrationComponent].intrinsic.k2"),
                "p1": row.get("[CameraCalibrationComponent].intrinsic.p1"),
                "p2": row.get("[CameraCalibrationComponent].intrinsic.p2"),
                "k3": row.get("[CameraCalibrationComponent].intrinsic.k3"),
            },
            "extrinsic_transform": list(extrinsic) if extrinsic is not None else [],
        }
    return rows


def lidar_category_lookup(lidar_box: pd.DataFrame) -> Dict[Tuple[int, str], Dict[str, Any]]:
    lookup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for _, row in lidar_box.iterrows():
        timestamp = int(row["key.frame_timestamp_micros"])
        object_id = str(row["key.laser_object_id"])
        obj_type = int(row.get("[LiDARBoxComponent].type") or 0)
        lookup[(timestamp, object_id)] = {
            "category_name": WAYMO_TYPE_NAMES.get(obj_type, "unknown"),
            "category_type": obj_type,
            "num_lidar_pts": row.get("[LiDARBoxComponent].num_lidar_points_in_box"),
        }
    return lookup


def build_fake_samples_and_annotations(
    scene_name: str,
    lidar_box: pd.DataFrame,
    sampled_timestamps: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    sampled_set = set(int(ts) for ts in sampled_timestamps)
    anns_by_timestamp: Dict[int, List[str]] = {int(ts): [] for ts in sampled_timestamps}
    ann_by_token: Dict[str, Dict[str, Any]] = {}
    for _, row in lidar_box.iterrows():
        timestamp = int(row["key.frame_timestamp_micros"])
        if timestamp not in sampled_set:
            continue
        object_id = str(row["key.laser_object_id"])
        token = f"{scene_name};{timestamp};{object_id}"
        category_name = WAYMO_TYPE_NAMES.get(int(row.get("[LiDARBoxComponent].type") or 0), "unknown")
        anns_by_timestamp[timestamp].append(token)
        ann_by_token[token] = {
            "token": token,
            "instance_token": object_id,
            "category_name": category_name,
            "sample_token": f"{scene_name};{timestamp}",
        }
    samples = [
        {
            "token": f"{scene_name};{timestamp}",
            "timestamp": timestamp,
            "anns": anns_by_timestamp.get(int(timestamp), []),
        }
        for timestamp in sampled_timestamps
    ]
    return samples, ann_by_token


def build_frame_records(
    scene_name: str,
    split: str,
    sampled_timestamps: Sequence[int],
    channels: Sequence[str],
    image_lookup: Dict[Tuple[int, int], bytes],
    calibration: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    frames = []
    for frame_index, timestamp in enumerate(sampled_timestamps):
        frame = {
            "frame_index": frame_index,
            "sample_token": f"{scene_name};{timestamp}",
            "timestamp": timestamp,
            "prev_sample_token": f"{scene_name};{sampled_timestamps[frame_index - 1]}" if frame_index > 0 else "",
            "next_sample_token": (
                f"{scene_name};{sampled_timestamps[frame_index + 1]}"
                if frame_index + 1 < len(sampled_timestamps)
                else ""
            ),
            "annotation_tokens": [],
            "num_annotations": 0,
            "cameras": {},
        }
        for channel in channels:
            camera_id = WAYMO_CAMERA_IDS[channel]
            if (int(timestamp), camera_id) not in image_lookup:
                frame["cameras"][channel] = None
                continue
            calib = calibration.get(camera_id, {})
            frame["cameras"][channel] = {
                "sample_data_token": f"{scene_name};{timestamp};{channel}",
                "filename": f"{split}/camera_image/{scene_name}.parquet#{timestamp}:{channel}",
                "timestamp": timestamp,
                "is_key_frame": True,
                "width": calib.get("width"),
                "height": calib.get("height"),
                "ego_pose_token": f"{scene_name};vehicle_pose;{timestamp}",
                "calibrated_sensor_token": f"{scene_name};camera_calibration;{channel}",
            }
        frames.append(frame)
    return frames


def materialize_scene_frames(
    image_lookup: Dict[Tuple[int, int], bytes],
    sampled_timestamps: Sequence[int],
    channels: Sequence[str],
    tmp_root: Path,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for timestamp in sampled_timestamps:
        for channel in channels:
            camera_id = WAYMO_CAMERA_IDS[channel]
            image = image_lookup.get((int(timestamp), camera_id))
            if image is None:
                continue
            sample_data_token = f"{timestamp};{channel}"
            path = tmp_root / channel / f"{timestamp}.jpg"
            mkdir(path.parent)
            path.write_bytes(image)
            paths[sample_data_token] = path
    return paths


def bbox_from_camera_box(row: pd.Series, width: int, height: int) -> List[int]:
    cx = float(row["[CameraBoxComponent].box.center.x"])
    cy = float(row["[CameraBoxComponent].box.center.y"])
    sx = float(row["[CameraBoxComponent].box.size.x"])
    sy = float(row["[CameraBoxComponent].box.size.y"])
    x1 = max(0, min(width - 1, int(round(cx - sx / 2.0))))
    y1 = max(0, min(height - 1, int(round(cy - sy / 2.0))))
    x2 = max(0, min(width - 1, int(round(cx + sx / 2.0))))
    y2 = max(0, min(height - 1, int(round(cy + sy / 2.0))))
    return [x1, y1, x2, y2]


def association_lookup(association: pd.DataFrame) -> Dict[Tuple[int, int, str], str]:
    lookup: Dict[Tuple[int, int, str], str] = {}
    for _, row in association.iterrows():
        lookup[
            (
                int(row["key.frame_timestamp_micros"]),
                int(row["key.camera_name"]),
                str(row["key.camera_object_id"]),
            )
        ] = str(row["key.laser_object_id"])
    return lookup


def build_frame_annotations(
    scene_name: str,
    split: str,
    camera_box: pd.DataFrame,
    association: pd.DataFrame,
    calibration: Dict[int, Dict[str, Any]],
    lidar_categories: Dict[Tuple[int, str], Dict[str, Any]],
    sampled_timestamps: Sequence[int],
    channels: Sequence[str],
) -> List[Dict[str, Any]]:
    sampled_set = set(int(ts) for ts in sampled_timestamps)
    channel_ids = {WAYMO_CAMERA_IDS[channel] for channel in channels}
    assoc = association_lookup(association)
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    for _, row in camera_box.iterrows():
        timestamp = int(row["key.frame_timestamp_micros"])
        camera_id = int(row["key.camera_name"])
        if timestamp not in sampled_set or camera_id not in channel_ids:
            continue
        camera_object_id = str(row["key.camera_object_id"])
        laser_object_id = assoc.get((timestamp, camera_id, camera_object_id))
        if not laser_object_id:
            continue
        calib = calibration.get(camera_id, {})
        width = int(calib.get("width") or 0)
        height = int(calib.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        bbox = bbox_from_camera_box(row, width, height)
        x1, y1, x2, y2 = bbox
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        category = lidar_categories.get((timestamp, laser_object_id), {})
        category_name = category.get("category_name", WAYMO_TYPE_NAMES.get(int(row.get("[CameraBoxComponent].type") or 0), "unknown"))
        grouped.setdefault((timestamp, camera_id), []).append(
            {
                "raw_agent_id": risk_agent_id(laser_object_id),
                "risk_agent_id": risk_agent_id(laser_object_id),
                "instance_token": laser_object_id,
                "sample_annotation_token": f"{scene_name};{timestamp};{laser_object_id}",
                "camera_object_id": camera_object_id,
                "category_name": category_name,
                "is_vehicle": is_vehicle_category(category_name) or category_name == "vehicle",
                "bbox_2d_projected": bbox,
                "bbox_area": bbox_area,
                "num_lidar_pts": category.get("num_lidar_pts"),
                "num_radar_pts": None,
                "visibility_token": None,
            }
        )

    frame_annotations: List[Dict[str, Any]] = []
    for timestamp in sampled_timestamps:
        for channel in channels:
            camera_id = WAYMO_CAMERA_IDS[channel]
            annotations = grouped.get((int(timestamp), camera_id), [])
            frame_annotations.append(
                {
                    "sample_data_token": f"{scene_name};{timestamp};{channel}",
                    "raw_frame_path": f"{split}/camera_image/{scene_name}.parquet#{timestamp}:{channel}",
                    "num_visible_agents": len(annotations),
                    "num_visible_vehicles": sum(1 for ann in annotations if ann.get("is_vehicle")),
                    "annotations": annotations,
                    "frame_index": sampled_timestamps.index(timestamp),
                    "sample_token": f"{scene_name};{timestamp}",
                    "timestamp": int(timestamp),
                    "camera_channel": channel,
                    "original_filename": f"{split}/camera_image/{scene_name}.parquet#{timestamp}:{channel}",
                }
            )
    return frame_annotations


def frame_annotation_index(frame_annotations: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["sample_data_token"]: row for row in frame_annotations}


def process_segment(
    dataroot: Path,
    split: str,
    segment_name: str,
    outdir: Path,
    channels: Sequence[str],
    fps: float,
    clip_len: int,
    clip_stride: int,
    interval_seconds: float,
    overwrite_videos: bool,
    max_keyframes: Optional[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    camera_image = read_parquet(dataroot / split / "camera_image" / f"{segment_name}.parquet")
    camera_box = read_parquet(dataroot / split / "camera_box" / f"{segment_name}.parquet")
    camera_calibration = read_parquet(dataroot / split / "camera_calibration" / f"{segment_name}.parquet")
    association = read_parquet(dataroot / split / "camera_to_lidar_box_association" / f"{segment_name}.parquet")
    lidar_box = read_parquet(dataroot / split / "lidar_box" / f"{segment_name}.parquet")

    sampled_timestamps = select_keyframe_timestamps(
        sorted(camera_image["key.frame_timestamp_micros"].unique().tolist()),
        interval_seconds=interval_seconds,
        max_keyframes=max_keyframes,
    )
    if not sampled_timestamps:
        raise ValueError(f"No sampled timestamps selected for segment {segment_name}")

    image_lookup = camera_image_lookup(camera_image)
    calibration = calibration_lookup(camera_calibration)
    frames = build_frame_records(segment_name, split, sampled_timestamps, channels, image_lookup, calibration)
    starts = build_clip_starts(len(frames), clip_len, clip_stride)
    fake_samples, ann_by_token = build_fake_samples_and_annotations(segment_name, lidar_box, sampled_timestamps)
    agent_reference_map = build_scene_agent_alias_map(segment_name, fake_samples, ann_by_token)
    lidar_categories = lidar_category_lookup(lidar_box)
    frame_annotations = build_frame_annotations(
        segment_name,
        split,
        camera_box,
        association,
        calibration,
        lidar_categories,
        sampled_timestamps,
        channels,
    )
    annotations_by_sample_data = frame_annotation_index(frame_annotations)

    scene_clip_records = []
    clip_entries = []
    with tempfile.TemporaryDirectory(prefix=f"waymo_rawclips_{segment_name}_") as tmp:
        tmp_root = Path(tmp)
        scene_frame_paths = materialize_scene_frames(image_lookup, sampled_timestamps, channels, tmp_root / "frames")

        for start in starts:
            end = start + clip_len - 1
            selected_frames = frames[start : end + 1]
            clip_id = f"{segment_name}_{start:04d}_{end:04d}"
            videos: Dict[str, str] = {}
            channel_records: Dict[str, Dict[str, Any]] = {}

            for channel in channels:
                channel_frames = [frame["cameras"].get(channel) for frame in selected_frames]
                if any(frame is None for frame in channel_frames):
                    continue
                clip_tmp_dir = tmp_root / "clips" / segment_name / channel / clip_id
                mkdir(clip_tmp_dir)
                frame_paths = []
                for index, frame in enumerate(channel_frames):
                    token_suffix = f"{selected_frames[index]['timestamp']};{channel}"
                    src = scene_frame_paths[token_suffix]
                    dst = clip_tmp_dir / f"frame_{index:06d}.jpg"
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(src)
                    frame_paths.append(dst)

                clip_filename = f"{segment_name}_{channel}_{start:04d}_{end:04d}.mp4"
                clip_path = outdir / "videos" / segment_name / channel / clip_filename
                wrote = encode_clip_with_ffmpeg(frame_paths, clip_path, fps, overwrite_videos)
                clip_rel_path = rel_to_root(clip_path, outdir)
                sample_data_tokens = [frame["sample_data_token"] for frame in channel_frames if frame is not None]
                videos[channel] = clip_rel_path
                channel_records[channel] = {
                    "camera_channel": channel,
                    "clip_path": clip_rel_path,
                    "wrote_video": wrote,
                    "sample_data_tokens": sample_data_tokens,
                    "original_filenames": [frame["filename"] for frame in channel_frames if frame is not None],
                    "visible_agent_counts": [
                        annotations_by_sample_data[token]["num_visible_agents"] for token in sample_data_tokens
                    ],
                    "visible_vehicle_counts": [
                        annotations_by_sample_data[token]["num_visible_vehicles"] for token in sample_data_tokens
                    ],
                }
                clip_entries.append(
                    {
                        "clip_id": clip_id,
                        "scene_name": segment_name,
                        "scene_token": segment_name,
                        "split": split,
                        "camera_channel": channel,
                        "clip_path": clip_rel_path,
                        "start_frame_index": start,
                        "end_frame_index": end,
                        "num_frames": len(selected_frames),
                        "fps": fps,
                        "duration_seconds": (selected_frames[-1]["timestamp"] - selected_frames[0]["timestamp"]) / 1e6,
                        "sample_tokens": [frame["sample_token"] for frame in selected_frames],
                        "sample_data_tokens": sample_data_tokens,
                        "timestamps": [frame["timestamp"] for frame in selected_frames],
                        "original_filenames": channel_records[channel]["original_filenames"],
                        "agent_reference_map": agent_reference_map,
                    }
                )

            scene_clip_records.append(
                {
                    "clip_id": clip_id,
                    "scene_name": segment_name,
                    "scene_token": segment_name,
                    "split": split,
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
                    "agent_reference_map": agent_reference_map,
                }
            )

    scene_manifest_path = outdir / "metadata" / f"{segment_name}_clips.json"
    scene_manifest = {
        "scene_name": segment_name,
        "scene_token": segment_name,
        "split": split,
        "description": f"Waymo Open Dataset v2 {split} segment",
        "num_keyframes": len(frames),
        "num_clips": len(scene_clip_records),
        "clip_len": clip_len,
        "clip_stride": clip_stride,
        "fps": fps,
        "channels": list(channels),
        "camera_name_mapping": WAYMO_CAMERA_IDS,
        "annotation_mode": (
            "raw Waymo camera videos; camera boxes and camera-to-lidar associations "
            "used as metadata-only visibility references"
        ),
        "agent_reference_map": agent_reference_map,
        "clips": scene_clip_records,
        "frame_annotations": frame_annotations,
        "frame_annotation_note": "Frame annotations are metadata only. Generated videos contain raw camera pixels.",
    }
    write_json(scene_manifest_path, scene_manifest)

    scene_record = {
        "scene_name": segment_name,
        "scene_token": segment_name,
        "split": split,
        "description": f"Waymo Open Dataset v2 {split} segment",
        "num_keyframes": len(frames),
        "num_clips": len(scene_clip_records),
        "first_sample_token": frames[0]["sample_token"] if frames else None,
        "last_sample_token": frames[-1]["sample_token"] if frames else None,
        "clip_manifest": rel_to_root(scene_manifest_path, outdir),
        "agent_reference_count": len(agent_reference_map.get("references", [])),
        "vehicle_reference_count": sum(1 for row in agent_reference_map.get("references", []) if row.get("is_vehicle")),
    }
    return scene_record, clip_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--outdir", default=None, help=f"Default: <dataroot>/{DEFAULT_CLIP_SUBDIR}")
    parser.add_argument("--segment-names", default=None, help="Optional comma-separated segment names")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between sampled Waymo frames. Default is 0.5 for 2Hz.",
    )
    parser.add_argument("--overwrite-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else dataroot / DEFAULT_CLIP_SUBDIR
    channels = parse_channels(args.channels)

    camera_image_dir = dataroot / args.split / "camera_image"
    if not camera_image_dir.exists():
        raise FileNotFoundError(f"Missing Waymo camera_image directory: {camera_image_dir}")

    if args.segment_names:
        selected_segments = [name.strip() for name in args.segment_names.split(",") if name.strip()]
    else:
        required_components = [
            "camera_image",
            "camera_box",
            "camera_calibration",
            "camera_to_lidar_box_association",
            "lidar_box",
        ]
        available_sets = [set(list_segments(dataroot / args.split / component)) for component in required_components]
        selected_segments = sorted(set.intersection(*available_sets))
    if args.max_scenes is not None:
        selected_segments = selected_segments[: args.max_scenes]
    if not selected_segments:
        raise ValueError("No Waymo segments selected for Stage 5a.")

    mkdir(outdir / "videos")
    mkdir(outdir / "metadata")
    mkdir(outdir / "splits")

    print("=" * 80)
    print("Waymo raw camera video clip builder with agent reference metadata")
    print("=" * 80)
    print(f"dataroot:    {dataroot}")
    print(f"split:       {args.split}")
    print(f"outdir:      {outdir}")
    print(f"channels:    {channels}")
    print(f"fps:         {args.fps}")
    print(f"clip_len:    {args.clip_len}")
    print(f"clip_stride: {args.clip_stride}")
    print(f"interval:    {args.sample_interval_seconds}")
    print("=" * 80)

    all_scene_records = []
    all_clip_entries = []
    for segment_name in tqdm(selected_segments, desc="Processing Waymo segments"):
        scene_record, clip_entries = process_segment(
            dataroot=dataroot,
            split=args.split,
            segment_name=segment_name,
            outdir=outdir,
            channels=channels,
            fps=args.fps,
            clip_len=args.clip_len,
            clip_stride=args.clip_stride,
            interval_seconds=args.sample_interval_seconds,
            overwrite_videos=args.overwrite_videos,
            max_keyframes=args.max_keyframes,
        )
        all_scene_records.append(scene_record)
        all_clip_entries.extend(clip_entries)

    write_json(outdir / "metadata" / "scenes.json", all_scene_records)
    write_jsonl(outdir / "metadata" / "clips.jsonl", all_clip_entries)
    train_scene_names = [row["scene_name"] for row in all_scene_records if row["split"] == "training"]
    val_scene_names = [row["scene_name"] for row in all_scene_records if row["split"] == "validation"]
    write_txt(outdir / "splits" / "train_scenes.txt", train_scene_names)
    write_txt(outdir / "splits" / "val_scenes.txt", val_scene_names)
    write_jsonl(outdir / "splits" / "train_clips.jsonl", [row for row in all_clip_entries if row["split"] == "training"])
    write_jsonl(outdir / "splits" / "val_clips.jsonl", [row for row in all_clip_entries if row["split"] == "validation"])

    print("Waymo Stage 5a done.")
    print(f"Scenes exported: {len(all_scene_records)}")
    print(f"Camera clip entries: {len(all_clip_entries)}")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
