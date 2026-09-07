#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 5a: build 5-frame raw Bench2Drive camera video clips.

Videos contain raw camera pixels only. Bench2Drive annotations are used offline
for agent category/id indexing and projected-box visibility metadata.
"""

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from agent_aliases import build_scene_agent_alias_map, category_label, is_vehicle_category, risk_agent_id
from common import DEFAULT_DATAROOT, DEFAULT_KEYFRAME_INTERVAL_SECONDS, mkdir


DEFAULT_CLIP_SUBDIR = "raw_video_clip_dataset_v1"

CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CAMERA_DIRS = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}

CATEGORY_MAP = {
    "vehicle": "vehicle.car",
    "walker": "pedestrian",
    "pedestrian": "pedestrian",
    "traffic_light": "traffic_light",
    "traffic_sign": "traffic_sign",
}


def load_frame(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
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


def load_clip_selection(path: Optional[str]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    if not path:
        return None
    selection_path = Path(path).expanduser().resolve()
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing clip selection manifest: {selection_path}")
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    with open(selection_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scene_name = row.get("scene_name")
            if not scene_name:
                raise ValueError(f"Clip selection row is missing scene_name: {row}")
            by_scene.setdefault(scene_name, []).append(row)
    for rows in by_scene.values():
        rows.sort(key=lambda item: int(item["start_frame_index"]))
    return by_scene


def rel_to_root(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def frame_index(path: Path) -> int:
    return int(path.name.split(".")[0])


def timestamp_for_frame(index: int, interval_seconds: float) -> int:
    return int(round(index * interval_seconds * 1_000_000))


def parse_channels(channels_arg: str) -> List[str]:
    if channels_arg.strip().lower() == "all":
        return CAMERA_CHANNELS
    channels = [channel.strip() for channel in channels_arg.split(",") if channel.strip()]
    invalid = [channel for channel in channels if channel not in CAMERA_CHANNELS]
    if invalid:
        raise ValueError(f"Invalid camera channels: {invalid}. Valid channels: {CAMERA_CHANNELS}")
    return channels


def list_scenes(raw_root: Path) -> List[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing Bench2Drive raw directory: {raw_root}")
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def category_name(box: Dict[str, Any]) -> str:
    raw_class = str(box.get("class") or "object").strip()
    if raw_class == "vehicle":
        base_type = str(box.get("base_type") or "").strip().lower()
        type_id = str(box.get("type_id") or "").strip().lower()
        if "motorcycle" in type_id or base_type == "motorcycle":
            return "vehicle.motorcycle"
        if "bike" in type_id or "bicycle" in type_id or base_type == "bicycle":
            return "vehicle.bicycle"
        if "truck" in type_id or base_type == "truck":
            return "vehicle.truck"
        if "bus" in type_id or base_type == "bus":
            return "vehicle.bus"
    return CATEGORY_MAP.get(raw_class, raw_class.lower())


def annotation_token(scene_name: str, frame_id: int, box: Dict[str, Any]) -> str:
    return f"{scene_name};{frame_id:05d};{box.get('id')}"


def build_frame_records(
    scene_path: Path,
    frames: Sequence[Tuple[Path, Dict[str, Any]]],
    channels: Sequence[str],
    interval_seconds: float,
) -> List[Dict[str, Any]]:
    records = []
    for timestep, (anno_path, data) in enumerate(frames):
        frame_id = frame_index(anno_path)
        sample_token = f"{scene_path.name};{frame_id:05d}"
        frame = {
            "frame_index": timestep,
            "source_frame_index": frame_id,
            "sample_token": sample_token,
            "timestamp": timestamp_for_frame(timestep, interval_seconds),
            "prev_sample_token": f"{scene_path.name};{frame_id - 1:05d}" if timestep > 0 else "",
            "next_sample_token": f"{scene_path.name};{frame_id + 1:05d}" if timestep < len(frames) - 1 else "",
            "annotation_tokens": [
                annotation_token(scene_path.name, frame_id, box)
                for box in data.get("bounding_boxes", [])
                if box.get("class") != "ego_vehicle" and box.get("id") is not None
            ],
            "num_annotations": sum(1 for box in data.get("bounding_boxes", []) if box.get("class") != "ego_vehicle"),
            "cameras": {},
        }
        for channel in channels:
            image_path = scene_path / "camera" / CAMERA_DIRS[channel] / f"{frame_id:05d}.jpg"
            if not image_path.exists():
                frame["cameras"][channel] = None
                continue
            sensor = data.get("sensors", {}).get(channel, {})
            frame["cameras"][channel] = {
                "sample_data_token": f"{scene_path.name};{channel};{frame_id:05d}",
                "filename": str(image_path),
                "timestamp": timestamp_for_frame(timestep, interval_seconds),
                "is_key_frame": True,
                "width": sensor.get("image_size_x"),
                "height": sensor.get("image_size_y"),
                "ego_pose_token": f"{scene_path.name};ego_pose;{frame_id:05d}",
                "calibrated_sensor_token": f"{scene_path.name};{channel};calib",
                "source_camera_dir": CAMERA_DIRS[channel],
            }
        records.append(frame)
    return records


def build_clip_starts(num_frames: int, clip_len: int, clip_stride: int) -> List[int]:
    if clip_len <= 0:
        raise ValueError("--clip-len must be positive")
    if clip_stride <= 0:
        raise ValueError("--clip-stride must be positive")
    if num_frames < clip_len:
        return []
    return list(range(0, num_frames - clip_len + 1, clip_stride))


def transform_points(world2cam: Sequence[Sequence[float]], points: Sequence[Sequence[float]]) -> np.ndarray:
    transform = np.asarray(world2cam, dtype=float)
    pts = np.asarray(points, dtype=float)
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=float)], axis=1)
    return (transform @ homogeneous.T).T[:, :3]


def projected_2d_bbox(box: Dict[str, Any], sensor: Dict[str, Any]) -> List[int]:
    points = box.get("world_cord") or []
    intrinsic = sensor.get("intrinsic")
    world2cam = sensor.get("world2cam")
    width = int(sensor.get("image_size_x") or 0)
    height = int(sensor.get("image_size_y") or 0)
    if not points or intrinsic is None or world2cam is None or width <= 0 or height <= 0:
        return [0, 0, 0, 0]

    camera_points = transform_points(world2cam, points)
    in_front = camera_points[:, 2] > 1e-3
    if not bool(in_front.any()):
        return [0, 0, 0, 0]
    visible = camera_points[in_front]
    k = np.asarray(intrinsic, dtype=float)
    pixels = (k @ visible.T).T
    xs = pixels[:, 0] / pixels[:, 2]
    ys = pixels[:, 1] / pixels[:, 2]
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not bool(finite.any()):
        return [0, 0, 0, 0]
    x1 = int(np.clip(np.min(xs[finite]), 0, width - 1))
    y1 = int(np.clip(np.min(ys[finite]), 0, height - 1))
    x2 = int(np.clip(np.max(xs[finite]), 0, width - 1))
    y2 = int(np.clip(np.max(ys[finite]), 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0, 0]
    return [x1, y1, x2, y2]


def inspect_camera_frame(
    scene_name: str,
    frame: Dict[str, Any],
    data: Dict[str, Any],
    channel: str,
) -> Dict[str, Any]:
    cam = frame["cameras"][channel]
    sensor = data.get("sensors", {}).get(channel, {})
    width = int(sensor.get("image_size_x") or cam.get("width") or 0)
    height = int(sensor.get("image_size_y") or cam.get("height") or 0)
    annotations = []
    for box in data.get("bounding_boxes", []):
        if box.get("class") == "ego_vehicle" or box.get("id") is None:
            continue
        actor_id = str(box["id"])
        cat_name = category_name(box)
        bbox = projected_2d_bbox(box, sensor)
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        annotations.append(
            {
                "raw_agent_id": risk_agent_id(actor_id),
                "risk_agent_id": risk_agent_id(actor_id),
                "instance_token": actor_id,
                "sample_annotation_token": annotation_token(scene_name, frame["source_frame_index"], box),
                "category_name": cat_name,
                "category": category_label(cat_name),
                "is_vehicle": is_vehicle_category(cat_name),
                "bbox_2d_projected": bbox,
                "bbox_area": area,
                "visibility_token": "projected" if area > 0 else "not_projected",
                "bench2drive_class": box.get("class"),
                "state": box.get("state"),
            }
        )
    return {
        "sample_data_token": cam["sample_data_token"],
        "raw_frame_path": cam["filename"],
        "num_visible_agents": sum(1 for ann in annotations if int(ann.get("bbox_area") or 0) > 0),
        "num_visible_vehicles": sum(
            1 for ann in annotations if ann.get("is_vehicle") and int(ann.get("bbox_area") or 0) > 0
        ),
        "image_width": width,
        "image_height": height,
        "annotations": annotations,
    }


def encode_clip_with_ffmpeg(
    frame_paths: Sequence[Path],
    output_path: Path,
    fps: float,
    overwrite: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        return False
    mkdir(output_path.parent)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return encode_clip_with_opencv(frame_paths, output_path, fps)

    suffix = frame_paths[0].suffix.lower() or ".jpg"
    pattern = str(frame_paths[0].parent / f"frame_%06d{suffix}")
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


def encode_clip_with_opencv(
    frame_paths: Sequence[Path],
    output_path: Path,
    fps: float,
) -> bool:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Stage 5a requires ffmpeg or opencv-python to encode video clips") from exc

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read frame for video encoding: {frame_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open OpenCV VideoWriter for {output_path}")
    try:
        writer.write(first)
        for frame_path in frame_paths[1:]:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame for video encoding: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return True


def materialize_clip_frames(
    scene_frame_paths: Dict[str, Path],
    selected_frames: Sequence[Dict[str, Any]],
    channel: str,
    clip_tmp_dir: Path,
) -> List[Path]:
    mkdir(clip_tmp_dir)
    frame_paths = []
    for index, frame in enumerate(selected_frames):
        cam = frame["cameras"][channel]
        src = scene_frame_paths[cam["sample_data_token"]]
        dst = clip_tmp_dir / f"frame_{index:06d}{src.suffix.lower() or '.jpg'}"
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        frame_paths.append(dst)
    return frame_paths


def inspect_scene_frames(
    scene_name: str,
    frames: Sequence[Dict[str, Any]],
    frame_data: Sequence[Dict[str, Any]],
    channels: Sequence[str],
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, Any]]]:
    frame_paths: Dict[str, Path] = {}
    frame_annotations: Dict[str, Dict[str, Any]] = {}
    for frame, data in zip(frames, frame_data):
        for channel in channels:
            cam = frame["cameras"].get(channel)
            if cam is None:
                continue
            record = inspect_camera_frame(scene_name, frame, data, channel)
            record.update(
                {
                    "frame_index": frame["frame_index"],
                    "source_frame_index": frame["source_frame_index"],
                    "sample_token": frame["sample_token"],
                    "timestamp": frame["timestamp"],
                    "camera_channel": channel,
                    "original_filename": cam["filename"],
                }
            )
            frame_paths[cam["sample_data_token"]] = Path(cam["filename"])
            frame_annotations[cam["sample_data_token"]] = record
    return frame_paths, frame_annotations


def build_scene_agent_inputs(
    scene_name: str,
    frames: Sequence[Dict[str, Any]],
    frame_data: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    samples = []
    ann_by_token: Dict[str, Dict[str, Any]] = {}
    for frame, data in zip(frames, frame_data):
        anns = []
        for box in data.get("bounding_boxes", []):
            if box.get("class") == "ego_vehicle" or box.get("id") is None:
                continue
            token = annotation_token(scene_name, frame["source_frame_index"], box)
            actor_id = str(box["id"])
            anns.append(token)
            ann_by_token[token] = {
                "token": token,
                "instance_token": actor_id,
                "category_name": category_name(box),
                "sample_token": frame["sample_token"],
            }
        samples.append({"token": frame["sample_token"], "anns": anns})
    return samples, ann_by_token


def process_scene(
    scene_path: Path,
    outdir: Path,
    channels: Sequence[str],
    fps: float,
    clip_len: int,
    clip_stride: int,
    keyframe_interval_seconds: float,
    overwrite_videos: bool,
    max_keyframes: Optional[int],
    selected_clip_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scene_name = scene_path.name
    anno_paths = sorted((scene_path / "anno").glob("*.json.gz"), key=frame_index)
    if max_keyframes is not None:
        anno_paths = anno_paths[:max_keyframes]
    frame_data = [load_frame(path) for path in anno_paths]
    source_frames = list(zip(anno_paths, frame_data))
    frames = build_frame_records(scene_path, source_frames, channels, keyframe_interval_seconds)
    if selected_clip_rows is None:
        starts = build_clip_starts(len(frames), clip_len, clip_stride)
    else:
        starts = sorted({int(row["start_frame_index"]) for row in selected_clip_rows})
    samples, ann_by_token = build_scene_agent_inputs(scene_name, frames, frame_data)

    scene_clip_records = []
    clip_entries = []
    with tempfile.TemporaryDirectory(prefix=f"b2dclips_{scene_name}_") as tmp:
        scene_frame_paths, scene_frame_annotations = inspect_scene_frames(
            scene_name=scene_name,
            frames=frames,
            frame_data=frame_data,
            channels=channels,
        )
        agent_reference_map = build_scene_agent_alias_map(scene_name, samples, ann_by_token)

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
                clip_filename = f"{scene_name}_{channel}_{start:04d}_{end:04d}.mp4"
                clip_path = outdir / "videos" / scene_name / channel / clip_filename
                clip_tmp_dir = Path(tmp) / "clips" / scene_name / channel / clip_id
                frame_paths = materialize_clip_frames(scene_frame_paths, selected_frames, channel, clip_tmp_dir)
                wrote = encode_clip_with_ffmpeg(frame_paths, clip_path, fps, overwrite_videos)
                clip_rel_path = rel_to_root(clip_path, outdir)
                videos[channel] = clip_rel_path
                channel_records[channel] = {
                    "camera_channel": channel,
                    "clip_path": clip_rel_path,
                    "wrote_video": wrote,
                    "sample_data_tokens": [frame["sample_data_token"] for frame in channel_frames if frame is not None],
                    "original_filenames": [frame["filename"] for frame in channel_frames if frame is not None],
                    "visible_agent_counts": [
                        scene_frame_annotations[frame["sample_data_token"]]["num_visible_agents"]
                        for frame in channel_frames
                        if frame is not None
                    ],
                    "visible_vehicle_counts": [
                        scene_frame_annotations[frame["sample_data_token"]]["num_visible_vehicles"]
                        for frame in channel_frames
                        if frame is not None
                    ],
                }
                clip_entries.append(
                    {
                        "clip_id": clip_id,
                        "scene_name": scene_name,
                        "scene_token": scene_name,
                        "split": "unknown",
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
                        "original_filenames": channel_records[channel]["original_filenames"],
                        "agent_reference_map": agent_reference_map,
                    }
                )

            scene_clip_records.append(
                {
                    "clip_id": clip_id,
                    "scene_name": scene_name,
                    "scene_token": scene_name,
                    "split": "unknown",
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

        scene_manifest_path = outdir / "metadata" / f"{scene_name}_clips.json"
        scene_manifest = {
            "scene_name": scene_name,
            "scene_token": scene_name,
            "split": "unknown",
            "description": scene_name,
            "source": "Bench2Drive-V0.0.4",
            "raw_scene_path": str(scene_path),
            "num_keyframes": len(frames),
            "num_clips": len(scene_clip_records),
            "clip_len": clip_len,
            "clip_stride": clip_stride,
            "fps": fps,
            "keyframe_interval_seconds": keyframe_interval_seconds,
            "channels": list(channels),
            "annotation_mode": (
                "raw videos; Bench2Drive annotations used offline for category/id indexing "
                "and clip-level close-agent target references only"
            ),
            "agent_reference_map": agent_reference_map,
            "clips": scene_clip_records,
            "frame_annotations": list(scene_frame_annotations.values()),
            "frame_annotation_note": "Frame annotations are metadata only. Generated videos contain raw camera pixels.",
        }
        write_json(scene_manifest_path, scene_manifest)

    scene_record = {
        "scene_name": scene_name,
        "scene_token": scene_name,
        "split": "unknown",
        "description": scene_name,
        "num_keyframes": len(frames),
        "num_clips": len(scene_clip_records),
        "first_sample_token": frames[0]["sample_token"] if frames else None,
        "last_sample_token": frames[-1]["sample_token"] if frames else None,
        "clip_manifest": rel_to_root(scene_manifest_path, outdir),
        "agent_reference_count": len(agent_reference_map.get("references", [])),
        "vehicle_reference_count": sum(1 for row in agent_reference_map.get("references", []) if row.get("is_vehicle")),
        "selected_clip_count": len(scene_clip_records) if selected_clip_rows is not None else None,
    }
    return scene_record, clip_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--raw-subdir", default="raw_camera_anno")
    parser.add_argument("--outdir", default=None, help=f"Default: <dataroot>/{DEFAULT_CLIP_SUBDIR}")
    parser.add_argument("--scene-names", default=None, help="Optional comma-separated scene names")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument(
        "--clip-selection",
        default=None,
        help="Optional Stage 4b selected_clips.jsonl. When provided, only listed clips are encoded.",
    )
    parser.add_argument(
        "--keyframe-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between exported Bench2Drive frames. Default is 0.5.",
    )
    parser.add_argument("--overwrite-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    raw_root = dataroot / args.raw_subdir
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else dataroot / DEFAULT_CLIP_SUBDIR
    channels = parse_channels(args.channels)
    clip_selection = load_clip_selection(args.clip_selection)

    mkdir(outdir / "videos")
    mkdir(outdir / "metadata")
    mkdir(outdir / "splits")

    scene_filter = None
    if args.scene_names:
        scene_filter = {name.strip() for name in args.scene_names.split(",") if name.strip()}
    selected_scenes = [scene for scene in list_scenes(raw_root) if scene_filter is None or scene.name in scene_filter]
    if clip_selection is not None:
        selected_names = set(clip_selection)
        selected_scenes = [scene for scene in selected_scenes if scene.name in selected_names]
    if args.max_scenes is not None:
        selected_scenes = selected_scenes[: args.max_scenes]
    if not selected_scenes:
        raise ValueError("No Bench2Drive scenes selected.")

    scene_records = []
    all_clip_entries = []
    for scene_path in tqdm(selected_scenes, desc="Building Bench2Drive clips"):
        scene_record, clip_entries = process_scene(
            scene_path=scene_path,
            outdir=outdir,
            channels=channels,
            fps=args.fps,
            clip_len=args.clip_len,
            clip_stride=args.clip_stride,
            keyframe_interval_seconds=args.keyframe_interval_seconds,
            overwrite_videos=args.overwrite_videos,
            max_keyframes=args.max_keyframes,
            selected_clip_rows=(clip_selection or {}).get(scene_path.name) if clip_selection is not None else None,
        )
        scene_records.append(scene_record)
        all_clip_entries.extend(clip_entries)

    write_json(
        outdir / "metadata" / "scenes.json",
        {
            "source": "Bench2Drive-V0.0.4",
            "dataroot": str(dataroot),
            "raw_root": str(raw_root),
            "num_scenes": len(scene_records),
            "num_clip_entries": len(all_clip_entries),
            "channels": channels,
            "scenes": scene_records,
        },
    )
    write_jsonl(outdir / "metadata" / "clips.jsonl", all_clip_entries)
    print(f"Bench2Drive Stage 5a done: {len(scene_records)} scenes, {len(all_clip_entries)} camera clip entries -> {outdir}")


if __name__ == "__main__":
    main()
