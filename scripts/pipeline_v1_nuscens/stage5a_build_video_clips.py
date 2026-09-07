#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 5a: build 5-keyframe raw camera video clips.

This stage preserves the existing NuRisk-style clip layout but keeps video
pixels raw. nuScenes annotations are used offline to build an agent
category/token index and visibility metadata only. Final VQA target references
are generated later at clip level from reference-frame close agents; no boxes,
ids, labels, or colors are rendered into the generated clips.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nuscenes-pipeline")

import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

from agent_aliases import build_scene_agent_alias_map, is_vehicle_category, risk_agent_id
from common import DEFAULT_DATAROOT, mkdir


DEFAULT_CLIP_SUBDIR = "raw_video_clip_dataset_v1"

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


def collect_scene_samples(nusc: NuScenes, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        if token == scene["last_sample_token"]:
            break
        token = sample["next"]
    return samples


def build_frame_records(
    nusc: NuScenes,
    samples: Sequence[Dict[str, Any]],
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
            if channel not in sample["data"]:
                frame["cameras"][channel] = None
                continue
            row = nusc.get("sample_data", sample["data"][channel])
            if not row.get("is_key_frame") or not row.get("filename", "").startswith("samples/"):
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


def projected_2d_bbox(box: Any, camera_intrinsic: np.ndarray, width: int, height: int) -> List[int]:
    corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
    xs = corners[0, :]
    ys = corners[1, :]
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.any():
        return [0, 0, 0, 0]
    x1 = int(np.clip(np.min(xs[finite]), 0, width - 1))
    y1 = int(np.clip(np.min(ys[finite]), 0, height - 1))
    x2 = int(np.clip(np.max(xs[finite]), 0, width - 1))
    y2 = int(np.clip(np.max(ys[finite]), 0, height - 1))
    return [x1, y1, x2, y2]


def inspect_camera_frame(
    nusc: NuScenes,
    sample_data_token: str,
) -> Dict[str, Any]:
    image_path, boxes, camera_intrinsic = nusc.get_sample_data(
        sample_data_token, box_vis_level=BoxVisibility.ANY
    )
    sample_data = nusc.get("sample_data", sample_data_token)
    width = int(sample_data.get("width") or 0)
    height = int(sample_data.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Missing camera image size for sample_data_token={sample_data_token}")
    annotations = []

    for box in boxes:
        ann = nusc.get("sample_annotation", box.token)
        instance_token = ann["instance_token"]
        bbox = projected_2d_bbox(box, camera_intrinsic, width, height)
        x1, y1, x2, y2 = bbox
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)

        annotations.append(
            {
                "raw_agent_id": risk_agent_id(instance_token),
                "risk_agent_id": risk_agent_id(instance_token),
                "instance_token": instance_token,
                "sample_annotation_token": box.token,
                "category_name": ann["category_name"],
                "is_vehicle": is_vehicle_category(ann.get("category_name", "")),
                "bbox_2d_projected": bbox,
                "bbox_area": bbox_area,
                "num_lidar_pts": ann.get("num_lidar_pts"),
                "num_radar_pts": ann.get("num_radar_pts"),
                "visibility_token": ann.get("visibility_token"),
            }
        )

    return {
        "sample_data_token": sample_data_token,
        "raw_frame_path": image_path,
        "num_visible_agents": len(annotations),
        "num_visible_vehicles": sum(1 for ann in annotations if ann.get("is_vehicle")),
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
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for Stage 5a but was not found in PATH")

    mkdir(output_path.parent)
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
    nusc: NuScenes,
    frames: Sequence[Dict[str, Any]],
    channels: Sequence[str],
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, Any]]]:
    frame_paths: Dict[str, Path] = {}
    frame_annotations: Dict[str, Dict[str, Any]] = {}
    for frame in frames:
        for channel in channels:
            cam = frame["cameras"].get(channel)
            if cam is None:
                continue
            sd_token = cam["sample_data_token"]
            record = inspect_camera_frame(nusc, sd_token)
            record.update(
                {
                    "frame_index": frame["frame_index"],
                    "sample_token": frame["sample_token"],
                    "timestamp": frame["timestamp"],
                    "camera_channel": channel,
                    "original_filename": cam["filename"],
                }
            )
            frame_paths[sd_token] = Path(record["raw_frame_path"])
            frame_annotations[sd_token] = record
    return frame_paths, frame_annotations


def process_scene(
    nusc: NuScenes,
    scene: Dict[str, Any],
    dataroot: Path,
    outdir: Path,
    version: str,
    channels: Sequence[str],
    fps: float,
    clip_len: int,
    clip_stride: int,
    overwrite_videos: bool,
    keep_annotated_frames: bool,
    line_width: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scene_name = scene["name"]
    scene_split = get_scene_split(scene_name, version)
    samples = collect_scene_samples(nusc, scene)
    frames = build_frame_records(nusc, samples, channels)
    starts = build_clip_starts(len(frames), clip_len, clip_stride)
    ann_by_token = {ann_token: nusc.get("sample_annotation", ann_token) for sample in samples for ann_token in sample.get("anns", [])}

    scene_clip_records = []
    clip_entries = []
    with tempfile.TemporaryDirectory(prefix=f"rawclips_{scene_name}_") as tmp:
        scene_frame_paths, scene_frame_annotations = inspect_scene_frames(
            nusc=nusc,
            frames=frames,
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
                        "original_filenames": channel_records[channel]["original_filenames"],
                        "agent_reference_map": agent_reference_map,
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
                    "agent_reference_map": agent_reference_map,
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
            "annotation_mode": (
                "raw videos; nuScenes annotations used offline for category/token indexing "
                "and clip-level close-agent target references only"
            ),
            "agent_reference_map": agent_reference_map,
            "clips": scene_clip_records,
            "frame_annotations": list(scene_frame_annotations.values()),
        }
        scene_manifest["frame_annotation_note"] = (
            "Frame annotations are metadata only. Generated videos contain raw camera pixels."
        )
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
        "agent_reference_count": len(agent_reference_map.get("references", [])),
        "vehicle_reference_count": sum(1 for row in agent_reference_map.get("references", []) if row.get("is_vehicle")),
    }
    return scene_record, clip_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--outdir", default=None, help=f"Default: <dataroot>/{DEFAULT_CLIP_SUBDIR}")
    parser.add_argument("--scene-names", default=None, help="Optional comma-separated scene names")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--overwrite-videos", action="store_true")
    parser.add_argument(
        "--keep-annotated-frames",
        action="store_true",
        help="Accepted for compatibility; v1 always writes raw videos and metadata-only annotations.",
    )
    parser.add_argument(
        "--box-line-width",
        type=int,
        default=2,
        help="Accepted for compatibility; v1 does not render boxes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else dataroot / DEFAULT_CLIP_SUBDIR
    channels = parse_channels(args.channels)

    if not dataroot.exists():
        raise FileNotFoundError(f"dataroot does not exist: {dataroot}")

    mkdir(outdir / "videos")
    mkdir(outdir / "metadata")
    mkdir(outdir / "splits")

    print("=" * 80)
    print("Raw nuScenes camera video clip builder with agent reference metadata")
    print("=" * 80)
    print(f"dataroot:    {dataroot}")
    print(f"version:     {args.version}")
    print(f"outdir:      {outdir}")
    print(f"channels:    {channels}")
    print(f"fps:         {args.fps}")
    print(f"clip_len:    {args.clip_len}")
    print(f"clip_stride: {args.clip_stride}")
    print("=" * 80)

    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=True)
    scene_filter = None
    if args.scene_names:
        scene_filter = {name.strip() for name in args.scene_names.split(",") if name.strip()}

    selected_scenes = [scene for scene in nusc.scene if scene_filter is None or scene["name"] in scene_filter]
    if args.max_scenes is not None:
        selected_scenes = selected_scenes[: args.max_scenes]
    if not selected_scenes:
        raise ValueError("No scenes selected. Check --scene-names or --max-scenes.")

    all_scene_records = []
    all_clip_entries = []
    for scene in tqdm(selected_scenes, desc="Processing scenes"):
        scene_record, clip_entries = process_scene(
            nusc=nusc,
            scene=scene,
            dataroot=dataroot,
            outdir=outdir,
            version=args.version,
            channels=channels,
            fps=args.fps,
            clip_len=args.clip_len,
            clip_stride=args.clip_stride,
            overwrite_videos=args.overwrite_videos,
            keep_annotated_frames=args.keep_annotated_frames,
            line_width=args.box_line_width,
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
