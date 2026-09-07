#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 5a: build 5-keyframe annotated camera video clips.

This stage preserves the existing NuRisk-style clip layout while changing the
video pixels: every camera keyframe is rendered with nuScenes 3D bounding boxes
and scene-local readable agent ids (A001, A002, ...). Raw nuScenes ids remain in
the manifest for exact joins with Stage 1-4 risk labels.
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

import cv2
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

from agent_aliases import build_scene_agent_alias_map, risk_agent_id
from common import DEFAULT_DATAROOT, mkdir


DEFAULT_CLIP_SUBDIR = "annotated_video_clip_dataset"

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


def draw_label(image: np.ndarray, label: str, x: int, y: int, color_bgr: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
    text_w, text_h = text_size
    x = max(0, min(x, image.shape[1] - text_w - 8))
    y = max(text_h + 5, min(y, image.shape[0] - baseline - 3))
    bg1 = (x, y - text_h - baseline - 5)
    bg2 = (x + text_w + 8, y + baseline + 3)
    cv2.rectangle(image, bg1, bg2, (0, 0, 0), thickness=-1)
    cv2.putText(image, label, (x + 4, y), font, scale, color_bgr, thickness, cv2.LINE_AA)


def annotate_camera_frame(
    nusc: NuScenes,
    sample_data_token: str,
    agent_alias_map: Dict[str, Any],
    output_path: Path,
    line_width: int,
) -> Dict[str, Any]:
    image_path, boxes, camera_intrinsic = nusc.get_sample_data(
        sample_data_token, box_vis_level=BoxVisibility.ANY
    )
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    instance_to_alias = agent_alias_map["instance_token_to_agent_id"]
    annotations = []

    for box in boxes:
        ann = nusc.get("sample_annotation", box.token)
        instance_token = ann["instance_token"]
        display_id = instance_to_alias[instance_token]
        rgb = nusc.explorer.get_color(box.name)
        color_bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

        box.render_cv2(image, view=camera_intrinsic, normalize=True, colors=(rgb, rgb, rgb), linewidth=line_width)
        bbox = projected_2d_bbox(box, camera_intrinsic, width, height)
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 1)
            draw_label(image, display_id, x1, max(0, y1 - 4), color_bgr)

        annotations.append(
            {
                "agent_id": display_id,
                "display_agent_id": display_id,
                "scene_agent_id": f"{agent_alias_map['scene_name']}_{display_id}",
                "raw_agent_id": risk_agent_id(instance_token),
                "risk_agent_id": risk_agent_id(instance_token),
                "instance_token": instance_token,
                "sample_annotation_token": box.token,
                "category_name": ann["category_name"],
                "bbox_2d_projected": bbox,
                "num_lidar_pts": ann.get("num_lidar_pts"),
                "num_radar_pts": ann.get("num_radar_pts"),
                "visibility_token": ann.get("visibility_token"),
            }
        )

    mkdir(output_path.parent)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write annotated frame: {output_path}")

    return {
        "sample_data_token": sample_data_token,
        "annotated_frame_path": str(output_path),
        "num_visible_boxes": len(annotations),
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


def annotate_scene_frames(
    nusc: NuScenes,
    frames: Sequence[Dict[str, Any]],
    channels: Sequence[str],
    agent_alias_map: Dict[str, Any],
    frame_root: Path,
    keep_annotated_frames: bool,
    line_width: int,
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, Any]]]:
    frame_paths: Dict[str, Path] = {}
    frame_annotations: Dict[str, Dict[str, Any]] = {}
    for frame in frames:
        for channel in channels:
            cam = frame["cameras"].get(channel)
            if cam is None:
                continue
            sd_token = cam["sample_data_token"]
            output_path = frame_root / channel / f"{frame['frame_index']:06d}_{sd_token}.jpg"
            record = annotate_camera_frame(nusc, sd_token, agent_alias_map, output_path, line_width)
            record.update(
                {
                    "frame_index": frame["frame_index"],
                    "sample_token": frame["sample_token"],
                    "timestamp": frame["timestamp"],
                    "camera_channel": channel,
                    "original_filename": cam["filename"],
                    "annotated_frame": str(output_path) if keep_annotated_frames else None,
                }
            )
            frame_paths[sd_token] = output_path
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
    agent_alias_map = build_scene_agent_alias_map(scene_name, samples, ann_by_token)

    scene_clip_records = []
    clip_entries = []
    with tempfile.TemporaryDirectory(prefix=f"annotated_{scene_name}_") as tmp:
        temp_frame_root = Path(tmp) / "frames" / scene_name
        permanent_frame_root = outdir / "annotated_frames" / scene_name
        frame_root = permanent_frame_root if keep_annotated_frames else temp_frame_root
        scene_frame_paths, scene_frame_annotations = annotate_scene_frames(
            nusc=nusc,
            frames=frames,
            channels=channels,
            agent_alias_map=agent_alias_map,
            frame_root=frame_root,
            keep_annotated_frames=keep_annotated_frames,
            line_width=line_width,
        )

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
                    "annotation_counts": [
                        scene_frame_annotations[frame["sample_data_token"]]["num_visible_boxes"]
                        for frame in channel_frames
                        if frame is not None
                    ],
                }
                if keep_annotated_frames:
                    channel_records[channel]["annotated_frames"] = [
                        rel_to_root(scene_frame_paths[frame["sample_data_token"]], outdir)
                        for frame in channel_frames
                        if frame is not None
                    ]
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
                        "agent_alias_map": agent_alias_map,
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
                    "agent_alias_map": agent_alias_map,
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
            "annotation_mode": "nuScenes 3D boxes plus scene-local display agent ids",
            "agent_alias_map": agent_alias_map,
            "clips": scene_clip_records,
            "frame_annotations": list(scene_frame_annotations.values()),
        }
        if not keep_annotated_frames:
            scene_manifest["frame_annotation_note"] = (
                "Annotated frames were temporary encoder inputs. Use videos and frame_annotations metadata."
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
        "agent_alias_count": len(agent_alias_map.get("aliases", [])),
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
    parser.add_argument("--keep-annotated-frames", action="store_true")
    parser.add_argument("--box-line-width", type=int, default=2)
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
    print("Annotated nuScenes camera video clip builder")
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
