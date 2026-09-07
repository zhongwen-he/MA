#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build nuScenes camera keyframe videos and metadata manifests.

Output structure:

nuscenes_video_dataset/
├── videos/
│   ├── scene-0061/
│   │   ├── CAM_FRONT.mp4
│   │   ├── CAM_FRONT_LEFT.mp4
│   │   ├── CAM_FRONT_RIGHT.mp4
│   │   ├── CAM_BACK.mp4
│   │   ├── CAM_BACK_LEFT.mp4
│   │   └── CAM_BACK_RIGHT.mp4
├── metadata/
│   ├── scenes.json
│   ├── scene-0061_frames.json
│   └── ...
├── labels/
│   ├── scene-0061_labels.json
│   └── ...
└── splits/
    ├── train_scenes.txt
    ├── val_scenes.txt
    ├── train_videos.jsonl
    ├── val_videos.jsonl
    ├── train_clips.jsonl
    └── val_clips.jsonl
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


DEFAULT_DATAROOT = "/home/dellpro2/zhongwen/nuscenes-devkit/data/sets/nuscenes_full"

CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_txt(path: Path, lines: List[str]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def rel_to_out(path: Path, outdir: Path) -> str:
    return str(path.relative_to(outdir)).replace("\\", "/")


def get_scene_split(scene_name: str, split_scenes: Dict[str, List[str]]) -> str:
    if scene_name in set(split_scenes.get("train", [])):
        return "train"
    if scene_name in set(split_scenes.get("val", [])):
        return "val"
    if scene_name in set(split_scenes.get("mini_train", [])):
        return "mini_train"
    if scene_name in set(split_scenes.get("mini_val", [])):
        return "mini_val"
    if scene_name in set(split_scenes.get("test", [])):
        return "test"
    return "unknown"


def collect_scene_samples(nusc: NuScenes, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect keyframe samples in this scene, following sample['next'].""" 
    samples = []

    sample_token = scene["first_sample_token"]
    while sample_token != "":
        sample = nusc.get("sample", sample_token)
        samples.append(sample)

        if sample_token == scene["last_sample_token"]:
            break

        sample_token = sample["next"]

    return samples


def get_log_info(nusc: NuScenes, scene: Dict[str, Any]) -> Dict[str, Any]:
    try:
        log = nusc.get("log", scene["log_token"])
        return {
            "log_token": scene["log_token"],
            "location": log.get("location"),
            "date_captured": log.get("date_captured"),
            "vehicle": log.get("vehicle"),
        }
    except Exception:
        return {
            "log_token": scene.get("log_token"),
            "location": None,
            "date_captured": None,
            "vehicle": None,
        }


class FFmpegH264VideoWriter:
    """Small writer wrapper with the same write/release surface as cv2.VideoWriter."""

    def __init__(
        self,
        video_path: Path,
        fps: float,
        frame_size: Tuple[int, int],
        crf: int = 23,
        preset: str = "medium",
    ) -> None:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg is required for H.264 output, but it was not found in PATH.")

        width, height = frame_size
        mkdir(video_path.parent)

        command = [
            ffmpeg_path,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", preset,
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(video_path),
        ]

        self.video_path = video_path
        self.frame_size = frame_size
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if self.process.stdin is None:
            raise RuntimeError(f"Failed to open ffmpeg stdin for {video_path}")

    def write(self, image: Any) -> None:
        if image.shape[1] != self.frame_size[0] or image.shape[0] != self.frame_size[1]:
            raise ValueError(
                f"Frame size mismatch for {self.video_path}: "
                f"expected {self.frame_size}, got {(image.shape[1], image.shape[0])}"
            )

        try:
            self.process.stdin.write(image.tobytes())
        except BrokenPipeError as exc:
            stderr = self._read_stderr()
            raise RuntimeError(f"ffmpeg stopped while writing {self.video_path}: {stderr}") from exc

    def release(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()

        return_code = self.process.wait()
        if return_code != 0:
            stderr = self._read_stderr()
            raise RuntimeError(f"ffmpeg failed for {self.video_path} with code {return_code}: {stderr}")

    def _read_stderr(self) -> str:
        if self.process.stderr is None:
            return ""
        return self.process.stderr.read().decode("utf-8", errors="replace").strip()


def open_video_writer(
    video_path: Path,
    fps: float,
    frame_size: Tuple[int, int],
    codec: str,
) -> Any:
    mkdir(video_path.parent)

    codec_normalized = codec.lower()
    if codec_normalized in {"h264", "libx264", "avc1"}:
        return FFmpegH264VideoWriter(
            video_path=video_path,
            fps=fps,
            frame_size=frame_size,
        )

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {video_path}. "
            f"Try a different codec, e.g. --codec avc1 or --codec MJPG with .avi output."
        )

    return writer


def get_frame_size_from_image(
    image_path: Path,
    resize_width: int,
    resize_height: int,
) -> Tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]

    if resize_width > 0 and resize_height > 0:
        return resize_width, resize_height

    return w, h


def create_label_placeholder(
    label_path: Path,
    scene: Dict[str, Any],
    scene_split: str,
    log_info: Dict[str, Any],
    overwrite_labels: bool,
) -> None:
    if label_path.exists() and not overwrite_labels:
        return

    label_data = {
        "scene_name": scene["name"],
        "scene_token": scene["token"],
        "split": scene_split,
        "description": scene.get("description", ""),
        "location": log_info.get("location"),
        "labels": {
            "scene_risk_score": None,
            "scene_risk_level": None,
            "scene_summary": None,
            "risk_factors": [],
            "qa_pairs": [],
            "frame_labels": [],
            "clip_labels": []
        },
        "notes": (
            "This is a placeholder label file. "
            "Add project-specific risk scores, QA pairs, CoT supervision, "
            "or frame/clip-level labels here."
        )
    }

    write_json(label_path, label_data, indent=2)


def build_clips_for_video(
    scene_name: str,
    scene_token: str,
    scene_split: str,
    channel: str,
    video_rel_path: str,
    frame_manifest_rel_path: str,
    label_rel_path: str,
    frames: List[Dict[str, Any]],
    clip_len: int,
    clip_stride: int,
) -> List[Dict[str, Any]]:
    """Build clip index records for one scene + one camera video."""
    if clip_len <= 0:
        return []

    num_frames = len(frames)
    if num_frames == 0:
        return []

    starts = []

    if num_frames <= clip_len:
        starts = [0]
    else:
        starts = list(range(0, num_frames - clip_len + 1, clip_stride))
        last_start = num_frames - clip_len
        if starts[-1] != last_start:
            starts.append(last_start)

    clips = []

    for start in starts:
        end_exclusive = min(start + clip_len, num_frames)
        selected_frames = frames[start:end_exclusive]

        clip_id = f"{scene_name}_{channel}_{start:04d}_{end_exclusive - 1:04d}"

        sample_tokens = [f["sample_token"] for f in selected_frames]
        timestamps = [f["timestamp"] for f in selected_frames]

        sample_data_tokens = []
        original_filenames = []

        for f in selected_frames:
            cam_info = f["cameras"].get(channel)
            if cam_info is None:
                sample_data_tokens.append(None)
                original_filenames.append(None)
            else:
                sample_data_tokens.append(cam_info["sample_data_token"])
                original_filenames.append(cam_info["filename"])

        clips.append({
            "clip_id": clip_id,
            "scene_name": scene_name,
            "scene_token": scene_token,
            "split": scene_split,
            "camera_channel": channel,
            "video_path": video_rel_path,
            "frame_manifest": frame_manifest_rel_path,
            "label_file": label_rel_path,
            "start_frame_index": start,
            "end_frame_index": end_exclusive - 1,
            "num_frames": len(selected_frames),
            "sample_tokens": sample_tokens,
            "sample_data_tokens": sample_data_tokens,
            "timestamps": timestamps,
            "original_filenames": original_filenames,
        })

    return clips


def process_scene(
    nusc: NuScenes,
    scene: Dict[str, Any],
    scene_split: str,
    dataroot: Path,
    outdir: Path,
    channels: List[str],
    fps: float,
    resize_width: int,
    resize_height: int,
    codec: str,
    overwrite_videos: bool,
    overwrite_labels: bool,
    clip_len: int,
    clip_stride: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    scene_name = scene["name"]
    scene_token = scene["token"]

    videos_dir = outdir / "videos" / scene_name
    metadata_dir = outdir / "metadata"
    labels_dir = outdir / "labels"

    mkdir(videos_dir)
    mkdir(metadata_dir)
    mkdir(labels_dir)

    samples = collect_scene_samples(nusc, scene)
    log_info = get_log_info(nusc, scene)

    frame_manifest_path = metadata_dir / f"{scene_name}_frames.json"
    label_path = labels_dir / f"{scene_name}_labels.json"

    video_paths_abs: Dict[str, Path] = {}
    video_paths_rel: Dict[str, str] = {}

    for channel in channels:
        video_path = videos_dir / f"{channel}.mp4"
        video_paths_abs[channel] = video_path
        video_paths_rel[channel] = rel_to_out(video_path, outdir)

    create_label_placeholder(
        label_path=label_path,
        scene=scene,
        scene_split=scene_split,
        log_info=log_info,
        overwrite_labels=overwrite_labels,
    )

    label_rel_path = rel_to_out(label_path, outdir)
    frame_manifest_rel_path = rel_to_out(frame_manifest_path, outdir)

    # Initialize video writers.
    writers: Dict[str, Optional[cv2.VideoWriter]] = {}

    try:
        for channel in channels:
            video_path = video_paths_abs[channel]

            if video_path.exists() and not overwrite_videos:
                writers[channel] = None
                continue

            if len(samples) == 0:
                writers[channel] = None
                continue

            first_sample = samples[0]
            if channel not in first_sample["data"]:
                writers[channel] = None
                continue

            first_sd_token = first_sample["data"][channel]
            first_sd = nusc.get("sample_data", first_sd_token)
            first_img_path = dataroot / first_sd["filename"]

            frame_size = get_frame_size_from_image(
                image_path=first_img_path,
                resize_width=resize_width,
                resize_height=resize_height,
            )

            writers[channel] = open_video_writer(
                video_path=video_path,
                fps=fps,
                frame_size=frame_size,
                codec=codec,
            )

        # Build frame manifest while writing videos.
        frames: List[Dict[str, Any]] = []

        for frame_index, sample in enumerate(samples):
            frame_record = {
                "frame_index": frame_index,
                "timestamp": sample["timestamp"],
                "sample_token": sample["token"],
                "prev_sample_token": sample["prev"],
                "next_sample_token": sample["next"],
                "annotation_tokens": sample.get("anns", []),
                "num_annotations": len(sample.get("anns", [])),
                "cameras": {}
            }

            for channel in channels:
                if channel not in sample["data"]:
                    frame_record["cameras"][channel] = None
                    continue

                sd_token = sample["data"][channel]
                sd = nusc.get("sample_data", sd_token)
                image_path = dataroot / sd["filename"]

                camera_record = {
                    "sample_data_token": sd_token,
                    "filename": sd["filename"],
                    "timestamp": sd["timestamp"],
                    "is_key_frame": sd["is_key_frame"],
                    "width": sd.get("width"),
                    "height": sd.get("height"),
                    "ego_pose_token": sd.get("ego_pose_token"),
                    "calibrated_sensor_token": sd.get("calibrated_sensor_token"),
                    "video_path": video_paths_rel[channel],
                }

                frame_record["cameras"][channel] = camera_record

                writer = writers.get(channel)
                if writer is not None:
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise FileNotFoundError(f"Cannot read image: {image_path}")

                    if resize_width > 0 and resize_height > 0:
                        image = cv2.resize(image, (resize_width, resize_height))

                    writer.write(image)

            frames.append(frame_record)

        frame_manifest = {
            "scene_name": scene_name,
            "scene_token": scene_token,
            "split": scene_split,
            "mode": "camera_keyframes",
            "description": scene.get("description", ""),
            "num_keyframes": len(samples),
            "fps": fps,
            "channels": channels,
            "videos": video_paths_rel,
            "label_file": label_rel_path,
            "log": log_info,
            "frames": frames,
        }

        write_json(frame_manifest_path, frame_manifest, indent=2)

    finally:
        for writer in writers.values():
            if writer is not None:
                writer.release()

    scene_record = {
        "scene_name": scene_name,
        "scene_token": scene_token,
        "split": scene_split,
        "description": scene.get("description", ""),
        "num_keyframes": len(samples),
        "first_sample_token": scene.get("first_sample_token"),
        "last_sample_token": scene.get("last_sample_token"),
        "videos": video_paths_rel,
        "frame_manifest": frame_manifest_rel_path,
        "label_file": label_rel_path,
        "log": log_info,
    }

    video_entries: List[Dict[str, Any]] = []
    clip_entries: List[Dict[str, Any]] = []

    for channel in channels:
        video_entry = {
            "sample_id": f"{scene_name}_{channel}",
            "scene_name": scene_name,
            "scene_token": scene_token,
            "split": scene_split,
            "camera_channel": channel,
            "video_path": video_paths_rel[channel],
            "frame_manifest": frame_manifest_rel_path,
            "label_file": label_rel_path,
            "num_frames": len(samples),
            "fps": fps,
            "description": scene.get("description", ""),
        }

        video_entries.append(video_entry)

        channel_clips = build_clips_for_video(
            scene_name=scene_name,
            scene_token=scene_token,
            scene_split=scene_split,
            channel=channel,
            video_rel_path=video_paths_rel[channel],
            frame_manifest_rel_path=frame_manifest_rel_path,
            label_rel_path=label_rel_path,
            frames=frame_manifest["frames"],
            clip_len=clip_len,
            clip_stride=clip_stride,
        )

        clip_entries.extend(channel_clips)

    return scene_record, video_entries, clip_entries


def parse_channels(channels_arg: str) -> List[str]:
    if channels_arg.strip().lower() == "all":
        return CAMERA_CHANNELS

    channels = [c.strip() for c in channels_arg.split(",") if c.strip()]
    invalid = [c for c in channels if c not in CAMERA_CHANNELS]

    if invalid:
        raise ValueError(f"Invalid camera channels: {invalid}. Valid channels: {CAMERA_CHANNELS}")

    return channels


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataroot",
        default=DEFAULT_DATAROOT,
        help="nuScenes root directory."
    )

    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes version, e.g. v1.0-trainval."
    )

    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory. Default: <dataroot>/nuscenes_video_dataset. "
            "For your setup this becomes "
            "/home/dellpro2/zhongwen/nuscenes-devkit/data/sets/nuscenes_full/nuscenes_video_dataset"
        )
    )

    parser.add_argument(
        "--channels",
        default="all",
        help="Camera channels to export. Use 'all' or comma-separated list, e.g. CAM_FRONT,CAM_BACK."
    )

    parser.add_argument(
        "--only-split",
        default="all",
        choices=["all", "train", "val"],
        help="Export all/train/val scenes."
    )

    parser.add_argument(
        "--scene-names",
        default=None,
        help="Optional comma-separated scene names, e.g. scene-0061,scene-0103."
    )

    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional limit for testing, e.g. --max-scenes 2."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Output FPS for keyframe videos. nuScenes keyframes are approximately 2 FPS."
    )

    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Resize output video width. 0 means keep original size."
    )

    parser.add_argument(
        "--resize-height",
        type=int,
        default=0,
        help="Resize output video height. 0 means keep original size."
    )

    parser.add_argument(
        "--codec",
        default="libx264",
        help="Video codec. Default: libx264 for H.264 MP4 output. Use mp4v for the old OpenCV MPEG-4 writer."
    )

    parser.add_argument(
        "--overwrite-videos",
        action="store_true",
        help="Overwrite existing mp4 videos."
    )

    parser.add_argument(
        "--overwrite-labels",
        action="store_true",
        help="Overwrite existing placeholder label files. Be careful if you have already edited labels."
    )

    parser.add_argument(
        "--clip-len",
        type=int,
        default=16,
        help="Clip length in frames for train_clips/val_clips JSONL. Use 0 to disable clip generation."
    )

    parser.add_argument(
        "--clip-stride",
        type=int,
        default=8,
        help="Clip stride in frames."
    )

    args = parser.parse_args()

    dataroot = Path(args.dataroot).expanduser().resolve()

    if args.outdir is None:
        outdir = dataroot / "nuscenes_video_dataset"
    else:
        outdir = Path(args.outdir).expanduser().resolve()

    channels = parse_channels(args.channels)

    print("=" * 80)
    print("nuScenes camera video builder")
    print("=" * 80)
    print(f"dataroot: {dataroot}")
    print(f"version:  {args.version}")
    print(f"outdir:   {outdir}")
    print(f"channels: {channels}")
    print(f"fps:      {args.fps}")
    print(f"codec:    {args.codec}")
    print("=" * 80)

    if not dataroot.exists():
        raise FileNotFoundError(f"dataroot does not exist: {dataroot}")

    for required in ["samples", "v1.0-trainval"]:
        required_path = dataroot / required
        if not required_path.exists():
            raise FileNotFoundError(f"Required path missing: {required_path}")

    mkdir(outdir / "videos")
    mkdir(outdir / "metadata")
    mkdir(outdir / "labels")
    mkdir(outdir / "splits")

    split_scenes = create_splits_scenes()

    print("Loading nuScenes metadata...")
    nusc = NuScenes(
        version=args.version,
        dataroot=str(dataroot),
        verbose=True
    )

    scene_name_filter = None
    if args.scene_names:
        scene_name_filter = set([s.strip() for s in args.scene_names.split(",") if s.strip()])

    selected_scenes = []

    for scene in nusc.scene:
        scene_name = scene["name"]
        scene_split = get_scene_split(scene_name, split_scenes)

        if args.only_split != "all" and scene_split != args.only_split:
            continue

        if scene_name_filter is not None and scene_name not in scene_name_filter:
            continue

        selected_scenes.append((scene, scene_split))

    if args.max_scenes is not None:
        selected_scenes = selected_scenes[:args.max_scenes]

    print(f"Selected scenes: {len(selected_scenes)}")

    all_scene_records: List[Dict[str, Any]] = []
    all_video_entries: List[Dict[str, Any]] = []
    all_clip_entries: List[Dict[str, Any]] = []

    for scene, scene_split in tqdm(selected_scenes, desc="Processing scenes"):
        scene_record, video_entries, clip_entries = process_scene(
            nusc=nusc,
            scene=scene,
            scene_split=scene_split,
            dataroot=dataroot,
            outdir=outdir,
            channels=channels,
            fps=args.fps,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
            codec=args.codec,
            overwrite_videos=args.overwrite_videos,
            overwrite_labels=args.overwrite_labels,
            clip_len=args.clip_len,
            clip_stride=args.clip_stride,
        )

        all_scene_records.append(scene_record)
        all_video_entries.extend(video_entries)
        all_clip_entries.extend(clip_entries)

    # Save global scene manifest.
    scenes_json_path = outdir / "metadata" / "scenes.json"
    write_json(scenes_json_path, all_scene_records, indent=2)

    # Split files.
    train_scene_names = [r["scene_name"] for r in all_scene_records if r["split"] == "train"]
    val_scene_names = [r["scene_name"] for r in all_scene_records if r["split"] == "val"]

    train_video_entries = [r for r in all_video_entries if r["split"] == "train"]
    val_video_entries = [r for r in all_video_entries if r["split"] == "val"]

    train_clip_entries = [r for r in all_clip_entries if r["split"] == "train"]
    val_clip_entries = [r for r in all_clip_entries if r["split"] == "val"]

    write_txt(outdir / "splits" / "train_scenes.txt", train_scene_names)
    write_txt(outdir / "splits" / "val_scenes.txt", val_scene_names)

    write_jsonl(outdir / "splits" / "train_videos.jsonl", train_video_entries)
    write_jsonl(outdir / "splits" / "val_videos.jsonl", val_video_entries)

    write_jsonl(outdir / "splits" / "train_clips.jsonl", train_clip_entries)
    write_jsonl(outdir / "splits" / "val_clips.jsonl", val_clip_entries)

    print("\nDone.")
    print(f"Output directory: {outdir}")
    print(f"Scenes exported:  {len(all_scene_records)}")
    print(f"Video entries:    {len(all_video_entries)}")
    print(f"Clip entries:     {len(all_clip_entries)}")
    print("\nImportant files:")
    print(f"  {outdir / 'metadata' / 'scenes.json'}")
    print(f"  {outdir / 'splits' / 'train_videos.jsonl'}")
    print(f"  {outdir / 'splits' / 'val_videos.jsonl'}")
    print(f"  {outdir / 'splits' / 'train_clips.jsonl'}")
    print(f"  {outdir / 'splits' / 'val_clips.jsonl'}")


if __name__ == "__main__":
    main()
