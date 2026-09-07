#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build keyframe-aligned nuScenes camera video clips.

Default behavior follows the NuRisk-style temporal window used by the VQA
pipeline: 5 keyframes per clip with stride 1. Since nuScenes keyframes are
approximately 2 Hz, a 5-frame clip spans about 2 seconds.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    return "unknown"


def collect_scene_samples(nusc: NuScenes, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = []
    sample_token = scene["first_sample_token"]
    while sample_token:
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


def open_video_writer(video_path: Path, fps: float, frame_size: Tuple[int, int], codec: str) -> Any:
    codec_normalized = codec.lower()
    if codec_normalized in {"h264", "libx264", "avc1"}:
        return FFmpegH264VideoWriter(video_path=video_path, fps=fps, frame_size=frame_size)

    mkdir(video_path.parent)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {video_path}")
    return writer


def get_frame_size_from_image(image_path: Path, resize_width: int, resize_height: int) -> Tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = image.shape[:2]
    if resize_width > 0 and resize_height > 0:
        return resize_width, resize_height
    return w, h


def parse_channels(channels_arg: str) -> List[str]:
    if channels_arg.strip().lower() == "all":
        return CAMERA_CHANNELS
    channels = [c.strip() for c in channels_arg.split(",") if c.strip()]
    invalid = [c for c in channels if c not in CAMERA_CHANNELS]
    if invalid:
        raise ValueError(f"Invalid camera channels: {invalid}. Valid channels: {CAMERA_CHANNELS}")
    return channels


def build_clip_starts(num_frames: int, clip_len: int, clip_stride: int, drop_short: bool) -> List[int]:
    if clip_len <= 0:
        raise ValueError("--clip-len must be positive.")
    if clip_stride <= 0:
        raise ValueError("--clip-stride must be positive.")
    if num_frames < clip_len:
        return [] if drop_short else [0]
    return list(range(0, num_frames - clip_len + 1, clip_stride))


def build_frame_records(
    nusc: NuScenes,
    samples: List[Dict[str, Any]],
    channels: List[str],
) -> List[Dict[str, Any]]:
    frames = []
    for frame_index, sample in enumerate(samples):
        frame_record = {
            "frame_index": frame_index,
            "timestamp": sample["timestamp"],
            "sample_token": sample["token"],
            "prev_sample_token": sample["prev"],
            "next_sample_token": sample["next"],
            "annotation_tokens": sample.get("anns", []),
            "num_annotations": len(sample.get("anns", [])),
            "cameras": {},
        }

        for channel in channels:
            if channel not in sample["data"]:
                frame_record["cameras"][channel] = None
                continue
            sd_token = sample["data"][channel]
            sd = nusc.get("sample_data", sd_token)
            frame_record["cameras"][channel] = {
                "sample_data_token": sd_token,
                "filename": sd["filename"],
                "timestamp": sd["timestamp"],
                "is_key_frame": sd["is_key_frame"],
                "width": sd.get("width"),
                "height": sd.get("height"),
                "ego_pose_token": sd.get("ego_pose_token"),
                "calibrated_sensor_token": sd.get("calibrated_sensor_token"),
            }
        frames.append(frame_record)
    return frames


def write_clip_video(
    dataroot: Path,
    clip_path: Path,
    channel: str,
    selected_frames: List[Dict[str, Any]],
    fps: float,
    resize_width: int,
    resize_height: int,
    codec: str,
    overwrite: bool,
) -> bool:
    if clip_path.exists() and not overwrite:
        return False

    first_cam = selected_frames[0]["cameras"].get(channel)
    if first_cam is None:
        return False

    frame_size = get_frame_size_from_image(
        image_path=dataroot / first_cam["filename"],
        resize_width=resize_width,
        resize_height=resize_height,
    )
    writer = open_video_writer(clip_path, fps=fps, frame_size=frame_size, codec=codec)
    try:
        for frame in selected_frames:
            cam = frame["cameras"].get(channel)
            if cam is None:
                raise RuntimeError(f"Missing {channel} frame for clip {clip_path}")
            image = cv2.imread(str(dataroot / cam["filename"]))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {dataroot / cam['filename']}")
            if resize_width > 0 and resize_height > 0:
                image = cv2.resize(image, (resize_width, resize_height))
            writer.write(image)
    finally:
        writer.release()
    return True


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
    clip_len: int,
    clip_stride: int,
    drop_short: bool,
    overwrite_videos: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scene_name = scene["name"]
    scene_token = scene["token"]
    samples = collect_scene_samples(nusc, scene)
    frames = build_frame_records(nusc, samples, channels)
    log_info = get_log_info(nusc, scene)

    starts = build_clip_starts(
        num_frames=len(frames),
        clip_len=clip_len,
        clip_stride=clip_stride,
        drop_short=drop_short,
    )

    clip_entries: List[Dict[str, Any]] = []
    scene_clip_records: List[Dict[str, Any]] = []

    for start in starts:
        end_exclusive = min(start + clip_len, len(frames))
        end = end_exclusive - 1
        selected_frames = frames[start:end_exclusive]
        clip_id = f"{scene_name}_{start:04d}_{end:04d}"
        timestamp_start = selected_frames[0]["timestamp"]
        timestamp_end = selected_frames[-1]["timestamp"]
        duration_seconds = (timestamp_end - timestamp_start) / 1e6

        videos: Dict[str, str] = {}
        channel_records = {}

        for channel in channels:
            clip_filename = f"{scene_name}_{channel}_{start:04d}_{end:04d}.mp4"
            clip_path = outdir / "videos" / scene_name / channel / clip_filename
            wrote = write_clip_video(
                dataroot=dataroot,
                clip_path=clip_path,
                channel=channel,
                selected_frames=selected_frames,
                fps=fps,
                resize_width=resize_width,
                resize_height=resize_height,
                codec=codec,
                overwrite=overwrite_videos,
            )
            clip_rel_path = rel_to_out(clip_path, outdir)
            videos[channel] = clip_rel_path
            channel_records[channel] = {
                "camera_channel": channel,
                "clip_path": clip_rel_path,
                "wrote_video": wrote,
                "sample_data_tokens": [
                    frame["cameras"][channel]["sample_data_token"]
                    if frame["cameras"].get(channel) is not None else None
                    for frame in selected_frames
                ],
                "original_filenames": [
                    frame["cameras"][channel]["filename"]
                    if frame["cameras"].get(channel) is not None else None
                    for frame in selected_frames
                ],
            }

            clip_entries.append({
                "clip_id": clip_id,
                "scene_name": scene_name,
                "scene_token": scene_token,
                "split": scene_split,
                "camera_channel": channel,
                "clip_path": clip_rel_path,
                "start_frame_index": start,
                "end_frame_index": end,
                "num_frames": len(selected_frames),
                "fps": fps,
                "duration_seconds": duration_seconds,
                "sample_tokens": [frame["sample_token"] for frame in selected_frames],
                "sample_data_tokens": channel_records[channel]["sample_data_tokens"],
                "timestamps": [frame["timestamp"] for frame in selected_frames],
                "original_filenames": channel_records[channel]["original_filenames"],
            })

        scene_clip_records.append({
            "clip_id": clip_id,
            "scene_name": scene_name,
            "scene_token": scene_token,
            "split": scene_split,
            "start_frame_index": start,
            "end_frame_index": end,
            "num_frames": len(selected_frames),
            "fps": fps,
            "duration_seconds": duration_seconds,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "sample_tokens": [frame["sample_token"] for frame in selected_frames],
            "timestamps": [frame["timestamp"] for frame in selected_frames],
            "videos": videos,
            "channels": channel_records,
        })

    scene_manifest_path = outdir / "metadata" / f"{scene_name}_clips.json"
    scene_manifest = {
        "scene_name": scene_name,
        "scene_token": scene_token,
        "split": scene_split,
        "description": scene.get("description", ""),
        "num_keyframes": len(frames),
        "num_clips": len(scene_clip_records),
        "clip_len": clip_len,
        "clip_stride": clip_stride,
        "fps": fps,
        "channels": channels,
        "log": log_info,
        "clips": scene_clip_records,
    }
    write_json(scene_manifest_path, scene_manifest, indent=2)

    scene_record = {
        "scene_name": scene_name,
        "scene_token": scene_token,
        "split": scene_split,
        "description": scene.get("description", ""),
        "num_keyframes": len(frames),
        "num_clips": len(scene_clip_records),
        "first_sample_token": scene.get("first_sample_token"),
        "last_sample_token": scene.get("last_sample_token"),
        "clip_manifest": rel_to_out(scene_manifest_path, outdir),
        "log": log_info,
    }

    return scene_record, clip_entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT, help="nuScenes full dataroot.")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes metadata version.")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Default: <dataroot>/video_clip_dataset.",
    )
    parser.add_argument("--channels", default="all", help="all or comma-separated camera channels.")
    parser.add_argument("--only-split", default="all", choices=["all", "train", "val"], help="Scene split filter.")
    parser.add_argument("--scene-names", default=None, help="Optional comma-separated scene names.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--fps", type=float, default=2.0, help="Output FPS. Keyframes are approximately 2 Hz.")
    parser.add_argument("--clip-len", type=int, default=5, help="Frames per clip. Default 5, about 2 seconds.")
    parser.add_argument("--clip-stride", type=int, default=1, help="Sliding window stride. Default 1.")
    parser.add_argument("--drop-short", action="store_true", default=True, help="Drop scenes shorter than clip length.")
    parser.add_argument("--keep-short", dest="drop_short", action="store_false", help="Keep short scenes as one shorter clip.")
    parser.add_argument("--resize-width", type=int, default=0, help="Resize output width. 0 keeps original.")
    parser.add_argument("--resize-height", type=int, default=0, help="Resize output height. 0 keeps original.")
    parser.add_argument("--codec", default="libx264", help="Default libx264 H.264 MP4. Use mp4v for OpenCV MPEG-4.")
    parser.add_argument("--overwrite-videos", action="store_true", help="Overwrite existing clip mp4 files.")
    args = parser.parse_args()

    dataroot = Path(args.dataroot).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else dataroot / "video_clip_dataset"
    channels = parse_channels(args.channels)

    if not dataroot.exists():
        raise FileNotFoundError(f"dataroot does not exist: {dataroot}")
    for required in ["samples", "v1.0-trainval"]:
        if not (dataroot / required).exists():
            raise FileNotFoundError(f"Required path missing: {dataroot / required}")

    mkdir(outdir / "videos")
    mkdir(outdir / "metadata")
    mkdir(outdir / "splits")

    print("=" * 80)
    print("nuScenes camera video clip builder")
    print("=" * 80)
    print(f"dataroot:    {dataroot}")
    print(f"version:     {args.version}")
    print(f"outdir:      {outdir}")
    print(f"channels:    {channels}")
    print(f"fps:         {args.fps}")
    print(f"clip_len:    {args.clip_len}")
    print(f"clip_stride: {args.clip_stride}")
    print(f"codec:       {args.codec}")
    print("=" * 80)

    split_scenes = create_splits_scenes()
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=True)

    scene_filter = None
    if args.scene_names:
        scene_filter = {name.strip() for name in args.scene_names.split(",") if name.strip()}

    selected_scenes = []
    for scene in nusc.scene:
        scene_name = scene["name"]
        scene_split = get_scene_split(scene_name, split_scenes)
        if args.only_split != "all" and scene_split != args.only_split:
            continue
        if scene_filter is not None and scene_name not in scene_filter:
            continue
        selected_scenes.append((scene, scene_split))

    if args.max_scenes is not None:
        selected_scenes = selected_scenes[:args.max_scenes]

    all_scene_records: List[Dict[str, Any]] = []
    all_clip_entries: List[Dict[str, Any]] = []

    for scene, scene_split in tqdm(selected_scenes, desc="Processing scenes"):
        scene_record, clip_entries = process_scene(
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
            clip_len=args.clip_len,
            clip_stride=args.clip_stride,
            drop_short=args.drop_short,
            overwrite_videos=args.overwrite_videos,
        )
        all_scene_records.append(scene_record)
        all_clip_entries.extend(clip_entries)

    write_json(outdir / "metadata" / "scenes.json", all_scene_records, indent=2)
    write_jsonl(outdir / "metadata" / "clips.jsonl", all_clip_entries)

    train_scene_names = [row["scene_name"] for row in all_scene_records if row["split"] == "train"]
    val_scene_names = [row["scene_name"] for row in all_scene_records if row["split"] == "val"]
    train_clips = [row for row in all_clip_entries if row["split"] == "train"]
    val_clips = [row for row in all_clip_entries if row["split"] == "val"]

    write_txt(outdir / "splits" / "train_scenes.txt", train_scene_names)
    write_txt(outdir / "splits" / "val_scenes.txt", val_scene_names)
    write_jsonl(outdir / "splits" / "train_clips.jsonl", train_clips)
    write_jsonl(outdir / "splits" / "val_clips.jsonl", val_clips)

    print("\nDone.")
    print(f"Output directory: {outdir}")
    print(f"Scenes exported:  {len(all_scene_records)}")
    print(f"Clip videos:      {len(all_clip_entries)}")
    print("\nImportant files:")
    print(f"  {outdir / 'metadata' / 'scenes.json'}")
    print(f"  {outdir / 'metadata' / 'clips.jsonl'}")
    print(f"  {outdir / 'splits' / 'train_clips.jsonl'}")
    print(f"  {outdir / 'splits' / 'val_clips.jsonl'}")


if __name__ == "__main__":
    main()
