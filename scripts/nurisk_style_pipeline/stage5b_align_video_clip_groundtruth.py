#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 5b: align video clips with enhanced risk labels."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common import (
    DEFAULT_DATAROOT,
    DEFAULT_KEYFRAME_INTERVAL_SECONDS,
    format_time_key,
    mkdir,
    output_root,
)


RISK_SCALE = {
    "0": "Collision Risk",
    "1": "Extreme Risk",
    "2": "High Risk",
    "3": "Medium Risk",
    "4": "Low Risk",
    "5": "Negligible Risk",
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
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


def risk_agent_to_groundtruth(agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
    risk_analysis = agent_data.get("Risk Analysis", {})
    distance_scores = risk_analysis.get("Distance Risk Scores", {})
    ttc_scores = risk_analysis.get("TTC Risk Scores", {})
    weighting = risk_analysis.get("Weighting Logic", {})

    explanations = {
        "dominant_weight_explanation": weighting.get("Explanation", ""),
        "longitudinal_distance_explanation": distance_scores.get("Longitudinal", {}).get("Explanation", ""),
        "lateral_distance_explanation": distance_scores.get("Lateral", {}).get("Explanation", ""),
        "longitudinal_ttc_explanation": ttc_scores.get("Longitudinal", {}).get("Explanation", ""),
        "lateral_ttc_explanation": ttc_scores.get("Lateral", {}).get("Explanation", ""),
        "overall_risk_explanation": risk_analysis.get("Calculation Process", ""),
    }

    return {
        "agent_id": agent_id,
        "position": {
            "relative_direction": agent_data.get("Relative Direction", "Unknown"),
            "distance_to_collision": agent_data.get("Distance to Collision", {}),
            "time_to_collision": agent_data.get("Time to Collision", {}),
        },
        "motion": {
            "description": agent_data.get("Motion Description", "Unknown"),
            "relative_velocity": agent_data.get("Relative Velocity", {}),
            "relative_acceleration": agent_data.get("Relative Acceleration", {}),
        },
        "risk_assessment": {
            "dominant_weight": weighting.get("Weight", 0.0),
            "overall_risk_score": risk_analysis.get("Overall Risk Score", -1),
            "risk_level": risk_analysis.get("Risk Level", "Unknown"),
            "distance_risk_scores": distance_scores,
            "ttc_risk_scores": ttc_scores,
        },
        "explanations": explanations,
    }


def distance_only_groundtruth(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "distance_to_collision": agent_data.get("Distance to Collision", {}),
    }


def frame_metadata(
    frame_index: int,
    sample_token: str,
    timestamp: int,
    risk_data: Dict[str, Any],
    keyframe_interval_seconds: float,
) -> Dict[str, Any]:
    key = format_time_key(frame_index, keyframe_interval_seconds)
    frame_agents = risk_data.get(key, {})
    return {
        "frame_index": frame_index,
        "time_key": key,
        "sample_token": sample_token,
        "timestamp": timestamp,
        "total_agents": len(frame_agents),
    }


def build_agent_tracks(
    frame_indices: Sequence[int],
    sample_tokens: Sequence[str],
    timestamps: Sequence[int],
    risk_data: Dict[str, Any],
    keyframe_interval_seconds: float,
) -> Dict[str, Dict[str, Any]]:
    agents: Dict[str, Dict[str, Any]] = {}
    frame_count = len(frame_indices)
    for offset, frame_index in enumerate(frame_indices, start=1):
        frame_key = f"frame_{offset}"
        time_key = format_time_key(frame_index, keyframe_interval_seconds)
        frame_agents = risk_data.get(time_key, {})
        for agent_id, agent_data in frame_agents.items():
            agents.setdefault(agent_id, {})
            if offset < frame_count:
                agents[agent_id][frame_key] = distance_only_groundtruth(agent_data)
            else:
                agents[agent_id][frame_key] = risk_agent_to_groundtruth(agent_id, agent_data)
    return agents


def summarize_agent_tracks(
    agents: Dict[str, Dict[str, Any]],
    reference_frame_key: str,
) -> Dict[str, Any]:
    reference_scores = []
    reference_agents = 0
    distance_only_labels = 0

    for agent_frames in agents.values():
        for frame_key, frame_data in agent_frames.items():
            if frame_key == reference_frame_key:
                if "risk_assessment" in frame_data:
                    reference_agents += 1
                    score = frame_data["risk_assessment"].get("overall_risk_score", -1)
                    if score != -1:
                        reference_scores.append(score)
            elif "distance_to_collision" in frame_data:
                distance_only_labels += 1

    return {
        "min_risk_score_reference_frame": min(reference_scores) if reference_scores else None,
        "max_risk_score_reference_frame": max(reference_scores) if reference_scores else None,
        "avg_risk_score_reference_frame": (
            sum(reference_scores) / len(reference_scores) if reference_scores else None
        ),
        "risk_distribution_reference_frame": {str(i): reference_scores.count(i) for i in range(6)},
        "total_agent_tracks": len(agents),
        "total_distance_only_labels": distance_only_labels,
        "total_agents_reference_frame": reference_agents,
    }


def build_clip_entry(
    clip: Dict[str, Any],
    risk_data: Dict[str, Any],
    keyframe_interval_seconds: float,
) -> Dict[str, Any]:
    frame_indices = list(range(int(clip["start_frame_index"]), int(clip["end_frame_index"]) + 1))
    frames = [
        frame_metadata(frame_index, sample_token, timestamp, risk_data, keyframe_interval_seconds)
        for frame_index, sample_token, timestamp in zip(frame_indices, clip["sample_tokens"], clip["timestamps"])
    ]
    agents = build_agent_tracks(
        frame_indices,
        clip["sample_tokens"],
        clip["timestamps"],
        risk_data,
        keyframe_interval_seconds,
    )
    reference_frame_key = f"frame_{len(frame_indices)}"
    reference_frame = frames[-1] if frames else {
        "frame_key": None,
        "frame_index": None,
        "time_key": None,
        "total_agents": 0,
    }
    if frames:
        reference_frame = {"frame_key": reference_frame_key, **reference_frame}
    return {
        "clip_id": clip["clip_id"],
        "scene": clip["scene_name"],
        "scene_token": clip.get("scene_token"),
        "split": clip.get("split", "unknown"),
        "start_frame_index": clip["start_frame_index"],
        "end_frame_index": clip["end_frame_index"],
        "frame_indices": frame_indices,
        "sample_tokens": clip["sample_tokens"],
        "timestamps": clip["timestamps"],
        "videos": clip.get("videos", {}),
        "frames": frames,
        "agents": agents,
        "reference_frame": reference_frame,
        "summary": summarize_agent_tracks(agents, reference_frame_key),
    }


def process_scene(
    scene_manifest_path: Path,
    risk_root: Path,
    keyframe_interval_seconds: float,
) -> Optional[Dict[str, Any]]:
    manifest = load_json(scene_manifest_path)
    scene_name = manifest["scene_name"]
    risk_path = risk_root / scene_name / "risk_scores_output_enhanced.json"
    if not risk_path.exists():
        print(f"Skipping {scene_name}: missing {risk_path}")
        return None

    risk_data = load_json(risk_path)
    entries = [
        build_clip_entry(clip, risk_data, keyframe_interval_seconds)
        for clip in manifest.get("clips", [])
    ]
    output = {
        "metadata": {
            "scenario": scene_name,
            "scene_token": manifest.get("scene_token"),
            "clip_len": manifest.get("clip_len"),
            "clip_stride": manifest.get("clip_stride"),
            "fps": manifest.get("fps"),
            "data_type": "multi_camera_video_clip",
            "label_source": str(risk_path),
            "clip_manifest": str(scene_manifest_path),
            "risk_scale": RISK_SCALE,
            "keyframe_interval_seconds": keyframe_interval_seconds,
            "label_structure": "agent-centric",
            "frame_structure": "First 4 frames: distance_to_collision only; reference frame: complete current risk label",
            "total_clips": len(entries),
        },
        "entries": entries,
    }
    output_path = risk_root / scene_name / "video_clip_groundtruth.json"
    write_json(output_path, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--risk-dir", default=None, help="Default: <dataroot>/nurisk_style")
    parser.add_argument("--clip-dir", default=None, help="Default: <dataroot>/video_clip_dataset")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument(
        "--keyframe-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between nuScenes keyframes. Default is 0.5 for 2Hz keyframes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    risk_root = output_root(str(dataroot), args.risk_dir)
    clip_root = Path(args.clip_dir).expanduser().resolve() if args.clip_dir else dataroot / "video_clip_dataset"
    metadata_dir = clip_root / "metadata"

    if args.scene_name:
        scene_manifest_paths = [metadata_dir / f"{args.scene_name}_clips.json"]
    else:
        scene_manifest_paths = sorted(metadata_dir.glob("*_clips.json"))

    all_entries = []
    for manifest_path in scene_manifest_paths:
        scene_output = process_scene(manifest_path, risk_root, args.keyframe_interval_seconds)
        if scene_output is None:
            continue
        all_entries.extend(scene_output["entries"])

    jsonl_path = risk_root / "video_clip_groundtruth.jsonl"
    write_jsonl(jsonl_path, all_entries)
    print("Stage 5b done.")
    print(f"Scene groundtruth files: {len(scene_manifest_paths)}")
    print(f"Clip groundtruth entries: {len(all_entries)}")
    print(f"Global JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
