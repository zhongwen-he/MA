#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build CSV tables directly from nuScenes raw trainval metadata.

This script uses the same scene/keyframe physical extraction logic as
scripts/build_physical.py, but writes flattened CSV files instead of JSON/JSONL
physical files.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from build_physical import (
    CAMERA_CHANNELS,
    DEFAULT_DATAROOT,
    collect_scene_samples,
    get_scene_split,
    process_scene,
    rel_to_root,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


SCENES_FIELDS = [
    "scene_name",
    "scene_token",
    "split",
    "description",
    "location",
    "log_token",
    "first_sample_token",
    "last_sample_token",
    "num_keyframes",
    "physical_file",
    "frame_manifest",
    "video_CAM_FRONT",
    "video_CAM_FRONT_LEFT",
    "video_CAM_FRONT_RIGHT",
    "video_CAM_BACK",
    "video_CAM_BACK_LEFT",
    "video_CAM_BACK_RIGHT",
]

EGO_FIELDS = [
    "scene_name",
    "scene_token",
    "frame_index",
    "sample_token",
    "timestamp",
    "ego_pose_token",
    "reference_sample_data_token",
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_qw",
    "ego_qx",
    "ego_qy",
    "ego_qz",
    "ego_yaw",
    "ego_vx",
    "ego_vy",
    "ego_speed",
    "ego_ax",
    "ego_ay",
    "ego_acceleration_norm",
    "ego_yaw_rate",
]

AGENT_FIELDS = [
    "scene_name",
    "scene_token",
    "frame_index",
    "sample_token",
    "timestamp",
    "annotation_token",
    "instance_token",
    "category_name",
    "attribute_names",
    "visibility_token",
    "agent_x",
    "agent_y",
    "agent_z",
    "box_width",
    "box_length",
    "box_height",
    "agent_qw",
    "agent_qx",
    "agent_qy",
    "agent_qz",
    "agent_yaw",
    "agent_vx",
    "agent_vy",
    "agent_speed",
    "agent_ax",
    "agent_ay",
    "agent_acceleration_norm",
    "agent_yaw_rate",
    "num_lidar_pts",
    "num_radar_pts",
    "prev_annotation_token",
    "next_annotation_token",
]

RELATIVE_FIELDS = [
    "scene_name",
    "scene_token",
    "frame_index",
    "sample_token",
    "timestamp",
    "annotation_token",
    "agent_instance_token",
    "category_name",
    "relative_x_forward_m",
    "relative_y_left_m",
    "relative_z_up_m",
    "relative_distance_m",
    "bearing_rad",
    "relative_vx_forward_mps",
    "relative_vy_left_mps",
    "relative_speed_mps",
    "closing_speed_mps",
    "relative_ax_forward_mps2",
    "relative_ay_left_mps2",
    "longitudinal_distance_m",
    "lateral_distance_m",
]

RISK_FIELDS = [
    "scene_name",
    "scene_token",
    "frame_index",
    "sample_token",
    "timestamp",
    "annotation_token",
    "agent_instance_token",
    "category_name",
    "dtc_longitudinal_m",
    "dtc_lateral_m",
    "dtc_euclidean_m",
    "ttc_longitudinal_s",
    "ttc_lateral_s",
    "min_ttc_s",
    "dtc_risk",
    "ttc_risk",
    "combined_risk_score",
    "risk_level",
    "risk_reason",
]

TRAINING_INDEX_FIELDS = [
    "scene_name",
    "scene_token",
    "split",
    "frame_index",
    "sample_token",
    "timestamp",
    "target_annotation_token",
    "target_agent_id",
    "target_category_name",
    "video_CAM_FRONT",
    "video_CAM_FRONT_LEFT",
    "video_CAM_FRONT_RIGHT",
    "video_CAM_BACK",
    "video_CAM_BACK_LEFT",
    "video_CAM_BACK_RIGHT",
    "optional_lidar_path",
    "ego_state_json",
    "target_agent_state_json",
    "target_relative_state_json",
    "target_risk_metrics_json",
    "neighbor_agent_states_json",
]


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def vector_get(values: List[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def open_writer(path: Path, fieldnames: List[str]) -> Any:
    mkdir(path.parent)
    f = open(path, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return f, writer


def scene_to_csv_row(scene_record: Dict[str, Any]) -> Dict[str, Any]:
    videos = scene_record.get("videos", {})
    row = {field: scene_record.get(field) for field in SCENES_FIELDS}
    for channel in CAMERA_CHANNELS:
        row[f"video_{channel}"] = videos.get(channel, "")
    return row


def ego_to_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    position = row.get("position_global", [])
    quat = row.get("rotation_quaternion", [])
    velocity = row.get("velocity_global", [])
    acceleration = row.get("acceleration_global", [])
    return {
        "scene_name": row.get("scene_name"),
        "scene_token": row.get("scene_token"),
        "frame_index": row.get("frame_index"),
        "sample_token": row.get("sample_token"),
        "timestamp": row.get("timestamp"),
        "ego_pose_token": row.get("ego_pose_token"),
        "reference_sample_data_token": row.get("reference_sample_data_token"),
        "ego_x": vector_get(position, 0),
        "ego_y": vector_get(position, 1),
        "ego_z": vector_get(position, 2),
        "ego_qw": vector_get(quat, 0),
        "ego_qx": vector_get(quat, 1),
        "ego_qy": vector_get(quat, 2),
        "ego_qz": vector_get(quat, 3),
        "ego_yaw": row.get("yaw"),
        "ego_vx": vector_get(velocity, 0),
        "ego_vy": vector_get(velocity, 1),
        "ego_speed": row.get("speed"),
        "ego_ax": vector_get(acceleration, 0),
        "ego_ay": vector_get(acceleration, 1),
        "ego_acceleration_norm": row.get("acceleration_norm"),
        "ego_yaw_rate": row.get("yaw_rate"),
    }


def agent_to_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    position = row.get("position_global", [])
    size = row.get("size", [])
    quat = row.get("rotation_quaternion", [])
    velocity = row.get("velocity_global", [])
    acceleration = row.get("acceleration_global", [])
    return {
        "scene_name": row.get("scene_name"),
        "scene_token": row.get("scene_token"),
        "frame_index": row.get("frame_index"),
        "sample_token": row.get("sample_token"),
        "timestamp": row.get("timestamp"),
        "annotation_token": row.get("annotation_token"),
        "instance_token": row.get("instance_token"),
        "category_name": row.get("category_name"),
        "attribute_names": "|".join(row.get("attribute_names", [])),
        "visibility_token": row.get("visibility_token"),
        "agent_x": vector_get(position, 0),
        "agent_y": vector_get(position, 1),
        "agent_z": vector_get(position, 2),
        "box_width": vector_get(size, 0),
        "box_length": vector_get(size, 1),
        "box_height": vector_get(size, 2),
        "agent_qw": vector_get(quat, 0),
        "agent_qx": vector_get(quat, 1),
        "agent_qy": vector_get(quat, 2),
        "agent_qz": vector_get(quat, 3),
        "agent_yaw": row.get("yaw"),
        "agent_vx": vector_get(velocity, 0),
        "agent_vy": vector_get(velocity, 1),
        "agent_speed": row.get("speed"),
        "agent_ax": vector_get(acceleration, 0),
        "agent_ay": vector_get(acceleration, 1),
        "agent_acceleration_norm": row.get("acceleration_norm"),
        "agent_yaw_rate": row.get("yaw_rate"),
        "num_lidar_pts": row.get("num_lidar_pts"),
        "num_radar_pts": row.get("num_radar_pts"),
        "prev_annotation_token": row.get("prev_annotation_token"),
        "next_annotation_token": row.get("next_annotation_token"),
    }


def relative_to_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rel_pos = row.get("relative_position_ego", {})
    rel_vel = row.get("relative_velocity_ego", {})
    rel_acc = row.get("relative_acceleration_ego", {})
    return {
        "scene_name": row.get("scene_name"),
        "scene_token": row.get("scene_token"),
        "frame_index": row.get("frame_index"),
        "sample_token": row.get("sample_token"),
        "timestamp": row.get("timestamp"),
        "annotation_token": row.get("annotation_token"),
        "agent_instance_token": row.get("agent_instance_token"),
        "category_name": row.get("category_name"),
        "relative_x_forward_m": rel_pos.get("x_forward_m"),
        "relative_y_left_m": rel_pos.get("y_left_m"),
        "relative_z_up_m": rel_pos.get("z_up_m"),
        "relative_distance_m": rel_pos.get("distance_m"),
        "bearing_rad": rel_pos.get("bearing_rad"),
        "relative_vx_forward_mps": rel_vel.get("vx_forward_mps"),
        "relative_vy_left_mps": rel_vel.get("vy_left_mps"),
        "relative_speed_mps": rel_vel.get("relative_speed_mps"),
        "closing_speed_mps": rel_vel.get("closing_speed_mps"),
        "relative_ax_forward_mps2": rel_acc.get("ax_forward_mps2"),
        "relative_ay_left_mps2": rel_acc.get("ay_left_mps2"),
        "longitudinal_distance_m": row.get("longitudinal_distance_m"),
        "lateral_distance_m": row.get("lateral_distance_m"),
    }


def risk_to_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    dtc = row.get("dtc", {})
    ttc = row.get("ttc", {})
    risk = row.get("risk", {})
    return {
        "scene_name": row.get("scene_name"),
        "scene_token": row.get("scene_token"),
        "frame_index": row.get("frame_index"),
        "sample_token": row.get("sample_token"),
        "timestamp": row.get("timestamp"),
        "annotation_token": row.get("annotation_token"),
        "agent_instance_token": row.get("agent_instance_token"),
        "category_name": row.get("category_name"),
        "dtc_longitudinal_m": dtc.get("longitudinal_m"),
        "dtc_lateral_m": dtc.get("lateral_m"),
        "dtc_euclidean_m": dtc.get("euclidean_m"),
        "ttc_longitudinal_s": ttc.get("longitudinal_s"),
        "ttc_lateral_s": ttc.get("lateral_s"),
        "min_ttc_s": ttc.get("min_ttc_s"),
        "dtc_risk": risk.get("dtc_risk"),
        "ttc_risk": risk.get("ttc_risk"),
        "combined_risk_score": risk.get("combined_risk_score"),
        "risk_level": risk.get("risk_level"),
        "risk_reason": row.get("risk_reason"),
    }


def compact_ego_state(ego: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "position_global": ego.get("position_global"),
        "rotation_quaternion": ego.get("rotation_quaternion"),
        "yaw": ego.get("yaw"),
        "velocity_global": ego.get("velocity_global"),
        "speed": ego.get("speed"),
        "acceleration_global": ego.get("acceleration_global"),
        "acceleration_norm": ego.get("acceleration_norm"),
        "yaw_rate": ego.get("yaw_rate"),
    }


def compact_agent_state(agent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "annotation_token": agent.get("annotation_token"),
        "instance_token": agent.get("instance_token"),
        "category_name": agent.get("category_name"),
        "attribute_names": agent.get("attribute_names"),
        "visibility_token": agent.get("visibility_token"),
        "position_global": agent.get("position_global"),
        "size": agent.get("size"),
        "yaw": agent.get("yaw"),
        "velocity_global": agent.get("velocity_global"),
        "speed": agent.get("speed"),
        "acceleration_global": agent.get("acceleration_global"),
        "acceleration_norm": agent.get("acceleration_norm"),
        "yaw_rate": agent.get("yaw_rate"),
        "num_lidar_pts": agent.get("num_lidar_pts"),
        "num_radar_pts": agent.get("num_radar_pts"),
    }


def compact_relative_state(relative: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "relative_position_ego": relative.get("relative_position_ego"),
        "relative_velocity_ego": relative.get("relative_velocity_ego"),
        "relative_acceleration_ego": relative.get("relative_acceleration_ego"),
        "longitudinal_distance_m": relative.get("longitudinal_distance_m"),
        "lateral_distance_m": relative.get("lateral_distance_m"),
    }


def compact_risk_metrics(risk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dtc": risk.get("dtc"),
        "ttc": risk.get("ttc"),
        "risk": risk.get("risk"),
        "risk_reason": risk.get("risk_reason"),
    }


def compact_neighbor(
    agent: Dict[str, Any],
    relative: Optional[Dict[str, Any]],
    risk: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    relative_position = relative.get("relative_position_ego", {}) if relative else {}
    relative_velocity = relative.get("relative_velocity_ego", {}) if relative else {}
    ttc = risk.get("ttc", {}) if risk else {}
    return {
        "annotation_token": agent.get("annotation_token"),
        "instance_token": agent.get("instance_token"),
        "category_name": agent.get("category_name"),
        "relative_x_forward_m": relative_position.get("x_forward_m"),
        "relative_y_left_m": relative_position.get("y_left_m"),
        "relative_distance_m": relative_position.get("distance_m"),
        "closing_speed_mps": relative_velocity.get("closing_speed_mps"),
        "min_ttc_s": ttc.get("min_ttc_s"),
    }


def build_lidar_paths(nusc: NuScenes, dataroot: Path, scene: Dict[str, Any]) -> Dict[str, str]:
    lidar_paths = {}
    for sample in collect_scene_samples(nusc, scene):
        token = sample["data"].get("LIDAR_TOP")
        if token is None:
            continue
        sample_data = nusc.get("sample_data", token)
        lidar_paths[sample["token"]] = rel_to_root(dataroot / sample_data["filename"], dataroot)
    return lidar_paths


def write_training_rows(
    writer: Any,
    result: Dict[str, Any],
    lidar_paths: Dict[str, str],
    max_neighbors: int,
) -> int:
    scene_record = result["scene_record"]
    videos = scene_record.get("videos", {})
    ego_by_sample = {row["sample_token"]: row for row in result["ego_states"]}
    rel_by_ann = {row["annotation_token"]: row for row in result["relative_agent_states"]}
    risk_by_ann = {row["annotation_token"]: row for row in result["risk_metrics"]}

    agents_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for agent in result["agent_states"]:
        agents_by_frame.setdefault(agent["frame_index"], []).append(agent)

    for frame_agents in agents_by_frame.values():
        frame_agents.sort(
            key=lambda agent: (
                rel_by_ann.get(agent["annotation_token"], {})
                .get("relative_position_ego", {})
                .get("distance_m", float("inf"))
            )
        )

    count = 0
    for agent in result["agent_states"]:
        ann_token = agent["annotation_token"]
        sample_token = agent["sample_token"]
        relative = rel_by_ann.get(ann_token, {})
        risk = risk_by_ann.get(ann_token, {})
        neighbors = []

        for neighbor in agents_by_frame.get(agent["frame_index"], []):
            if neighbor["annotation_token"] == ann_token:
                continue
            neighbor_ann = neighbor["annotation_token"]
            neighbors.append(compact_neighbor(
                neighbor,
                rel_by_ann.get(neighbor_ann),
                risk_by_ann.get(neighbor_ann),
            ))
            if len(neighbors) >= max_neighbors:
                break

        writer.writerow({
            "scene_name": scene_record.get("scene_name"),
            "scene_token": scene_record.get("scene_token"),
            "split": scene_record.get("split"),
            "frame_index": agent.get("frame_index"),
            "sample_token": sample_token,
            "timestamp": agent.get("timestamp"),
            "target_annotation_token": ann_token,
            "target_agent_id": agent.get("instance_token"),
            "target_category_name": agent.get("category_name"),
            "video_CAM_FRONT": videos.get("CAM_FRONT", ""),
            "video_CAM_FRONT_LEFT": videos.get("CAM_FRONT_LEFT", ""),
            "video_CAM_FRONT_RIGHT": videos.get("CAM_FRONT_RIGHT", ""),
            "video_CAM_BACK": videos.get("CAM_BACK", ""),
            "video_CAM_BACK_LEFT": videos.get("CAM_BACK_LEFT", ""),
            "video_CAM_BACK_RIGHT": videos.get("CAM_BACK_RIGHT", ""),
            "optional_lidar_path": lidar_paths.get(sample_token, ""),
            "ego_state_json": compact_json(compact_ego_state(ego_by_sample[sample_token])),
            "target_agent_state_json": compact_json(compact_agent_state(agent)),
            "target_relative_state_json": compact_json(compact_relative_state(relative)),
            "target_risk_metrics_json": compact_json(compact_risk_metrics(risk)),
            "neighbor_agent_states_json": compact_json(neighbors),
        })
        count += 1

    return count


def write_csv_schema(path: Path, max_neighbors: int) -> None:
    schema = {
        "version": "physical_csv_raw_nuscenes_v0",
        "source": "Direct extraction from nuScenes v1.0-trainval metadata.",
        "max_neighbors": max_neighbors,
        "files": {
            "scenes_physical.csv": SCENES_FIELDS,
            "ego_states.csv": EGO_FIELDS,
            "agent_states.csv": AGENT_FIELDS,
            "relative_agent_states.csv": RELATIVE_FIELDS,
            "risk_metrics.csv": RISK_FIELDS,
            "training_target_index.csv": TRAINING_INDEX_FIELDS,
        },
        "notes": {
            "training_target_index.csv": "One row per target agent at one keyframe.",
            "json_columns": [
                "ego_state_json",
                "target_agent_state_json",
                "target_relative_state_json",
                "target_risk_metrics_json",
                "neighbor_agent_states_json",
            ],
        },
    }
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def select_scenes(
    nusc: NuScenes,
    only_split: str,
    scene_name: Optional[str],
    max_scenes: Optional[int],
) -> List[Dict[str, Any]]:
    split_scenes = create_splits_scenes()
    selected = []
    for scene in nusc.scene:
        split = get_scene_split(scene["name"], split_scenes)
        if scene_name is not None and scene["name"] != scene_name:
            continue
        if only_split != "all" and split != only_split:
            continue
        selected.append({"scene": scene, "split": split})
    if max_scenes is not None:
        selected = selected[:max_scenes]
    if not selected:
        raise ValueError("No scenes selected. Check --scene-name, --only-split, and --max-scenes.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT, help="nuScenes full dataroot.")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes metadata version.")
    parser.add_argument("--outdir", default=None, help="Default: <dataroot>/physical_csv.")
    parser.add_argument("--scene-name", default=None, help="Optional single scene for testing.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional scene limit for smoke tests.")
    parser.add_argument("--only-split", default="all", choices=["all", "train", "val"], help="Scene split filter.")
    parser.add_argument("--max-neighbors", type=int, default=8, help="Neighbors stored in training_target_index.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV files.")
    args = parser.parse_args()

    dataroot = Path(args.dataroot).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else dataroot / "physical_csv"

    output_files = [
        outdir / "scenes_physical.csv",
        outdir / "ego_states.csv",
        outdir / "agent_states.csv",
        outdir / "relative_agent_states.csv",
        outdir / "risk_metrics.csv",
        outdir / "training_target_index.csv",
        outdir / "schema.json",
    ]
    if not args.overwrite:
        existing = [str(path) for path in output_files if path.exists()]
        if existing:
            raise FileExistsError(f"Output files already exist: {existing}. Use --overwrite.")

    mkdir(outdir)

    print("=" * 80)
    print("nuScenes raw physical CSV builder")
    print("=" * 80)
    print(f"dataroot:      {dataroot}")
    print(f"version:       {args.version}")
    print(f"outdir:        {outdir}")
    print(f"scene_name:    {args.scene_name or 'all'}")
    print(f"only_split:    {args.only_split}")
    print(f"max_scenes:    {args.max_scenes}")
    print(f"max_neighbors: {args.max_neighbors}")
    print("=" * 80)

    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=True)
    selected_scenes = select_scenes(
        nusc=nusc,
        only_split=args.only_split,
        scene_name=args.scene_name,
        max_scenes=args.max_scenes,
    )

    files = []
    try:
        scenes_f, scenes_writer = open_writer(outdir / "scenes_physical.csv", SCENES_FIELDS)
        ego_f, ego_writer = open_writer(outdir / "ego_states.csv", EGO_FIELDS)
        agent_f, agent_writer = open_writer(outdir / "agent_states.csv", AGENT_FIELDS)
        rel_f, rel_writer = open_writer(outdir / "relative_agent_states.csv", RELATIVE_FIELDS)
        risk_f, risk_writer = open_writer(outdir / "risk_metrics.csv", RISK_FIELDS)
        target_f, target_writer = open_writer(outdir / "training_target_index.csv", TRAINING_INDEX_FIELDS)
        files = [scenes_f, ego_f, agent_f, rel_f, risk_f, target_f]

        counts = {
            "scenes_physical.csv": 0,
            "ego_states.csv": 0,
            "agent_states.csv": 0,
            "relative_agent_states.csv": 0,
            "risk_metrics.csv": 0,
            "training_target_index.csv": 0,
        }

        for item in tqdm(selected_scenes, desc="Processing scenes"):
            scene = item["scene"]
            scene_split = item["split"]
            result = process_scene(
                nusc=nusc,
                dataroot=dataroot,
                outdir=dataroot / "physical",
                scene=scene,
                scene_split=scene_split,
            )
            lidar_paths = build_lidar_paths(nusc, dataroot, scene)

            scenes_writer.writerow(scene_to_csv_row(result["scene_record"]))
            counts["scenes_physical.csv"] += 1

            for row in result["ego_states"]:
                ego_writer.writerow(ego_to_csv_row(row))
                counts["ego_states.csv"] += 1
            for row in result["agent_states"]:
                agent_writer.writerow(agent_to_csv_row(row))
                counts["agent_states.csv"] += 1
            for row in result["relative_agent_states"]:
                rel_writer.writerow(relative_to_csv_row(row))
                counts["relative_agent_states.csv"] += 1
            for row in result["risk_metrics"]:
                risk_writer.writerow(risk_to_csv_row(row))
                counts["risk_metrics.csv"] += 1

            counts["training_target_index.csv"] += write_training_rows(
                writer=target_writer,
                result=result,
                lidar_paths=lidar_paths,
                max_neighbors=args.max_neighbors,
            )
    finally:
        for f in files:
            f.close()

    write_csv_schema(outdir / "schema.json", args.max_neighbors)

    print("\nDone.")
    for name, count in counts.items():
        print(f"{name}: {count} rows")
    print(f"schema.json: {outdir / 'schema.json'}")


if __name__ == "__main__":
    main()
