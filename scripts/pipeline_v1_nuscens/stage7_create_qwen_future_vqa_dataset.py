#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 7: create clean Qwen/LLaVA-style VQA conversations from Stage 6 labels.

Each output entry is one target-agent VQA sample:

    one 5-frame observed multi-camera video clip
    + one named target agent from the reference frame
    -> one JSON answer with current risk, future worst risk, risk change,
       target-vehicle scene role, and scene-consistent quantitative ego
       meta-action.

The training conversations intentionally use clip-level target references such
as "the closest car ahead-left in the adjacent lane at the last observed
frame". nuScenes tokens and raw risk ids are written only to metadata sidecars
for traceability and are not part of the model target.

By default, target agents with missing future_worst labels are skipped because
they cannot supervise a future-risk prediction question. Use
--include-unavailable to keep them.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common import DEFAULT_DATAROOT, DEFAULT_EGO_LENGTH, DEFAULT_EGO_WIDTH, mkdir, output_root, read_scene_dirs
from stage5b_align_video_clip_groundtruth import RISK_SCALE


CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


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


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def distance_for_frame(observed_agents: Dict[str, Any], agent_id: str, frame_key: str) -> Optional[Dict[str, Any]]:
    frame = observed_agents.get(agent_id, {}).get(frame_key)
    if not frame:
        return None
    if "distance_to_collision" in frame:
        return frame["distance_to_collision"]
    return frame.get("position", {}).get("distance_to_collision")


def build_observed_distances(observed_agents: Dict[str, Any], agent_id: str) -> List[Dict[str, Any]]:
    distances = []
    for frame_number in range(1, 6):
        frame_key = f"frame_{frame_number}"
        distance = distance_for_frame(observed_agents, agent_id, frame_key)
        if distance is None:
            continue
        distances.append(
            {
                "frame": frame_number,
                "longitudinal": distance.get("Longitudinal"),
                "lateral": distance.get("Lateral"),
            }
        )
    return distances


def ordered_video_paths(videos: Dict[str, str]) -> List[str]:
    paths = [videos[camera] for camera in CAMERA_ORDER if camera in videos]
    paths.extend(path for camera, path in videos.items() if camera not in CAMERA_ORDER)
    return paths


def axis_pair(values: Any) -> Dict[str, Any]:
    if not isinstance(values, dict):
        return {"longitudinal": None, "lateral": None}
    return {
        "longitudinal": values.get("Longitudinal", values.get("longitudinal")),
        "lateral": values.get("Lateral", values.get("lateral")),
    }


def compact_current(current: Dict[str, Any], final_timestep: int) -> Dict[str, Any]:
    return {
        "frame": final_timestep,
        "relative_direction": current.get("relative_direction"),
        "distance_to_collision": axis_pair(current.get("dtc", {})),
        "time_to_collision": axis_pair(current.get("ttc", {})),
        "velocity": axis_pair(current.get("relative_velocity", {})),
        "acceleration": axis_pair(current.get("relative_acceleration", {})),
        "risk_level": current.get("risk_level"),
        "risk_score": current.get("risk_score"),
        "ego_speed_mps": current.get("ego_speed_mps"),
    }


def compact_future(future: Optional[Dict[str, Any]], horizon_seconds: float) -> Dict[str, Any]:
    if future is None:
        return {
            "horizon_seconds": horizon_seconds,
            "available": False,
            "risk_score": None,
            "risk_level": "Unavailable",
        }
    return {
        "horizon_seconds": horizon_seconds,
        "available": True,
        "frame_index": future.get("frame_index"),
        "delta_seconds": future.get("delta_seconds"),
        "relative_direction": future.get("relative_direction"),
        "distance_to_collision": axis_pair(future.get("dtc", {})),
        "time_to_collision": axis_pair(future.get("ttc", {})),
        "velocity": axis_pair(future.get("relative_velocity", {})),
        "acceleration": axis_pair(future.get("relative_acceleration", {})),
        "risk_score": future.get("risk_score"),
        "risk_level": future.get("risk_level"),
    }


def compact_risk_change(change: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "current_score": change.get("current_score"),
        "future_worst_score": change.get("future_worst_score"),
        "delta_score": change.get("delta_score"),
        "trend": change.get("trend"),
        "future_intervention_required": change.get("is_future_intervention_required"),
    }


def compact_ego_meta_action(mitigation: Dict[str, Any]) -> Dict[str, Any]:
    action = mitigation.get("ego_meta_action", {})
    return {
        "longitudinal": action.get("longitudinal"),
        "lateral": action.get("lateral"),
    }


def compact_quantitative_suggestion(mitigation: Dict[str, Any]) -> Dict[str, Any]:
    suggestion = mitigation.get("quantitative_suggestion", {})
    return {
        "duration_s": suggestion.get("duration_s"),
        "current_speed_mps": suggestion.get("current_speed_mps"),
        "target_speed_mps": suggestion.get("target_speed_mps"),
        "target_acceleration_mps2": suggestion.get("target_acceleration_mps2"),
        "target_lateral_offset_m": suggestion.get("target_lateral_offset_m"),
    }


def build_reasoning_summary(agent_name: str, answer: Dict[str, Any]) -> str:
    current = answer["current_risk"]
    future = answer["future_worst_risk"]
    change = answer["risk_change"]
    role = answer.get("target_agent_role", "unknown")
    meta_action = answer.get("ego_meta_action", {})
    quantitative = answer.get("quantitative_suggestion", {})

    if future.get("available"):
        future_text = (
            f"within {future['horizon_seconds']} seconds the worst future risk is "
            f"{future['risk_level']} (score {future['risk_score']}) at frame "
            f"{future['frame_index']}, {future['delta_seconds']} seconds after the reference frame"
        )
    else:
        future_text = "future risk is unavailable within the selected horizon"

    longitudinal = meta_action.get("longitudinal", "unknown")
    lateral = meta_action.get("lateral", "unknown")
    target_speed = quantitative.get("target_speed_mps")
    acceleration = quantitative.get("target_acceleration_mps2")
    lateral_offset = quantitative.get("target_lateral_offset_m")
    return (
        f"Current score {current['risk_score']} ({current['risk_level']}) at {current['relative_direction']}; "
        f"{future_text}. Trend={change.get('trend')}, delta={change.get('delta_score')}, role={role}. "
        f"Action: {longitudinal}+{lateral}, target_speed={target_speed} m/s, "
        f"acceleration={acceleration} m/s^2, lateral_offset={lateral_offset} m."
    )


def build_explanations(answer: Dict[str, Any]) -> Dict[str, Any]:
    current = answer["current_risk"]
    future = answer["future_worst_risk"]
    change = answer["risk_change"]
    action = answer["ego_meta_action"]
    suggestion = answer["quantitative_suggestion"]
    return {
        "current": f"Final-frame DTC/TTC give risk score {current.get('risk_score')}.",
        "future": f"Worst {future.get('horizon_seconds')}s future score is {future.get('risk_score')}; trend={change.get('trend')}.",
        "action": f"{action.get('longitudinal')}+{action.get('lateral')} to speed {suggestion.get('target_speed_mps')} m/s.",
    }

def build_training_answer(
    clip_entry: Dict[str, Any],
    raw_agent_id: str,
    agent_future: Dict[str, Any],
) -> Dict[str, Any]:
    observed_agents = clip_entry.get("observed_groundtruth", {}).get("agents", {})
    future_groundtruth = clip_entry["future_groundtruth"]
    identity = agent_future.get("agent_identity", {})
    agent_name = identity.get("clip_reference_name") or agent_future.get("clip_reference_name")
    if not agent_name:
        raise ValueError(f"Missing clip_reference_name for {raw_agent_id}")
    per_timestep_distances = build_observed_distances(observed_agents, raw_agent_id)
    final_timestep = len(per_timestep_distances) if per_timestep_distances else 5
    current = compact_current(agent_future["current"], final_timestep)
    future = compact_future(agent_future.get("future_worst"), future_groundtruth["horizon_seconds"])
    mitigation = agent_future.get("unified_ego_mitigation", {})
    answer = {
        "ego_dimensions": {
            "length": DEFAULT_EGO_LENGTH,
            "width": DEFAULT_EGO_WIDTH,
            "unit": "meters",
        },
        "agent_id": agent_name,
        "final_timestep": final_timestep,
        "per_timestep_distances": per_timestep_distances,
        "current_risk": current,
        "future_worst_risk": future,
        "risk_change": compact_risk_change(agent_future.get("risk_change", {})),
        "target_agent_role": agent_future.get("target_agent_role", "future_unavailable"),
        "ego_meta_action": compact_ego_meta_action(mitigation),
        "quantitative_suggestion": compact_quantitative_suggestion(mitigation),
    }
    answer["explanations"] = build_explanations(answer)
    answer["reasoning_summary"] = build_reasoning_summary(agent_name, answer)
    return answer

def build_question(agent_name: str, horizon_seconds: float) -> str:
    return (
        "<video>\n"
        "Analyze the 5-frame observed multi-camera nuScenes video clip. "
        "The input contains only the observed clip; future labels are not visible. "
        f"For the target agent {agent_name}, return JSON only with current risk, "
        f"predicted future {horizon_seconds:.1f}-second worst risk, risk-change analysis, "
        "this target vehicle's role in the scene-level mitigation decision, and the "
        "scene-consistent quantitative ego meta-action suggestion."
    )

def create_vqa_entry(
    clip_entry: Dict[str, Any],
    raw_agent_id: str,
    agent_future: Dict[str, Any],
) -> Dict[str, Any]:
    horizon_seconds = clip_entry["future_groundtruth"]["horizon_seconds"]
    answer = build_training_answer(clip_entry, raw_agent_id, agent_future)
    agent_name = answer["agent_id"]
    videos = clip_entry["input"].get("videos", {})
    return {
        "id": f"{clip_entry['clip_id']}__{safe_id(agent_name)}",
        "scene": clip_entry["scene"],
        "clip_id": clip_entry["clip_id"],
        "target_vehicle": agent_name,
        "canonical_agent_name": answer.get("canonical_agent_name"),
        "clip_reference_name": agent_name,
        "video": ordered_video_paths(videos),
        "conversations": [
            {
                "from": "human",
                "value": build_question(agent_name, horizon_seconds),
            },
            {
                "from": "gpt",
                "value": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def create_metadata_entry(
    clip_entry: Dict[str, Any],
    raw_agent_id: str,
    agent_future: Dict[str, Any],
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    identity = agent_future.get("agent_identity", {})
    videos = clip_entry["input"].get("videos", {})
    return {
        "id": entry["id"],
        "scene": entry["scene"],
        "clip_id": entry["clip_id"],
        "target_vehicle": entry["target_vehicle"],
        "canonical_agent_name": entry.get("canonical_agent_name"),
        "clip_reference_name": entry.get("clip_reference_name"),
        "agent_identity": {
            "raw_agent_id": identity.get("raw_agent_id", raw_agent_id),
            "risk_agent_id": identity.get("risk_agent_id", raw_agent_id),
            "instance_token": identity.get("instance_token"),
            "is_named_agent": identity.get("is_named_agent", False),
            "is_named_vehicle": identity.get("is_named_vehicle", False),
            "is_vehicle": identity.get("is_vehicle", False),
            "canonical_agent_name": identity.get("canonical_agent_name"),
            "clip_reference_name": identity.get("clip_reference_name"),
            "target_reference": identity.get("target_reference"),
            "category_name": identity.get("category_name"),
            "category": identity.get("category"),
            "relative_position": identity.get("relative_position"),
            "relative_direction": identity.get("relative_direction"),
            "distance_meters": identity.get("distance_meters"),
            "distance_bucket": identity.get("distance_bucket"),
            "near_far": identity.get("near_far"),
            "visible_camera": identity.get("visible_camera"),
            "visible_cameras": identity.get("visible_cameras", []),
            "projected_bbox_area": identity.get("projected_bbox_area"),
            "rank_in_group": identity.get("rank_in_group"),
            "rank_label": identity.get("rank_label"),
            "position_group_size": identity.get("position_group_size"),
            "reference_quality": identity.get("reference_quality"),
            "skip_vqa": identity.get("skip_vqa", False),
            "skip_reason": identity.get("skip_reason"),
            "is_reference_ambiguous": identity.get("is_reference_ambiguous", False),
        },
        "raw_agent_id": identity.get("raw_agent_id", raw_agent_id),
        "risk_agent_id": identity.get("risk_agent_id", raw_agent_id),
        "instance_token": identity.get("instance_token"),
        "videos": videos,
        "video": ordered_video_paths(videos),
        "observed_frame_indices": clip_entry["input"].get("observed_frame_indices", []),
        "reference_frame_index": clip_entry["input"].get("reference_frame_index"),
    }


def process_scene(scene_dir: Path, include_unavailable: bool) -> Optional[Dict[str, Any]]:
    source_path = scene_dir / "video_future_groundtruth.json"
    if not source_path.exists():
        print(f"Skipping {scene_dir.name}: missing video_future_groundtruth.json")
        return None

    source = load_json(source_path)
    entries = []
    metadata_entries = []
    skipped_unavailable = 0
    skipped_unnamed = 0
    skipped_ambiguous = 0
    skipped_low_quality_reference = 0
    for clip_entry in source.get("entries", []):
        target_agents = clip_entry.get("future_groundtruth", {}).get("target_agents", {})
        for raw_agent_id, agent_future in target_agents.items():
            identity = agent_future.get("agent_identity", {})
            if not identity.get("is_named_agent") or not identity.get("clip_reference_name"):
                skipped_unnamed += 1
                continue
            if identity.get("skip_vqa"):
                skipped_low_quality_reference += 1
                continue
            if identity.get("is_reference_ambiguous"):
                skipped_ambiguous += 1
                continue
            if agent_future.get("future_worst") is None and not include_unavailable:
                skipped_unavailable += 1
                continue
            entry = create_vqa_entry(clip_entry, raw_agent_id, agent_future)
            entries.append(entry)
            metadata_entries.append(create_metadata_entry(clip_entry, raw_agent_id, agent_future, entry))

    output = {
        "metadata": {
            "description": f"Qwen/LLaVA-style future-risk VQA dataset for {scene_dir.name}",
            "scenario": scene_dir.name,
            "source": str(source_path),
            "format": "clean conversation pairs with multi-camera video paths",
            "task": "current risk, future risk, target-vehicle role, and scene-consistent quantitative ego meta-action mitigation",
            "total_conversations": len(entries),
            "include_unavailable": include_unavailable,
            "skipped_unavailable": skipped_unavailable,
            "skipped_unnamed": skipped_unnamed,
            "skipped_ambiguous": skipped_ambiguous,
            "skipped_low_quality_reference": skipped_low_quality_reference,
            "risk_scale": RISK_SCALE,
            "metadata_sidecar": str(scene_dir / "qwen_future_vqa_metadata.json"),
        },
        "entries": entries,
    }
    write_json(scene_dir / "qwen_future_vqa_dataset.json", output)
    write_json(
        scene_dir / "qwen_future_vqa_metadata.json",
        {
            "metadata": {
                "description": f"Traceability metadata for {scene_dir.name} clean VQA samples",
                "source": str(source_path),
                "total_entries": len(metadata_entries),
            },
            "entries": metadata_entries,
        },
    )
    output["metadata_entries"] = metadata_entries
    output["skipped_unnamed"] = skipped_unnamed
    output["skipped_ambiguous"] = skipped_ambiguous
    output["skipped_low_quality_reference"] = skipped_low_quality_reference
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None, help="Default: <dataroot>/nurisk_style")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument(
        "--include-unavailable",
        action="store_true",
        help="Keep target agents whose future_worst label is unavailable. Default: skip them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = output_root(args.dataroot, args.input_dir)
    all_entries = []
    all_metadata_entries = []
    processed = 0
    skipped_unavailable = 0
    skipped_unnamed = 0
    skipped_ambiguous = 0
    skipped_low_quality_reference = 0
    for scene_dir in read_scene_dirs(root, args.scene_name):
        scene_output = process_scene(scene_dir, args.include_unavailable)
        if scene_output is None:
            continue
        processed += 1
        all_entries.extend(scene_output["entries"])
        all_metadata_entries.extend(scene_output["metadata_entries"])
        skipped_unavailable += scene_output["metadata"]["skipped_unavailable"]
        skipped_unnamed += scene_output.get("skipped_unnamed", 0)
        skipped_ambiguous += scene_output.get("skipped_ambiguous", 0)
        skipped_low_quality_reference += scene_output.get("skipped_low_quality_reference", 0)

    output = {
        "metadata": {
            "description": "Clean Qwen/LLaVA-style VQA dataset for 5-frame nuScenes video future-risk analysis",
            "format": "clean conversation pairs with multi-camera video paths",
            "task": "current risk, future risk, target-vehicle role, and scene-consistent quantitative ego meta-action mitigation",
            "total_conversations": len(all_entries),
            "scenes_processed": processed,
            "include_unavailable": args.include_unavailable,
            "skipped_unavailable": skipped_unavailable,
            "skipped_unnamed": skipped_unnamed,
            "skipped_ambiguous": skipped_ambiguous,
            "skipped_low_quality_reference": skipped_low_quality_reference,
            "risk_scale": RISK_SCALE,
            "metadata_sidecar": str(root / "qwen_future_vqa_metadata.json"),
        },
        "entries": all_entries,
    }
    write_json(root / "qwen_future_vqa_dataset.json", output)
    write_jsonl(root / "qwen_future_vqa_dataset.jsonl", all_entries)
    write_json(
        root / "qwen_future_vqa_metadata.json",
        {
            "metadata": {
                "description": "Traceability metadata for clean VQA samples",
                "total_entries": len(all_metadata_entries),
            },
            "entries": all_metadata_entries,
        },
    )
    write_jsonl(root / "qwen_future_vqa_metadata.jsonl", all_metadata_entries)
    print("Stage 7 done.")
    print(f"Scene VQA files: {processed}")
    print(f"VQA conversations: {len(all_entries)}")
    print(f"Skipped unavailable target vehicles: {skipped_unavailable}")
    print(f"Skipped unnamed target vehicles: {skipped_unnamed}")
    print(f"Skipped ambiguous target vehicles: {skipped_ambiguous}")
    print(f"Skipped low-quality target references: {skipped_low_quality_reference}")
    print(f"Global JSON: {root / 'qwen_future_vqa_dataset.json'}")
    print(f"Global JSONL: {root / 'qwen_future_vqa_dataset.jsonl'}")
    print(f"Metadata sidecar: {root / 'qwen_future_vqa_metadata.json'}")


if __name__ == "__main__":
    main()
