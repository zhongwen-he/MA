#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 7: create Qwen/LLaVA-style VQA conversations from Stage 6 labels.

Each output entry is one target-agent VQA sample:

    one 5-frame observed multi-camera video clip
    + one target agent from the reference frame
    -> one JSON answer with current risk, future worst risk, risk change,
       target-agent scene role, scene-consistent quantitative ego meta-action,
       and step-by-step explanations.

By default, target agents with missing future_worst labels are skipped because
they cannot supervise a future-risk prediction question. Use
--include-unavailable to keep them.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common import DEFAULT_DATAROOT, mkdir, output_root, read_scene_dirs
from stage5b_align_video_clip_groundtruth import RISK_SCALE


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


def compact_current(current: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "frame_index": current.get("frame_index"),
        "time_key": current.get("time_key"),
        "risk_score": current.get("risk_score"),
        "risk_level": current.get("risk_level"),
        "relative_direction": current.get("relative_direction"),
        "distance_to_collision": current.get("dtc", {}),
        "time_to_collision": current.get("ttc", {}),
        "relative_velocity": current.get("relative_velocity", {}),
        "relative_acceleration": current.get("relative_acceleration", {}),
        "ego_speed_mps": current.get("ego_speed_mps"),
        "ego_speed_kph": current.get("ego_speed_kph"),
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
        "time_key": future.get("time_key"),
        "delta_seconds": future.get("delta_seconds"),
        "risk_score": future.get("risk_score"),
        "risk_level": future.get("risk_level"),
        "relative_direction": future.get("relative_direction"),
        "distance_to_collision": future.get("dtc", {}),
        "time_to_collision": future.get("ttc", {}),
        "relative_velocity": future.get("relative_velocity", {}),
        "relative_acceleration": future.get("relative_acceleration", {}),
    }


def build_reasoning_summary(agent_id: str, answer: Dict[str, Any]) -> str:
    current = answer["current_risk"]
    future = answer["predicted_future_worst_risk"]
    change = answer["risk_change_analysis"]
    role = answer.get("target_agent_role", "unknown")
    mitigation = answer.get("scene_consistent_ego_mitigation", {})
    meta_action = mitigation.get("ego_meta_action", {})
    quantitative = mitigation.get("quantitative_suggestion", {})

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
        f"Agent {agent_id} has current risk {current['risk_level']} "
        f"(score {current['risk_score']}) at the last observed frame. "
        f"The future-risk analysis shows that {future_text}. "
        f"The risk trend is {change.get('trend')} with delta score {change.get('delta_score')}. "
        f"This target agent role is {role}. The clip-level scene-consistent ego meta-action is "
        f"longitudinal={longitudinal}, lateral={lateral}, with target speed {target_speed} m/s, "
        f"target acceleration {acceleration} m/s^2, and target lateral offset {lateral_offset} m."
    )

def build_answer(
    clip_entry: Dict[str, Any],
    agent_id: str,
    agent_future: Dict[str, Any],
) -> Dict[str, Any]:
    observed_agents = clip_entry.get("observed_groundtruth", {}).get("agents", {})
    future_groundtruth = clip_entry["future_groundtruth"]
    current = compact_current(agent_future["current"])
    future = compact_future(agent_future.get("future_worst"), future_groundtruth["horizon_seconds"])
    answer = {
        "agent_id": agent_id,
        "task": "current_future_role_and_scene_consistent_meta_action_video_risk_assessment",
        "observed_clip": {
            "clip_id": clip_entry["clip_id"],
            "observed_frame_indices": clip_entry["input"].get("observed_frame_indices", []),
            "reference_frame_index": clip_entry["input"].get("reference_frame_index"),
        },
        "per_timestep_distances": build_observed_distances(observed_agents, agent_id),
        "current_risk": current,
        "predicted_future_worst_risk": future,
        "risk_change_analysis": agent_future.get("risk_change", {}),
        "target_agent_role": agent_future.get("target_agent_role", "future_unavailable"),
        "scene_risk_context": agent_future.get("scene_risk_context", {}),
        "scene_consistent_ego_mitigation": agent_future.get("unified_ego_mitigation", {}),
        "mitigation_explanation": agent_future.get("mitigation_explanation", {}),
        "explanations": agent_future.get("explanations", {}),
    }
    answer["reasoning_summary"] = build_reasoning_summary(agent_id, answer)
    return answer

def build_question(agent_id: str, horizon_seconds: float) -> str:
    return (
        "<video>\n"
        "Analyze the 5-frame observed multi-camera nuScenes video clip. "
        "The input contains only the observed clip; future labels are not visible. "
        f"For target agent id {agent_id}, return JSON only with current risk, "
        f"predicted future {horizon_seconds:.1f}-second worst risk, risk-change analysis, "
        "this target agent's role in the scene-level mitigation decision, and the "
        "scene-consistent quantitative ego meta-action suggestion."
    )

def create_vqa_entry(
    clip_entry: Dict[str, Any],
    agent_id: str,
    agent_future: Dict[str, Any],
) -> Dict[str, Any]:
    horizon_seconds = clip_entry["future_groundtruth"]["horizon_seconds"]
    answer = build_answer(clip_entry, agent_id, agent_future)
    return {
        "id": f"{clip_entry['clip_id']}__{safe_id(agent_id)}",
        "scene": clip_entry["scene"],
        "clip_id": clip_entry["clip_id"],
        "agent_id": agent_id,
        "videos": clip_entry["input"].get("videos", {}),
        "video": list(clip_entry["input"].get("videos", {}).values()),
        "conversations": [
            {
                "from": "human",
                "value": build_question(agent_id, horizon_seconds),
            },
            {
                "from": "gpt",
                "value": json.dumps(answer, ensure_ascii=False, indent=2),
            },
        ],
    }


def process_scene(scene_dir: Path, include_unavailable: bool) -> Optional[Dict[str, Any]]:
    source_path = scene_dir / "video_future_groundtruth.json"
    if not source_path.exists():
        print(f"Skipping {scene_dir.name}: missing video_future_groundtruth.json")
        return None

    source = load_json(source_path)
    entries = []
    skipped_unavailable = 0
    for clip_entry in source.get("entries", []):
        target_agents = clip_entry.get("future_groundtruth", {}).get("target_agents", {})
        for agent_id, agent_future in target_agents.items():
            if agent_future.get("future_worst") is None and not include_unavailable:
                skipped_unavailable += 1
                continue
            entries.append(create_vqa_entry(clip_entry, agent_id, agent_future))

    output = {
        "metadata": {
            "description": f"Qwen/LLaVA-style future-risk VQA dataset for {scene_dir.name}",
            "scenario": scene_dir.name,
            "source": str(source_path),
            "format": "conversation pairs with multi-camera video paths",
            "task": "current risk, future risk, target-agent role, and scene-consistent quantitative ego meta-action mitigation",
            "total_conversations": len(entries),
            "include_unavailable": include_unavailable,
            "skipped_unavailable": skipped_unavailable,
            "risk_scale": RISK_SCALE,
        },
        "entries": entries,
    }
    write_json(scene_dir / "qwen_future_vqa_dataset.json", output)
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
    processed = 0
    skipped_unavailable = 0
    for scene_dir in read_scene_dirs(root, args.scene_name):
        scene_output = process_scene(scene_dir, args.include_unavailable)
        if scene_output is None:
            continue
        processed += 1
        all_entries.extend(scene_output["entries"])
        skipped_unavailable += scene_output["metadata"]["skipped_unavailable"]

    output = {
        "metadata": {
            "description": "Qwen/LLaVA-style VQA dataset for 5-frame nuScenes video future-risk analysis",
            "format": "conversation pairs with multi-camera video paths",
            "task": "current risk, future risk, target-agent role, and scene-consistent quantitative ego meta-action mitigation",
            "total_conversations": len(all_entries),
            "scenes_processed": processed,
            "include_unavailable": args.include_unavailable,
            "skipped_unavailable": skipped_unavailable,
            "risk_scale": RISK_SCALE,
        },
        "entries": all_entries,
    }
    write_json(root / "qwen_future_vqa_dataset.json", output)
    write_jsonl(root / "qwen_future_vqa_dataset.jsonl", all_entries)
    print("Stage 7 done.")
    print(f"Scene VQA files: {processed}")
    print(f"VQA conversations: {len(all_entries)}")
    print(f"Skipped unavailable target agents: {skipped_unavailable}")
    print(f"Global JSON: {root / 'qwen_future_vqa_dataset.json'}")
    print(f"Global JSONL: {root / 'qwen_future_vqa_dataset.jsonl'}")


if __name__ == "__main__":
    main()
