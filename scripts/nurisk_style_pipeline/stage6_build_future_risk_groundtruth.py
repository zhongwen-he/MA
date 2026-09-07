#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 6: add future-risk supervision and scene-level ego meta-action labels.

This stage keeps the Stage 5b observed label structure unchanged:
first 4 observed frames contain distance_to_collision only, and the 5th
reference frame contains the complete current risk label.

For Stage 6, only agents that are close agents at the 5th/reference frame are
treated as target agents. For each target agent, the script evaluates the next
3 seconds of keyframes, recomputes NuRisk-style risk scores from
relative_metrics.csv, and selects the minimum future risk score. A single
offline rule-based quantitative ego meta-action is then selected at clip level
and copied to every target-agent label from the same clip. This is VQA
ground-truth annotation only; it does not plan or roll out new ego trajectories.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common import (
    DEFAULT_DATAROOT,
    DEFAULT_KEYFRAME_INTERVAL_SECONDS,
    ensure_csv_field_size,
    format_time_key,
    mkdir,
    output_root,
    parse_float,
    parse_int,
    read_scene_dirs,
)
from stage4_compute_risk_scores_enhanced import enhanced_row, row_to_json_entry
from stage5b_align_video_clip_groundtruth import RISK_SCALE, risk_agent_to_groundtruth


DEFAULT_FUTURE_HORIZON_SECONDS = 3.0
DEFAULT_INTERVENTION_SCORE_THRESHOLD = 2
DEFAULT_MONITOR_SCORE_THRESHOLD = 3

LONGITUDINAL_META_ACTIONS = {
    "maintain_speed": {"target_acceleration_mps2": 0.0, "delta_speed_over_3s_mps": 0.0},
    "accelerate": {"target_acceleration_mps2": 1.0, "delta_speed_over_3s_mps": 3.0},
    "mild_decelerate": {"target_acceleration_mps2": -1.0, "delta_speed_over_3s_mps": -3.0},
    "moderate_decelerate": {"target_acceleration_mps2": -2.0, "delta_speed_over_3s_mps": -6.0},
    "strong_decelerate": {"target_acceleration_mps2": -3.5, "delta_speed_over_3s_mps": -10.5},
    "stopping": {"target_speed_mps": 0.0},
}

LATERAL_META_ACTIONS = {
    "maintain_lane": {"target_lateral_offset_m": 0.0},
    "lateral_shift_left": {"target_lateral_offset_m": 0.7},
    "lateral_shift_right": {"target_lateral_offset_m": -0.7},
    "steer_left": {"target_lateral_offset_m": 3.5},
    "steer_right": {"target_lateral_offset_m": -3.5},
}

LONGITUDINAL_ACTION_STRENGTH = {
    "maintain_speed": 0,
    "accelerate": 1,
    "mild_decelerate": 2,
    "moderate_decelerate": 3,
    "strong_decelerate": 4,
    "stopping": 5,
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


def obstacle_id_from_agent_id(agent_id: str) -> str:
    prefix = "Obstacle "
    return agent_id[len(prefix) :] if agent_id.startswith(prefix) else agent_id


def load_relative_metrics(path: Path) -> Dict[Tuple[int, str], Dict[str, str]]:
    metrics: Dict[Tuple[int, str], Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            metrics[(parse_int(row.get("timestep")), row.get("obstacle_id", ""))] = row
    return metrics


def load_ego_speed_by_timestep(path: Path) -> Dict[int, float]:
    speeds: Dict[int, float] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            speeds[parse_int(row.get("timestep"))] = parse_float(row.get("velocity"))
    return speeds


def numeric_or_none(value: Any) -> Optional[float]:
    if value in {None, "", "Infinity", "-Infinity"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def round_float(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    rounded = round(value, digits)
    return 0.0 if rounded == -0.0 else rounded


def speed_kph(speed_mps: Optional[float]) -> Optional[float]:
    return round_float(speed_mps * 3.6 if speed_mps is not None else None, 1)


def risk_level(score: Optional[int]) -> str:
    if score is None:
        return "Unknown Risk"
    return RISK_SCALE.get(str(score), "Unknown Risk")


def compact_state(
    agent: Dict[str, Any],
    frame_index: int,
    time_key: str,
    ego_speed_mps: Optional[float] = None,
    delta_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    risk_score = agent.get("risk_assessment", {}).get("overall_risk_score")
    state = {
        "frame_index": frame_index,
        "time_key": time_key,
        "risk_score": risk_score,
        "risk_level": agent.get("risk_assessment", {}).get("risk_level", risk_level(risk_score)),
        "relative_direction": agent.get("position", {}).get("relative_direction", "Unknown"),
        "dtc": agent.get("position", {}).get("distance_to_collision", {}),
        "ttc": agent.get("position", {}).get("time_to_collision", {}),
        "relative_velocity": agent.get("motion", {}).get("relative_velocity", {}),
        "relative_acceleration": agent.get("motion", {}).get("relative_acceleration", {}),
    }
    if ego_speed_mps is not None:
        state["ego_speed_mps"] = round_float(ego_speed_mps)
        state["ego_speed_kph"] = speed_kph(ego_speed_mps)
    if delta_seconds is not None:
        state["delta_seconds"] = round_float(delta_seconds)
    return state


def enhanced_metric_to_agent(agent_id: str, metric_row: Dict[str, str]) -> Dict[str, Any]:
    return risk_agent_to_groundtruth(agent_id, row_to_json_entry(enhanced_row(metric_row)))


def select_future_worst(
    agent_id: str,
    obstacle_id: str,
    reference_frame_index: int,
    future_frame_indices: Sequence[int],
    keyframe_interval_seconds: float,
    relative_metrics: Dict[Tuple[int, str], Dict[str, str]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    for frame_index in future_frame_indices:
        metric_row = relative_metrics.get((frame_index, obstacle_id))
        if metric_row is None:
            continue
        agent = enhanced_metric_to_agent(agent_id, metric_row)
        score = agent.get("risk_assessment", {}).get("overall_risk_score", -1)
        delta_seconds = (frame_index - reference_frame_index) * keyframe_interval_seconds
        candidates.append(
            {
                "frame_index": frame_index,
                "time_key": format_time_key(frame_index, keyframe_interval_seconds),
                "delta_seconds": round_float(delta_seconds),
                "risk_score": score,
                "risk_level": agent.get("risk_assessment", {}).get("risk_level", risk_level(score)),
                "agent": agent,
            }
        )

    if not candidates:
        return None, candidates

    # Lower NuRisk score means more dangerous. Tie-break by earlier occurrence.
    worst = min(candidates, key=lambda row: (row["risk_score"], row["frame_index"]))
    return worst, candidates


def risk_change(current_score: Optional[int], future_score: Optional[int], intervention_threshold: int) -> Dict[str, Any]:
    if current_score is None or future_score is None:
        return {
            "current_score": current_score,
            "future_worst_score": future_score,
            "delta_score": None,
            "trend": "unknown",
            "is_future_intervention_required": future_score is not None and future_score <= intervention_threshold,
            "intervention_score_threshold": intervention_threshold,
            "score_definition": "0 = most dangerous, 5 = safest",
        }
    delta = future_score - current_score
    if delta < 0:
        trend = "worsening"
    elif delta > 0:
        trend = "improving"
    else:
        trend = "stable"
    return {
        "current_score": current_score,
        "future_worst_score": future_score,
        "delta_score": delta,
        "trend": trend,
        "is_future_intervention_required": future_score <= intervention_threshold,
        "intervention_score_threshold": intervention_threshold,
        "score_definition": "0 = most dangerous, 5 = safest",
    }

def normalized_direction(relative_direction: Any) -> str:
    return str(relative_direction or "unknown").strip().lower()


def is_front_direction(direction: str) -> bool:
    return direction in {"front", "front-left", "front-right", "collision"}


def is_side_direction(direction: str) -> bool:
    return direction in {"left", "right"}


def is_rear_direction(direction: str) -> bool:
    return direction in {"behind", "rear", "rear-left", "rear-right"}


def future_score(agent_future: Dict[str, Any]) -> Optional[int]:
    future = agent_future.get("future_worst")
    if future is None:
        return None
    score = future.get("risk_score")
    return int(score) if score is not None else None


def future_direction(agent_future: Dict[str, Any]) -> str:
    future = agent_future.get("future_worst") or {}
    return normalized_direction(future.get("relative_direction"))


def risk_set_row(agent_id: str, agent_future: Dict[str, Any]) -> Dict[str, Any]:
    future = agent_future.get("future_worst") or {}
    return {
        "agent_id": agent_id,
        "future_worst_score": future.get("risk_score"),
        "risk_level": future.get("risk_level", risk_level(future.get("risk_score"))),
        "relative_direction": future.get("relative_direction", "Unknown"),
        "frame_index": future.get("frame_index"),
        "delta_seconds": future.get("delta_seconds"),
    }


def longitudinal_action_for_front_score(score: int, direction: str) -> str:
    if direction == "collision":
        return "stopping"
    if score <= 0:
        return "strong_decelerate"
    if score == 1:
        return "moderate_decelerate"
    return "mild_decelerate"


def strongest_longitudinal(actions: Sequence[str]) -> str:
    if not actions:
        return "maintain_speed"
    return max(actions, key=lambda action: LONGITUDINAL_ACTION_STRENGTH.get(action, 0))


def select_lateral_action(triggered: Sequence[Tuple[str, Dict[str, Any]]]) -> str:
    directions = [future_direction(data) for _, data in triggered]
    if any(direction in {"front", "collision"} for direction in directions):
        return "maintain_lane"

    shift_candidates = []
    for direction in directions:
        if direction in {"front-left", "left"}:
            shift_candidates.append("lateral_shift_right")
        elif direction in {"front-right", "right"}:
            shift_candidates.append("lateral_shift_left")

    if shift_candidates and len(set(shift_candidates)) == 1:
        return shift_candidates[0]
    return "maintain_lane"


def quantitative_suggestion(
    longitudinal: str,
    lateral: str,
    current_ego_speed_mps: Optional[float],
    horizon_seconds: float,
) -> Dict[str, Any]:
    current_speed = numeric_or_none(current_ego_speed_mps) or 0.0
    long_template = LONGITUDINAL_META_ACTIONS[longitudinal]
    lateral_template = LATERAL_META_ACTIONS[lateral]

    suggestion = {
        "duration_s": round_float(horizon_seconds, 1),
        "current_speed_mps": round_float(current_speed),
    }
    if "target_speed_mps" in long_template:
        target_speed = max(0.0, float(long_template["target_speed_mps"]))
        delta_speed = target_speed - current_speed
        acceleration = delta_speed / max(horizon_seconds, 1e-6)
        suggestion.update(
            {
                "target_acceleration_mps2": round_float(acceleration),
                "delta_speed_over_3s_mps": round_float(delta_speed),
                "target_speed_mps": round_float(target_speed),
            }
        )
    else:
        delta_speed = float(long_template["delta_speed_over_3s_mps"])
        target_speed = max(0.0, current_speed + delta_speed)
        suggestion.update(
            {
                "target_acceleration_mps2": long_template["target_acceleration_mps2"],
                "delta_speed_over_3s_mps": delta_speed,
                "target_speed_mps": round_float(target_speed),
            }
        )
    suggestion["target_lateral_offset_m"] = lateral_template["target_lateral_offset_m"]
    return suggestion


def select_clip_level_meta_action(
    target_agents: Dict[str, Dict[str, Any]],
    current_ego_speed_mps: Optional[float],
    horizon_seconds: float,
    intervention_threshold: int,
    monitor_threshold: int,
) -> Dict[str, Any]:
    risk_rows = []
    monitor_rows = []
    triggered: List[Tuple[str, Dict[str, Any]]] = []

    for agent_id, agent_future in target_agents.items():
        score = future_score(agent_future)
        if score is None:
            continue
        if score <= intervention_threshold:
            triggered.append((agent_id, agent_future))
            risk_rows.append(risk_set_row(agent_id, agent_future))
        elif score == monitor_threshold:
            monitor_rows.append(risk_set_row(agent_id, agent_future))

    if not triggered:
        mitigation_status = "monitor_or_prepare" if monitor_rows else "no_active_mitigation_required"
        longitudinal = "maintain_speed"
        lateral = "maintain_lane"
        basis = (
            "No target agent crosses the active intervention risk gate; monitor agents remain below active mitigation."
            if monitor_rows
            else "No target agent crosses the monitor or active intervention risk gates."
        )
    else:
        mitigation_status = "active_mitigation_required"
        directions = [future_direction(data) for _, data in triggered]
        scores = [future_score(data) for _, data in triggered if future_score(data) is not None]
        worst_score = min(scores) if scores else intervention_threshold
        has_front = any(is_front_direction(direction) for direction in directions)
        has_side_only = all(is_side_direction(direction) for direction in directions)
        has_rear_only = all(is_rear_direction(direction) for direction in directions)

        longitudinal_candidates = []
        if has_front:
            for _, data in triggered:
                direction = future_direction(data)
                score = future_score(data)
                if score is not None and is_front_direction(direction):
                    longitudinal_candidates.append(longitudinal_action_for_front_score(score, direction))
        elif has_side_only:
            longitudinal_candidates.append("mild_decelerate" if worst_score <= 1 else "maintain_speed")
        elif has_rear_only:
            longitudinal_candidates.append("accelerate" if worst_score <= 1 else "maintain_speed")
        else:
            for _, data in triggered:
                direction = future_direction(data)
                score = future_score(data)
                if score is None:
                    continue
                if is_front_direction(direction):
                    longitudinal_candidates.append(longitudinal_action_for_front_score(score, direction))
                elif is_side_direction(direction) and score <= 1:
                    longitudinal_candidates.append("mild_decelerate")
                elif is_rear_direction(direction) and score <= 1 and not has_front:
                    longitudinal_candidates.append("accelerate")

        longitudinal = strongest_longitudinal(longitudinal_candidates)
        lateral = select_lateral_action(triggered)
        basis = (
            f"{len(triggered)} target agent(s) crossed the active risk gate. "
            f"The selected action uses the strongest required longitudinal response and "
            f"keeps the lane unless all lateral/front-side risks agree on one safe shift direction."
        )

    selected_template_id = f"{longitudinal}__{lateral}"
    return {
        "source": "offline_rule_based_label_generator",
        "selection_policy": "risk_gated_scene_consistent_quantitative_meta_action",
        "mitigation_status": mitigation_status,
        "risk_gate": {
            "score_definition": "0 = most dangerous, 5 = safest",
            "intervention_score_threshold": intervention_threshold,
            "monitor_score_threshold": monitor_threshold,
            "triggered_agent_count": len(risk_rows),
            "monitor_agent_count": len(monitor_rows),
        },
        "risk_set": risk_rows,
        "monitor_set": monitor_rows,
        "selected_template_id": selected_template_id,
        "ego_meta_action": {
            "longitudinal": longitudinal,
            "lateral": lateral,
        },
        "quantitative_suggestion": quantitative_suggestion(
            longitudinal, lateral, current_ego_speed_mps, horizon_seconds
        ),
        "basis": basis,
    }


def target_agent_role(agent_future: Dict[str, Any], intervention_threshold: int, monitor_threshold: int) -> str:
    score = future_score(agent_future)
    if score is None:
        return "future_unavailable"
    if score <= intervention_threshold:
        return "triggering_risk_agent"
    if score == monitor_threshold:
        return "monitoring_agent"
    return "non_triggering_close_agent"


def build_scene_risk_context(
    role: str,
    unified_ego_mitigation: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "risk_set": unified_ego_mitigation.get("risk_set", []),
        "monitor_set": unified_ego_mitigation.get("monitor_set", []),
        "selected_scene_template_id": unified_ego_mitigation.get("selected_template_id"),
        "this_agent_role": role,
    }


def build_mitigation_explanation(
    agent_id: str,
    role: str,
    unified_ego_mitigation: Dict[str, Any],
) -> Dict[str, Any]:
    status = unified_ego_mitigation.get("mitigation_status")
    template_id = unified_ego_mitigation.get("selected_template_id")
    if role == "triggering_risk_agent":
        result = (
            f"Target agent {agent_id} is in the active risk set and directly contributes to the "
            f"scene-level ego meta-action {template_id}."
        )
    elif role == "monitoring_agent":
        result = (
            f"Target agent {agent_id} is in the monitor set. It does not trigger active mitigation, "
            f"but inherits the same scene-level ego meta-action {template_id}."
        )
    elif role == "non_triggering_close_agent":
        result = (
            f"Target agent {agent_id} remains above the monitor/intervention risk gates and inherits "
            f"the scene-level ego meta-action selected from other agents or maintain behavior."
        )
    else:
        result = (
            f"Target agent {agent_id} has no available future risk label within the horizon and inherits "
            f"the scene-level ego meta-action without contributing to the risk gate."
        )
    return {
        "selection_policy": unified_ego_mitigation.get("selection_policy"),
        "mitigation_status": status,
        "target_agent_role": role,
        "selected_scene_template_id": template_id,
        "result": result,
    }

def current_risk_explanation(current_agent: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    explanations = current_agent.get("explanations", {})
    return {
        "step_1": f"Use the complete Stage 5b label at the reference frame {current['time_key']}.",
        "step_2": explanations.get("dominant_weight_explanation", ""),
        "step_3": explanations.get("longitudinal_distance_explanation", ""),
        "step_4": explanations.get("lateral_distance_explanation", ""),
        "step_5": explanations.get("longitudinal_ttc_explanation", ""),
        "step_6": explanations.get("lateral_ttc_explanation", ""),
        "result": explanations.get("overall_risk_explanation", ""),
    }


def future_selection_explanation(
    candidates: Sequence[Dict[str, Any]],
    future_worst: Optional[Dict[str, Any]],
    horizon_seconds: float,
) -> Dict[str, Any]:
    candidate_scores = [
        {
            "frame_index": row["frame_index"],
            "delta_seconds": row["delta_seconds"],
            "risk_score": row["risk_score"],
            "risk_level": row["risk_level"],
        }
        for row in candidates
    ]
    if future_worst is None:
        result = "No future relative metric is available for this target agent within the selected horizon."
    else:
        result = (
            f"The worst future risk occurs at frame {future_worst['frame_index']}, "
            f"{future_worst['delta_seconds']} seconds after the reference frame, "
            f"with risk score {future_worst['risk_score']} ({future_worst['risk_level']})."
        )
    return {
        "step_1": f"Evaluate the same target agent over the next {horizon_seconds:.1f} seconds.",
        "step_2": "Compute NuRisk-style risk score at each available future keyframe using DTC, TTC, relative velocity, and relative acceleration.",
        "step_3": "Select the minimum risk score because lower NuRisk score means higher danger.",
        "candidate_scores": candidate_scores,
        "result": result,
    }


def risk_change_explanation(change: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step_1": f"Current risk score at the reference frame is {change['current_score']}.",
        "step_2": f"Worst future risk score within the horizon is {change['future_worst_score']}.",
        "step_3": f"Delta score = future_worst_score - current_score = {change['delta_score']}.",
        "result": f"The risk trend is {change['trend']}.",
    }


def build_target_agent_future(
    agent_id: str,
    current_agent: Dict[str, Any],
    reference_frame_index: int,
    reference_time_key: str,
    future_frame_indices: Sequence[int],
    keyframe_interval_seconds: float,
    horizon_seconds: float,
    relative_metrics: Dict[Tuple[int, str], Dict[str, str]],
    ego_speeds: Dict[int, float],
    intervention_score_threshold: int,
) -> Dict[str, Any]:
    current_ego_speed = ego_speeds.get(reference_frame_index)
    current = compact_state(current_agent, reference_frame_index, reference_time_key, current_ego_speed)
    obstacle_id = obstacle_id_from_agent_id(agent_id)
    future_worst_raw, candidates = select_future_worst(
        agent_id,
        obstacle_id,
        reference_frame_index,
        future_frame_indices,
        keyframe_interval_seconds,
        relative_metrics,
    )

    future_worst_state: Optional[Dict[str, Any]]
    if future_worst_raw is None:
        future_worst_state = None
        future_score = None
    else:
        future_worst_state = compact_state(
            future_worst_raw["agent"],
            future_worst_raw["frame_index"],
            future_worst_raw["time_key"],
            delta_seconds=future_worst_raw["delta_seconds"],
        )
        future_score = future_worst_state["risk_score"]

    change = risk_change(current["risk_score"], future_score, intervention_score_threshold)
    return {
        "current": current,
        "future_worst": future_worst_state,
        "risk_change": change,
        "explanations": {
            "current_risk_explanation": current_risk_explanation(current_agent, current),
            "future_worst_selection_explanation": future_selection_explanation(
                candidates, future_worst_raw, horizon_seconds
            ),
            "risk_change_explanation": risk_change_explanation(change),
        },
    }

def clip_future_summary(
    target_agents: Dict[str, Dict[str, Any]],
    unified_ego_mitigation: Dict[str, Any],
) -> Dict[str, Any]:
    rows = []
    for agent_id, data in target_agents.items():
        future = data.get("future_worst")
        if future is None or future.get("risk_score") is None:
            continue
        rows.append((agent_id, future))
    if rows:
        agent_id, future = min(rows, key=lambda item: (item[1]["risk_score"], item[1]["frame_index"]))
        min_future_risk_score = future["risk_score"]
        most_critical_agent = agent_id
        most_critical_frame_index = future["frame_index"]
        delta_seconds_from_reference = future["delta_seconds"]
    else:
        min_future_risk_score = None
        most_critical_agent = None
        most_critical_frame_index = None
        delta_seconds_from_reference = None

    return {
        "min_future_risk_score": min_future_risk_score,
        "most_critical_agent": most_critical_agent,
        "most_critical_frame_index": most_critical_frame_index,
        "delta_seconds_from_reference": delta_seconds_from_reference,
        "risk_set": unified_ego_mitigation.get("risk_set", []),
        "monitor_set": unified_ego_mitigation.get("monitor_set", []),
        "selected_scene_template_id": unified_ego_mitigation.get("selected_template_id"),
        "selected_scene_meta_action": unified_ego_mitigation.get("ego_meta_action", {}),
    }

def build_future_entry(
    entry: Dict[str, Any],
    relative_metrics: Dict[Tuple[int, str], Dict[str, str]],
    ego_speeds: Dict[int, float],
    keyframe_interval_seconds: float,
    horizon_seconds: float,
    intervention_score_threshold: int,
    monitor_score_threshold: int,
) -> Dict[str, Any]:
    reference = entry["reference_frame"]
    reference_frame_index = int(reference["frame_index"])
    reference_time_key = reference["time_key"]
    future_frame_count = int(round(horizon_seconds / keyframe_interval_seconds))
    future_frame_indices = list(range(reference_frame_index + 1, reference_frame_index + future_frame_count + 1))

    target_agents = {}
    reference_frame_key = reference["frame_key"]
    for agent_id, agent_track in entry.get("agents", {}).items():
        current_agent = agent_track.get(reference_frame_key)
        if not current_agent or "risk_assessment" not in current_agent:
            continue
        target_agents[agent_id] = build_target_agent_future(
            agent_id,
            current_agent,
            reference_frame_index,
            reference_time_key,
            future_frame_indices,
            keyframe_interval_seconds,
            horizon_seconds,
            relative_metrics,
            ego_speeds,
            intervention_score_threshold,
        )

    current_ego_speed = ego_speeds.get(reference_frame_index)
    unified_ego_mitigation = select_clip_level_meta_action(
        target_agents,
        current_ego_speed,
        horizon_seconds,
        intervention_score_threshold,
        monitor_score_threshold,
    )

    for agent_id, agent_future in target_agents.items():
        role = target_agent_role(agent_future, intervention_score_threshold, monitor_score_threshold)
        agent_future["target_agent_role"] = role
        agent_future["future_risk_triggered"] = role == "triggering_risk_agent"
        agent_future["scene_risk_context"] = build_scene_risk_context(role, unified_ego_mitigation)
        agent_future["unified_ego_mitigation"] = json.loads(json.dumps(unified_ego_mitigation))
        agent_future["mitigation_explanation"] = build_mitigation_explanation(
            agent_id, role, unified_ego_mitigation
        )

    return {
        "clip_id": entry["clip_id"],
        "scene": entry["scene"],
        "scene_token": entry.get("scene_token"),
        "input": {
            "type": "5-frame multi-camera video clip",
            "videos": entry.get("videos", {}),
            "observed_frame_indices": entry.get("frame_indices", []),
            "reference_frame_index": reference_frame_index,
            "sample_tokens": entry.get("sample_tokens", []),
            "timestamps": entry.get("timestamps", []),
        },
        "observed_groundtruth": {
            "frame_structure": "frame_1-4: distance_to_collision only; frame_5: complete current risk label",
            "agents": entry.get("agents", {}),
        },
        "future_groundtruth": {
            "horizon_seconds": horizon_seconds,
            "keyframe_interval_seconds": keyframe_interval_seconds,
            "future_frame_indices": future_frame_indices,
            "intervention_score_threshold": intervention_score_threshold,
            "monitor_score_threshold": monitor_score_threshold,
            "unified_ego_mitigation": unified_ego_mitigation,
            "target_agents": target_agents,
            "clip_future_summary": clip_future_summary(target_agents, unified_ego_mitigation),
        },
    }


def process_scene(
    scene_dir: Path,
    keyframe_interval_seconds: float,
    horizon_seconds: float,
    intervention_score_threshold: int,
    monitor_score_threshold: int,
) -> Optional[Dict[str, Any]]:
    input_path = scene_dir / "video_clip_groundtruth.json"
    relative_path = scene_dir / "relative_metrics.csv"
    ego_path = scene_dir / "ego_trajectory.csv"
    if not input_path.exists():
        print(f"Skipping {scene_dir.name}: missing video_clip_groundtruth.json")
        return None
    if not relative_path.exists() or not ego_path.exists():
        print(f"Skipping {scene_dir.name}: missing relative_metrics.csv or ego_trajectory.csv")
        return None

    stage5 = load_json(input_path)
    relative_metrics = load_relative_metrics(relative_path)
    ego_speeds = load_ego_speed_by_timestep(ego_path)
    entries = [
        build_future_entry(
            entry,
            relative_metrics,
            ego_speeds,
            keyframe_interval_seconds,
            horizon_seconds,
            intervention_score_threshold,
            monitor_score_threshold,
        )
        for entry in stage5.get("entries", [])
    ]
    output = {
        "metadata": {
            "scenario": stage5.get("metadata", {}).get("scenario", scene_dir.name),
            "scene_token": stage5.get("metadata", {}).get("scene_token"),
            "source": str(input_path),
            "future_label_source": str(relative_path),
            "data_type": "multi_camera_video_clip_future_risk_groundtruth",
            "target_agent_rule": "only agents that are close agents at the 5th/reference observed frame",
            "observed_frame_structure": "frame_1-4: distance_to_collision only; frame_5: complete current risk label",
            "future_horizon_seconds": horizon_seconds,
            "keyframe_interval_seconds": keyframe_interval_seconds,
            "future_keyframes": int(round(horizon_seconds / keyframe_interval_seconds)),
            "intervention_score_threshold": intervention_score_threshold,
            "monitor_score_threshold": monitor_score_threshold,
            "mitigation_policy": "risk_gated_scene_consistent_quantitative_meta_action",
            "risk_scale": RISK_SCALE,
            "total_clips": len(entries),
        },
        "entries": entries,
    }
    output_path = scene_dir / "video_future_groundtruth.json"
    write_json(output_path, output)
    return output

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None, help="Default: <dataroot>/nurisk_style")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument(
        "--keyframe-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between nuScenes keyframes. Default is 0.5 for 2Hz keyframes.",
    )
    parser.add_argument("--future-horizon-seconds", type=float, default=DEFAULT_FUTURE_HORIZON_SECONDS)
    parser.add_argument(
        "--intervention-score-threshold",
        type=int,
        default=None,
        help="Active mitigation gate. Lower NuRisk scores are more dangerous; default is 2.",
    )
    parser.add_argument(
        "--monitor-score-threshold",
        type=int,
        default=DEFAULT_MONITOR_SCORE_THRESHOLD,
        help="Monitor/prepare gate. Default is 3.",
    )
    parser.add_argument(
        "--acceptable-score-threshold",
        type=int,
        default=None,
        help="Deprecated alias for --intervention-score-threshold, kept for backward compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_csv_field_size()
    root = output_root(args.dataroot, args.input_dir)
    intervention_score_threshold = (
        args.intervention_score_threshold
        if args.intervention_score_threshold is not None
        else (
            args.acceptable_score_threshold
            if args.acceptable_score_threshold is not None
            else DEFAULT_INTERVENTION_SCORE_THRESHOLD
        )
    )
    monitor_score_threshold = args.monitor_score_threshold

    all_entries = []
    processed = 0
    for scene_dir in read_scene_dirs(root, args.scene_name):
        scene_output = process_scene(
            scene_dir,
            args.keyframe_interval_seconds,
            args.future_horizon_seconds,
            intervention_score_threshold,
            monitor_score_threshold,
        )
        if scene_output is None:
            continue
        processed += 1
        all_entries.extend(scene_output["entries"])

    jsonl_path = root / "video_future_groundtruth.jsonl"
    write_jsonl(jsonl_path, all_entries)
    print("Stage 6 done.")
    print(f"Scene future groundtruth files: {processed}")
    print(f"Future groundtruth entries: {len(all_entries)}")
    print(f"Global JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
