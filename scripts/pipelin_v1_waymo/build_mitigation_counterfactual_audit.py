#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build mitigation counterfactual audit tables from Waymo Stage 6 outputs.

This mirrors the nuscenes_full mitigation audit at sample and clip level:

- sample = one target agent in one video clip with available future labels
- original score = Stage 6 future_worst risk score under the recorded ego path
- counterfactual score = minimum future risk score after rolling out the
  clip-level meta-action / mitigation suggestion for the ego vehicle

NuRisk scores use 0 as most dangerous and 5 as safest, so positive
delta_score = counterfactual - original means safer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import (
    DEFAULT_DATAROOT,
    DEFAULT_EGO_LENGTH,
    DEFAULT_EGO_WIDTH,
    DEFAULT_KEYFRAME_INTERVAL_SECONDS,
    calculate_adjusted_relative_distances,
    calculate_time_to_collision,
    ensure_csv_field_size,
    mkdir,
    output_root,
    parse_float,
    parse_int,
    round_value,
)
from stage4_compute_risk_scores_enhanced import enhanced_row


ACTIVE_THRESHOLD = 2

SAMPLE_FIELDS = [
    "scene",
    "clip_id",
    "raw_agent_id",
    "obstacle_id",
    "agent_name",
    "canonical_agent_name",
    "category_name",
    "target_agent_role",
    "mitigation_status",
    "selected_template_id",
    "meta_longitudinal",
    "meta_lateral",
    "in_risk_set",
    "original_future_min_score",
    "counterfactual_future_min_score",
    "delta_score",
    "change",
    "original_score_le_2",
    "counterfactual_score_le_2",
    "exits_active_gate",
    "enters_active_gate",
    "reference_frame_index",
    "counterfactual_worst_frame_index",
    "counterfactual_worst_delta_seconds",
    "target_acceleration_mps2",
    "target_lateral_offset_m",
]

CLIP_FIELDS = [
    "scene",
    "clip_id",
    "mitigation_status",
    "selected_template_id",
    "meta_longitudinal",
    "meta_lateral",
    "clip_original_min_score",
    "clip_counterfactual_min_score",
    "clip_delta_min_score",
    "clip_change",
    "clip_exits_active_gate",
    "clip_enters_active_gate",
    "sample_count",
    "sample_improved",
    "sample_unchanged",
    "sample_worse",
    "sample_entered_active_gate",
    "sample_exited_active_gate",
    "risk_set_sample_count",
]


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def score_change(delta: int) -> str:
    if delta > 0:
        return "improved_safer"
    if delta < 0:
        return "worse_riskier"
    return "unchanged"


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def obstacle_id_from_agent_id(agent_id: str) -> str:
    prefix = "Obstacle "
    return agent_id[len(prefix) :] if agent_id.startswith(prefix) else agent_id


def load_relative_metrics(path: Path) -> Dict[Tuple[int, str], Dict[str, str]]:
    rows: Dict[Tuple[int, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[(parse_int(row.get("timestep")), row.get("obstacle_id", ""))] = row
    return rows


def jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_distribution(scores: Iterable[int]) -> Dict[str, int]:
    counter = Counter(scores)
    return {str(score): counter[score] for score in range(6)}


def calculate_distance_risk(distance: float) -> int:
    d_abs = abs(distance)
    if d_abs < 0.3:
        return 0
    if d_abs < 0.8:
        return 1
    if d_abs < 1.3:
        return 2
    if d_abs < 3:
        return 3
    if d_abs < 5:
        return 4
    return 5


def identify_relative_direction(d_long: float, d_lat: float) -> str:
    if d_long > DEFAULT_EGO_LENGTH:
        if d_lat > DEFAULT_EGO_WIDTH:
            return "Front-left"
        if d_lat < -DEFAULT_EGO_WIDTH:
            return "Front-right"
        return "Front"
    if d_long < -DEFAULT_EGO_LENGTH:
        if d_lat > DEFAULT_EGO_WIDTH:
            return "Rear-left"
        if d_lat < -DEFAULT_EGO_WIDTH:
            return "Rear-right"
        return "Behind"
    if d_lat > DEFAULT_EGO_WIDTH:
        return "Left"
    if d_lat < -DEFAULT_EGO_WIDTH:
        return "Right"
    return "Collision"


def rollout_adjustments(
    delta_seconds: float,
    horizon_seconds: float,
    target_acceleration: float,
    target_lateral_offset: float,
    current_speed: float,
) -> Tuple[float, float, float, float]:
    dt = max(0.0, float(delta_seconds))
    horizon = max(float(horizon_seconds), 1e-6)
    if target_acceleration < 0 and current_speed > 0:
        stop_time = current_speed / abs(target_acceleration)
        effective_dt = min(dt, stop_time)
        longitudinal_extra = 0.5 * target_acceleration * effective_dt * effective_dt
        if dt > stop_time:
            original_after_stop = current_speed * (dt - stop_time)
            longitudinal_extra -= original_after_stop
        delta_speed = -current_speed if dt > stop_time else target_acceleration * dt
    else:
        longitudinal_extra = 0.5 * target_acceleration * dt * dt
        delta_speed = target_acceleration * dt
    lateral_fraction = min(dt / horizon, 1.0)
    lateral_offset = target_lateral_offset * lateral_fraction
    lateral_velocity = target_lateral_offset / horizon if horizon > 0 and dt <= horizon else 0.0
    return longitudinal_extra, delta_speed, lateral_offset, lateral_velocity


def counterfactual_score_for_row(
    row: Dict[str, str],
    reference_frame_index: int,
    horizon_seconds: float,
    keyframe_interval_seconds: float,
    target_acceleration: float,
    target_lateral_offset: float,
    current_speed: float,
) -> Dict[str, Any]:
    frame_index = parse_int(row.get("timestep"))
    delta_seconds = (frame_index - reference_frame_index) * keyframe_interval_seconds
    d_long = parse_float(row.get("d_long"))
    d_lat = parse_float(row.get("d_lat"))
    v_rel_long = parse_float(row.get("v_rel_long"))
    v_rel_lat = parse_float(row.get("v_rel_lat"))
    a_rel_long = parse_float(row.get("a_rel_long"))
    a_rel_lat = parse_float(row.get("a_rel_lat"))

    long_extra, delta_speed, lateral_offset, lateral_velocity = rollout_adjustments(
        delta_seconds,
        horizon_seconds,
        target_acceleration,
        target_lateral_offset,
        current_speed,
    )
    cf_d_long = d_long - long_extra
    cf_d_lat = d_lat - lateral_offset
    cf_v_rel_long = v_rel_long - delta_speed
    cf_v_rel_lat = v_rel_lat - lateral_velocity
    cf_a_rel_long = a_rel_long - target_acceleration
    cf_a_rel_lat = a_rel_lat
    adjusted_d_long, adjusted_d_lat = calculate_adjusted_relative_distances(cf_d_long, cf_d_lat)
    relative_direction = identify_relative_direction(cf_d_long, cf_d_lat)
    ttc_long, ttc_lat, motion_description = calculate_time_to_collision(
        adjusted_d_long,
        adjusted_d_lat,
        cf_v_rel_long,
        cf_v_rel_lat,
        relative_direction,
    )
    metric = {
        **row,
        "relative_direction": relative_direction,
        "d_long": round_value(cf_d_long, 4),
        "d_lat": round_value(cf_d_lat, 4),
        "adjusted_d_long": round_value(adjusted_d_long, 4),
        "adjusted_d_lat": round_value(adjusted_d_lat, 4),
        "v_rel_long": round_value(cf_v_rel_long, 4),
        "v_rel_lat": round_value(cf_v_rel_lat, 4),
        "a_rel_long": round_value(cf_a_rel_long, 4),
        "a_rel_lat": round_value(cf_a_rel_lat, 4),
        "ttc_long": round_value(ttc_long, 4),
        "ttc_lat": round_value(ttc_lat, 4),
        "motion_description": motion_description,
    }
    scored = enhanced_row({key: str(value) for key, value in metric.items()})
    return {
        "frame_index": frame_index,
        "delta_seconds": round(delta_seconds, 3),
        "risk_score": int(scored["risk_score"]),
        "relative_direction": relative_direction,
    }


def select_counterfactual_worst(
    obstacle_id: str,
    reference_frame_index: int,
    future_frame_indices: Sequence[int],
    horizon_seconds: float,
    keyframe_interval_seconds: float,
    relative_metrics: Dict[Tuple[int, str], Dict[str, str]],
    suggestion: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    target_acceleration = parse_float(suggestion.get("target_acceleration_mps2"))
    target_lateral_offset = parse_float(suggestion.get("target_lateral_offset_m"))
    current_speed = parse_float(suggestion.get("current_speed_mps"))
    candidates: List[Dict[str, Any]] = []
    for frame_index in future_frame_indices:
        metric_row = relative_metrics.get((int(frame_index), obstacle_id))
        if metric_row is None:
            continue
        candidates.append(
            counterfactual_score_for_row(
                metric_row,
                reference_frame_index=reference_frame_index,
                horizon_seconds=horizon_seconds,
                keyframe_interval_seconds=keyframe_interval_seconds,
                target_acceleration=target_acceleration,
                target_lateral_offset=target_lateral_offset,
                current_speed=current_speed,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item["risk_score"], item["frame_index"]))


def sample_row(
    clip: Dict[str, Any],
    agent_id: str,
    agent_future: Dict[str, Any],
    relative_metrics: Dict[Tuple[int, str], Dict[str, str]],
) -> Optional[Dict[str, str]]:
    future = clip.get("future_groundtruth", {})
    mitigation = future.get("unified_ego_mitigation", {})
    suggestion = mitigation.get("quantitative_suggestion", {})
    original_score = safe_int((agent_future.get("future_worst") or {}).get("risk_score"))
    if original_score is None:
        return None

    reference_frame_index = parse_int(clip.get("input", {}).get("reference_frame_index"))
    future_frame_indices = [int(idx) for idx in future.get("future_frame_indices", [])]
    obstacle_id = obstacle_id_from_agent_id(agent_id)
    cf = select_counterfactual_worst(
        obstacle_id,
        reference_frame_index=reference_frame_index,
        future_frame_indices=future_frame_indices,
        horizon_seconds=parse_float(future.get("horizon_seconds"), 3.0),
        keyframe_interval_seconds=parse_float(
            future.get("keyframe_interval_seconds"),
            DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        ),
        relative_metrics=relative_metrics,
        suggestion=suggestion,
    )
    if cf is None:
        return None

    counterfactual_score = int(cf["risk_score"])
    delta = counterfactual_score - original_score
    identity = agent_future.get("agent_identity", {})
    role = agent_future.get("target_agent_role", "")
    in_risk_set = role == "triggering_risk_agent"
    meta_action = mitigation.get("ego_meta_action", {})
    return {
        "scene": str(clip.get("scene", "")),
        "clip_id": str(clip.get("clip_id", "")),
        "raw_agent_id": str(agent_id),
        "obstacle_id": str(obstacle_id),
        "agent_name": str(agent_future.get("clip_reference_name") or identity.get("clip_reference_name") or ""),
        "canonical_agent_name": str(agent_future.get("canonical_agent_name") or identity.get("canonical_agent_name") or ""),
        "category_name": str(identity.get("category_name") or identity.get("category") or ""),
        "target_agent_role": str(role),
        "mitigation_status": str(mitigation.get("mitigation_status", "")),
        "selected_template_id": str(mitigation.get("selected_template_id", "")),
        "meta_longitudinal": str(meta_action.get("longitudinal", "")),
        "meta_lateral": str(meta_action.get("lateral", "")),
        "in_risk_set": bool_text(in_risk_set),
        "original_future_min_score": str(original_score),
        "counterfactual_future_min_score": str(counterfactual_score),
        "delta_score": str(delta),
        "change": score_change(delta),
        "original_score_le_2": bool_text(original_score <= ACTIVE_THRESHOLD),
        "counterfactual_score_le_2": bool_text(counterfactual_score <= ACTIVE_THRESHOLD),
        "exits_active_gate": bool_text(original_score <= ACTIVE_THRESHOLD and counterfactual_score > ACTIVE_THRESHOLD),
        "enters_active_gate": bool_text(original_score > ACTIVE_THRESHOLD and counterfactual_score <= ACTIVE_THRESHOLD),
        "reference_frame_index": str(reference_frame_index),
        "counterfactual_worst_frame_index": str(cf["frame_index"]),
        "counterfactual_worst_delta_seconds": str(cf["delta_seconds"]),
        "target_acceleration_mps2": str(suggestion.get("target_acceleration_mps2", "")),
        "target_lateral_offset_m": str(suggestion.get("target_lateral_offset_m", "")),
    }


def build_clip_row(clip: Dict[str, Any], rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    mitigation = clip.get("future_groundtruth", {}).get("unified_ego_mitigation", {})
    meta_action = mitigation.get("ego_meta_action", {})
    original_scores = [int(row["original_future_min_score"]) for row in rows]
    counterfactual_scores = [int(row["counterfactual_future_min_score"]) for row in rows]
    original_min = min(original_scores) if original_scores else 5
    counterfactual_min = min(counterfactual_scores) if counterfactual_scores else 5
    delta_min = counterfactual_min - original_min
    return {
        "scene": str(clip.get("scene", "")),
        "clip_id": str(clip.get("clip_id", "")),
        "mitigation_status": str(mitigation.get("mitigation_status", "")),
        "selected_template_id": str(mitigation.get("selected_template_id", "")),
        "meta_longitudinal": str(meta_action.get("longitudinal", "")),
        "meta_lateral": str(meta_action.get("lateral", "")),
        "clip_original_min_score": str(original_min),
        "clip_counterfactual_min_score": str(counterfactual_min),
        "clip_delta_min_score": str(delta_min),
        "clip_change": score_change(delta_min),
        "clip_exits_active_gate": bool_text(original_min <= ACTIVE_THRESHOLD and counterfactual_min > ACTIVE_THRESHOLD),
        "clip_enters_active_gate": bool_text(original_min > ACTIVE_THRESHOLD and counterfactual_min <= ACTIVE_THRESHOLD),
        "sample_count": str(len(rows)),
        "sample_improved": str(sum(1 for row in rows if int(row["delta_score"]) > 0)),
        "sample_unchanged": str(sum(1 for row in rows if int(row["delta_score"]) == 0)),
        "sample_worse": str(sum(1 for row in rows if int(row["delta_score"]) < 0)),
        "sample_entered_active_gate": str(sum(1 for row in rows if row["enters_active_gate"] == "true")),
        "sample_exited_active_gate": str(sum(1 for row in rows if row["exits_active_gate"] == "true")),
        "risk_set_sample_count": str(sum(1 for row in rows if row["in_risk_set"] == "true")),
    }


@dataclass
class Agg:
    samples: int = 0
    improved: int = 0
    unchanged: int = 0
    worse: int = 0
    delta_sum: int = 0
    original_le2: int = 0
    counterfactual_le2: int = 0
    exits_gate: int = 0
    enters_gate: int = 0
    original_scores: Counter = field(default_factory=Counter)
    counterfactual_scores: Counter = field(default_factory=Counter)

    def add(self, row: Dict[str, str]) -> None:
        original = int(row["original_future_min_score"])
        counterfactual = int(row["counterfactual_future_min_score"])
        delta = int(row["delta_score"])
        self.samples += 1
        self.delta_sum += delta
        self.original_scores[original] += 1
        self.counterfactual_scores[counterfactual] += 1
        if delta > 0:
            self.improved += 1
        elif delta < 0:
            self.worse += 1
        else:
            self.unchanged += 1
        if original <= ACTIVE_THRESHOLD:
            self.original_le2 += 1
        if counterfactual <= ACTIVE_THRESHOLD:
            self.counterfactual_le2 += 1
        if row["exits_active_gate"] == "true":
            self.exits_gate += 1
        if row["enters_active_gate"] == "true":
            self.enters_gate += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "improved_safer": self.improved,
            "unchanged": self.unchanged,
            "worse_riskier": self.worse,
            "improved_rate": round(self.improved / self.samples, 4) if self.samples else 0.0,
            "worse_rate": round(self.worse / self.samples, 4) if self.samples else 0.0,
            "avg_delta_score": round(self.delta_sum / self.samples, 4) if self.samples else 0.0,
            "original_score_distribution": {str(i): self.original_scores[i] for i in range(6)},
            "counterfactual_score_distribution": {str(i): self.counterfactual_scores[i] for i in range(6)},
            "original_score_le_2": self.original_le2,
            "counterfactual_score_le_2": self.counterfactual_le2,
            "counterfactual_score_gt_2": self.samples - self.counterfactual_le2,
            "counterfactual_score_gt_2_rate": round((self.samples - self.counterfactual_le2) / self.samples, 4)
            if self.samples
            else 0.0,
            "exits_active_gate": self.exits_gate,
            "enters_active_gate": self.enters_gate,
        }


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> None:
    mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_samples(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    subsets = {
        "all_evaluated_targets_all_clips": lambda row: True,
        "active_mitigation_clips_all_targets": lambda row: row["mitigation_status"] == "active_mitigation_required",
        "active_mitigation_clips_risk_set_only": lambda row: row["mitigation_status"] == "active_mitigation_required"
        and row["in_risk_set"] == "true",
        "active_mitigation_clips_non_risk_set_targets": lambda row: row["mitigation_status"] == "active_mitigation_required"
        and row["in_risk_set"] != "true",
        "monitor_or_prepare_clips_all_targets": lambda row: row["mitigation_status"] == "monitor_or_prepare",
        "no_active_mitigation_required_clips_all_targets": lambda row: row["mitigation_status"]
        == "no_active_mitigation_required",
    }
    result: Dict[str, Any] = {}
    for name, pred in subsets.items():
        agg = Agg()
        for row in rows:
            if pred(row):
                agg.add(row)
        result[name] = agg.summary()
    return result


def aggregate_by_action(rows: Sequence[Dict[str, str]], risk_set_only: bool) -> Dict[str, Any]:
    aggs: Dict[str, Agg] = defaultdict(Agg)
    for row in rows:
        if row["mitigation_status"] != "active_mitigation_required":
            continue
        if risk_set_only and row["in_risk_set"] != "true":
            continue
        aggs[row["selected_template_id"]].add(row)
    return {action: aggs[action].summary() for action in sorted(aggs)}


def aggregate_by_role(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    aggs: Dict[str, Agg] = defaultdict(Agg)
    for row in rows:
        if row["mitigation_status"] == "active_mitigation_required":
            aggs[row["target_agent_role"]].add(row)
    return {role: aggs[role].summary() for role in sorted(aggs)}


def problem_clip_table(clip_rows: Sequence[Dict[str, str]], limit: int = 50) -> List[Dict[str, Any]]:
    ordered = sorted(
        clip_rows,
        key=lambda row: (
            -int(row["sample_worse"]),
            -int(row["sample_entered_active_gate"]),
            int(row["clip_delta_min_score"]),
            row["scene"],
            row["clip_id"],
        ),
    )
    return [
        {
            "scene": row["scene"],
            "clip_id": row["clip_id"],
            "mitigation_status": row["mitigation_status"],
            "selected_template_id": row["selected_template_id"],
            "clip_original_min_score": int(row["clip_original_min_score"]),
            "clip_counterfactual_min_score": int(row["clip_counterfactual_min_score"]),
            "clip_delta_min_score": int(row["clip_delta_min_score"]),
            "sample_count": int(row["sample_count"]),
            "sample_worse": int(row["sample_worse"]),
            "sample_entered_active_gate": int(row["sample_entered_active_gate"]),
            "sample_exited_active_gate": int(row["sample_exited_active_gate"]),
        }
        for row in ordered[:limit]
    ]


def markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    summary = audit["summary"]
    subset_rows = []
    for name, item in summary.items():
        subset_rows.append(
            [
                name,
                item["samples"],
                item["improved_safer"],
                item["unchanged"],
                item["worse_riskier"],
                item["improved_rate"],
                item["worse_rate"],
                item["counterfactual_score_le_2"],
                item["counterfactual_score_gt_2"],
                item["counterfactual_score_gt_2_rate"],
                item["exits_active_gate"],
                item["enters_active_gate"],
                item["avg_delta_score"],
            ]
        )

    action_rows = []
    for action, item in audit["by_action_active_risk_set"].items():
        action_rows.append(
            [
                action,
                item["samples"],
                item["improved_safer"],
                item["unchanged"],
                item["worse_riskier"],
                item["counterfactual_score_le_2"],
                item["counterfactual_score_gt_2"],
                item["counterfactual_score_gt_2_rate"],
                item["exits_active_gate"],
                item["avg_delta_score"],
            ]
        )

    text = [
        "# Waymo mitigation counterfactual audit",
        "",
        "NuRisk score interpretation: 0 is most dangerous, 5 is safest. A positive delta means safer.",
        "Counterfactual caveat: this is an offline kinematic approximation, not closed-loop simulation.",
        "",
        "## Main subsets",
        markdown_table(
            subset_rows,
            [
                "subset",
                "samples",
                "improved",
                "unchanged",
                "worse",
                "improved rate",
                "worse rate",
                "cf score <=2",
                "cf score >2",
                "cf >2 rate",
                "exits active gate",
                "enters active gate",
                "avg delta",
            ],
        ),
        "",
        "## Active risk-set by action",
        markdown_table(
            action_rows,
            [
                "action",
                "samples",
                "improved",
                "unchanged",
                "worse",
                "cf <=2",
                "cf >2",
                "cf >2 rate",
                "exits active gate",
                "avg delta",
            ],
        ),
        "",
        "## Files",
        f"- JSON: `{audit['files']['json']}`",
        f"- sample CSV: `{audit['files']['sample_csv']}`",
        f"- clip CSV: `{audit['files']['clip_csv']}`",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def build_audit(dataroot: Path, input_dir: Optional[str], output_dir: Optional[str]) -> Dict[str, Any]:
    risk_root = output_root(str(dataroot), input_dir)
    audit_dir = Path(output_dir).expanduser().resolve() if output_dir else risk_root / "mitigation_counterfactual_audit"
    mkdir(audit_dir)

    relative_cache: Dict[str, Dict[Tuple[int, str], Dict[str, str]]] = {}
    sample_rows: List[Dict[str, str]] = []
    clip_rows: List[Dict[str, str]] = []
    for clip in jsonl_rows(risk_root / "video_future_groundtruth.jsonl"):
        scene = str(clip.get("scene"))
        if scene not in relative_cache:
            relative_cache[scene] = load_relative_metrics(risk_root / scene / "relative_metrics.csv")
        per_clip_rows: List[Dict[str, str]] = []
        for agent_id, agent_future in clip.get("future_groundtruth", {}).get("target_agents", {}).items():
            row = sample_row(clip, agent_id, agent_future, relative_cache[scene])
            if row is not None:
                sample_rows.append(row)
                per_clip_rows.append(row)
        if per_clip_rows:
            clip_rows.append(build_clip_row(clip, per_clip_rows))

    sample_csv = audit_dir / "mitigation_counterfactual_comprehensive_samples.csv"
    clip_csv = audit_dir / "mitigation_counterfactual_comprehensive_clip_risk_summary.csv"
    json_path = audit_dir / "mitigation_counterfactual_comprehensive_audit.json"
    md_path = audit_dir / "mitigation_counterfactual_comprehensive_audit.md"
    write_csv(sample_csv, sample_rows, SAMPLE_FIELDS)
    write_csv(clip_csv, clip_rows, CLIP_FIELDS)

    audit = {
        "method": {
            "description": (
                "Roll out each clip-level Waymo Stage 6 mitigation suggestion for 3 seconds, "
                "adjust ego longitudinal/lateral motion in the original ego frame, keep obstacle "
                "future trajectories unchanged, and recompute NuRisk-style future scores."
            ),
            "safer_definition": "delta_score = counterfactual_future_min_score - original_future_min_score; positive means safer",
            "active_safety_gate": "score <= 2",
            "caveat": "offline kinematic approximation, not closed-loop simulation",
        },
        "summary": aggregate_samples(sample_rows),
        "by_action_active_clips_all_targets": aggregate_by_action(sample_rows, risk_set_only=False),
        "by_action_active_risk_set": aggregate_by_action(sample_rows, risk_set_only=True),
        "by_role_active_clips": aggregate_by_role(sample_rows),
        "problem_clip_table": problem_clip_table(clip_rows),
        "files": {
            "json": str(json_path),
            "sample_csv": str(sample_csv),
            "clip_csv": str(clip_csv),
        },
    }
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None, help="Default: <dataroot>/nurisk_style")
    parser.add_argument("--output-dir", default=None, help="Default: <input-dir>/mitigation_counterfactual_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_csv_field_size()
    audit = build_audit(
        Path(args.dataroot).expanduser().resolve(),
        args.input_dir,
        args.output_dir,
    )
    summary = audit["summary"]["active_mitigation_clips_risk_set_only"]
    print("Waymo mitigation counterfactual audit done.")
    print(f"Active risk-set samples: {summary['samples']}")
    print(f"Improved: {summary['improved_safer']} ({summary['improved_rate']})")
    print(f"Unchanged: {summary['unchanged']}")
    print(f"Worse: {summary['worse_riskier']} ({summary['worse_rate']})")
    print(f"Counterfactual score >2: {summary['counterfactual_score_gt_2']} ({summary['counterfactual_score_gt_2_rate']})")
    print(f"JSON: {audit['files']['json']}")


if __name__ == "__main__":
    main()
