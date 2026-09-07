#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 4: compute NuRisk-style enhanced risk scores and explanations."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common import (
    DEFAULT_DATAROOT,
    DEFAULT_KEYFRAME_INTERVAL_SECONDS,
    ensure_csv_field_size,
    format_time_key,
    open_writer,
    output_root,
    parse_float,
    read_scene_dirs,
    require_absent,
)


RISK_LEVELS = {
    0: "Collision Risk",
    1: "Extreme Risk",
    2: "High Risk",
    3: "Medium Risk",
    4: "Low Risk",
    5: "Negligible Risk",
}

ENHANCED_FIELDS = [
    "wdominant",
    "wdominant_explanation",
    "long_dsc",
    "long_dsc_explanation",
    "lat_dsc",
    "lat_dsc_explanation",
    "dsc",
    "long_tsc",
    "long_tsc_explanation",
    "lat_tsc",
    "lat_tsc_explanation",
    "tsc",
    "risk_score",
    "risk_score_explanation",
]

SUMMARY_FIELDS = [
    "scenario",
    "min_risk_score",
    "num_obstacles",
    "num_timesteps",
]


def parse_metric(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    text = str(value).strip().lower()
    if text in {"inf", "infinity"}:
        return float("inf")
    if text in {"-inf", "-infinity"}:
        return float("-inf")
    return float(value)


def get_dominant_weight(relative_direction: str) -> float:
    direction = relative_direction.lower()
    if direction in {"front", "behind", "rear"}:
        return 1.0
    if direction in {"left", "right"}:
        return 0.0
    return 0.5


def get_dominant_weight_explanation(relative_direction: str) -> str:
    direction = relative_direction.lower()
    if direction in {"front", "behind", "rear"}:
        return "Front/Behind direction: wdominant = 1.0 (longitudinal focus)"
    if direction in {"left", "right"}:
        return "Left/Right direction: wdominant = 0.0 (lateral focus)"
    return f"{relative_direction} direction: wdominant = 0.5(balanced focus)"


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


def get_distance_risk_explanation(distance: float, direction: str) -> str:
    d_abs = abs(distance)
    prefix = "long" if direction == "long" else "lat"
    if d_abs < 0.3:
        return f"{prefix}_dsc = 0 (Collision Risk: {d_abs:.2f}m < 0.3m)"
    if d_abs < 0.8:
        return f"{prefix}_dsc = 1 (Extreme Risk: 0.3m <= {d_abs:.2f}m < 0.8m)"
    if d_abs < 1.3:
        return f"{prefix}_dsc = 2 (High Risk: 0.8m <= {d_abs:.2f}m < 1.3m)"
    if d_abs < 3:
        return f"{prefix}_dsc = 3 (Medium Risk: 1.3m <= {d_abs:.2f}m < 3.0m)"
    if d_abs < 5:
        return f"{prefix}_dsc = 4 (Low Risk: 3.0m <= {d_abs:.2f}m < 5.0m)"
    return f"{prefix}_dsc = 5 (Negligible Risk: {d_abs:.2f}m > 5.0m)"


def calculate_ttc_risk(ttc: float) -> int:
    if math.isnan(ttc) or math.isinf(ttc):
        return 5
    if ttc < 0.15:
        return 0
    if ttc <= 0.65:
        return 1
    if ttc <= 1.15:
        return 2
    if ttc <= 3:
        return 3
    if ttc <= 5:
        return 4
    return 5


def get_ttc_risk_explanation(ttc: float, direction: str) -> str:
    prefix = "long" if direction == "long" else "lat"
    if math.isnan(ttc) or math.isinf(ttc):
        return f"{prefix}_tsc = 5 (Negligible Risk: TTC = Infinity)"
    if ttc < 0.15:
        return f"{prefix}_tsc = 0 (Collision Risk: TTC = {ttc:.2f}s < 0.15s)"
    if ttc <= 0.65:
        return f"{prefix}_tsc = 1 (Extreme Risk: 0.15s <= TTC = {ttc:.2f}s <= 0.65s)"
    if ttc <= 1.15:
        return f"{prefix}_tsc = 2 (High Risk: 0.65s < TTC = {ttc:.2f}s <= 1.15s)"
    if ttc <= 3:
        return f"{prefix}_tsc = 3 (Medium Risk: 1.15s < TTC = {ttc:.2f}s <= 3.0s)"
    if ttc <= 5:
        return f"{prefix}_tsc = 4 (Low Risk: 3.0s < TTC = {ttc:.2f}s <= 5.0s)"
    return f"{prefix}_tsc = 5 (Negligible Risk: TTC = {ttc:.2f}s > 5.0s)"


def risk_level(risk_score: int) -> str:
    return RISK_LEVELS.get(risk_score, "Unknown Risk")


def enhanced_row(row: Dict[str, str]) -> Dict[str, Any]:
    relative_direction = row["relative_direction"]
    adjusted_d_long = parse_metric(row["adjusted_d_long"])
    adjusted_d_lat = parse_metric(row["adjusted_d_lat"])
    ttc_long = parse_metric(row["ttc_long"])
    ttc_lat = parse_metric(row["ttc_lat"])

    wdominant = get_dominant_weight(relative_direction)
    long_dsc = calculate_distance_risk(adjusted_d_long)
    lat_dsc = calculate_distance_risk(adjusted_d_lat)
    dsc = long_dsc * wdominant + lat_dsc * (1.0 - wdominant)
    long_tsc = calculate_ttc_risk(ttc_long)
    lat_tsc = calculate_ttc_risk(ttc_lat)
    tsc = long_tsc * wdominant + lat_tsc * (1.0 - wdominant)
    risk_score = int((dsc + tsc) / 2.0)
    risk_score_explanation = (
        f"DSC = {wdominant:.1f} * {long_dsc} + {1.0 - wdominant:.1f} * {lat_dsc} = {dsc:.1f}; "
        f"TSC = {wdominant:.1f} * {long_tsc} + {1.0 - wdominant:.1f} * {lat_tsc} = {tsc:.1f}; "
        f"Risk = floor(({dsc:.1f} + {tsc:.1f}) / 2) = {risk_score}"
    )

    return {
        **row,
        "wdominant": f"{wdominant:.1f}",
        "wdominant_explanation": get_dominant_weight_explanation(relative_direction),
        "long_dsc": long_dsc,
        "long_dsc_explanation": get_distance_risk_explanation(adjusted_d_long, "long"),
        "lat_dsc": lat_dsc,
        "lat_dsc_explanation": get_distance_risk_explanation(adjusted_d_lat, "lat"),
        "dsc": f"{dsc:.1f}",
        "long_tsc": long_tsc,
        "long_tsc_explanation": get_ttc_risk_explanation(ttc_long, "long"),
        "lat_tsc": lat_tsc,
        "lat_tsc_explanation": get_ttc_risk_explanation(ttc_lat, "lat"),
        "tsc": f"{tsc:.1f}",
        "risk_score": risk_score,
        "risk_score_explanation": risk_score_explanation,
    }


def json_number(value: Any) -> Any:
    parsed = parse_metric(value)
    if math.isnan(parsed):
        return None
    if math.isinf(parsed):
        return "Infinity" if parsed > 0 else "-Infinity"
    return round(parsed, 2)


def row_to_json_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    risk_score = int(row["risk_score"])
    return {
        "Relative Direction": row["relative_direction"],
        "Distance to Collision": {
            "Longitudinal": round(parse_metric(row["adjusted_d_long"]), 2),
            "Lateral": round(parse_metric(row["adjusted_d_lat"]), 2),
        },
        "Relative Velocity": {
            "Longitudinal": round(parse_metric(row["v_rel_long"]), 2),
            "Lateral": round(parse_metric(row["v_rel_lat"]), 2),
        },
        "Relative Acceleration": {
            "Longitudinal": round(parse_metric(row["a_rel_long"]), 2),
            "Lateral": round(parse_metric(row["a_rel_lat"]), 2),
        },
        "Time to Collision": {
            "Longitudinal": json_number(row["ttc_long"]),
            "Lateral": json_number(row["ttc_lat"]),
        },
        "Motion Description": row["motion_description"],
        "Risk Analysis": {
            "Weighting Logic": {
                "Weight": parse_float(row["wdominant"]),
                "Explanation": row["wdominant_explanation"],
            },
            "Distance Risk Scores": {
                "Longitudinal": {
                    "Score": int(row["long_dsc"]),
                    "Explanation": row["long_dsc_explanation"],
                },
                "Lateral": {
                    "Score": int(row["lat_dsc"]),
                    "Explanation": row["lat_dsc_explanation"],
                },
                "Weighted": parse_float(row["dsc"]),
            },
            "TTC Risk Scores": {
                "Longitudinal": {
                    "Score": int(row["long_tsc"]),
                    "Explanation": row["long_tsc_explanation"],
                },
                "Lateral": {
                    "Score": int(row["lat_tsc"]),
                    "Explanation": row["lat_tsc_explanation"],
                },
                "Weighted": parse_float(row["tsc"]),
            },
            "Overall Risk Score": risk_score,
            "Risk Level": risk_level(risk_score),
            "Calculation Process": row["risk_score_explanation"],
        },
    }


def write_enhanced_json(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    keyframe_interval_seconds: float,
) -> None:
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        time_key = format_time_key(row["timestep"], keyframe_interval_seconds)
        obstacle_key = f"Obstacle {row['obstacle_id']}"
        output.setdefault(time_key, {})[obstacle_key] = row_to_json_entry(row)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
        f.write("\n")


def process_scene(
    scene_dir: Path,
    overwrite: bool,
    keyframe_interval_seconds: float,
) -> Optional[Dict[str, Any]]:
    input_path = scene_dir / "close_relative_metrics.csv"
    csv_output_path = scene_dir / "risk_scores_close_relative_metrics_enhanced.csv"
    json_output_path = scene_dir / "risk_scores_output_enhanced.json"
    if not input_path.exists():
        print(f"Skipping {scene_dir.name}: missing close_relative_metrics.csv")
        return None
    require_absent([csv_output_path, json_output_path], overwrite)

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        input_fields = reader.fieldnames or []
        rows = [enhanced_row(row) for row in reader]

    csv_f, writer = open_writer(csv_output_path, list(input_fields) + ENHANCED_FIELDS)
    try:
        writer.writerows(rows)
    finally:
        csv_f.close()
    write_enhanced_json(json_output_path, rows, keyframe_interval_seconds)

    if rows:
        min_risk_score = min(int(row["risk_score"]) for row in rows)
        num_obstacles = len({row["obstacle_id"] for row in rows})
        num_timesteps = len({row["timestep"] for row in rows})
    else:
        min_risk_score = ""
        num_obstacles = 0
        num_timesteps = 0

    return {
        "scenario": scene_dir.name,
        "min_risk_score": min_risk_score,
        "num_obstacles": num_obstacles,
        "num_timesteps": num_timesteps,
    }


def write_summary(path: Path, rows: List[Dict[str, Any]], overwrite: bool) -> None:
    require_absent([path], overwrite)
    rows.sort(key=lambda row: (row["min_risk_score"] == "", row["min_risk_score"], row["scenario"]))
    summary_f, writer = open_writer(path, SUMMARY_FIELDS)
    try:
        writer.writerows(rows)
    finally:
        summary_f.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument(
        "--keyframe-interval-seconds",
        type=float,
        default=DEFAULT_KEYFRAME_INTERVAL_SECONDS,
        help="Seconds between Bench2Drive frames. Default is 0.5 for 2Hz keyframes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_csv_field_size()
    root = output_root(args.dataroot, args.input_dir)
    summary_rows = []
    for scene_dir in read_scene_dirs(root, args.scene_name):
        summary = process_scene(scene_dir, args.overwrite, args.keyframe_interval_seconds)
        if summary is not None:
            summary_rows.append(summary)
    write_summary(root / "risk_score_summary_enhanced.csv", summary_rows, args.overwrite)
    print(f"Stage 4 done: {len(summary_rows)} scenes -> {root}")


if __name__ == "__main__":
    main()
