#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent reference helpers for Bench2Drive raw-video VQA generation.

Pipeline v1 keeps Bench2Drive actor ids as internal join keys only. The VQA
text refers to target agents with clip-level expressions anchored at the last
observed/reference frame, for example:

    canonical_agent_name: car
    clip_reference_name: the closest car ahead-left in the adjacent lane at the last observed frame

All Bench2Drive annotated agents can receive names. The final clip references are built
from Stage 4 close-agent risk rows, so far or irrelevant scene instances do not
become VQA targets merely because they exist in Bench2Drive.
"""

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


RISK_AGENT_PREFIX = "Obstacle "

VEHICLE_CATEGORY_NAMES = {
    "vehicle": "vehicle",
    "vehicle.car": "car",
    "vehicle.bus": "bus",
    "vehicle.truck": "truck",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction vehicle",
    "vehicle.emergency.ambulance": "ambulance",
    "vehicle.emergency.police": "police car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

CATEGORY_LABEL_NAMES = {
    **VEHICLE_CATEGORY_NAMES,
    "walker": "pedestrian",
    "pedestrian": "pedestrian",
    "traffic_light": "traffic light",
    "traffic_sign": "traffic sign",
    "static.traffic_light": "traffic light",
    "static.traffic_sign": "traffic sign",
    "human.pedestrian.adult": "adult pedestrian",
    "human.pedestrian.child": "child pedestrian",
    "human.pedestrian.construction_worker": "construction worker",
    "human.pedestrian.personal_mobility": "personal mobility rider",
    "human.pedestrian.police_officer": "police officer",
    "human.pedestrian.stroller": "stroller pedestrian",
    "human.pedestrian.wheelchair": "wheelchair pedestrian",
    "human.pedestrian": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "debris",
    "movable_object.pushable_pullable": "pushable object",
    "movable_object.trafficcone": "traffic cone",
    "static_object.bicycle_rack": "bicycle rack",
    "animal": "animal",
}

DIRECTION_PHRASES = {
    "front": "ahead in the same lane",
    "front-left": "ahead-left in the adjacent lane",
    "front-right": "ahead-right in the adjacent lane",
    "left": "on the left side",
    "right": "on the right side",
    "behind": "behind in the same lane",
    "rear-left": "behind-left in the adjacent lane",
    "rear-right": "behind-right in the adjacent lane",
    "collision": "very close to the ego path",
}

COARSE_DIRECTION_PHRASES = {
    "front": "ahead",
    "front-left": "ahead-left",
    "front-right": "ahead-right",
    "left": "left",
    "right": "right",
    "behind": "behind",
    "rear-left": "behind-left",
    "rear-right": "behind-right",
    "collision": "near ego",
}


def risk_agent_id(instance_token: str) -> str:
    return f"{RISK_AGENT_PREFIX}{instance_token}"


def instance_token_from_risk_agent_id(agent_id: str) -> str:
    if agent_id.startswith(RISK_AGENT_PREFIX):
        return agent_id[len(RISK_AGENT_PREFIX) :]
    return agent_id


def normalize_category_name(category_name: str) -> str:
    return str(category_name or "").strip().lower()


def is_vehicle_category(category_name: str) -> bool:
    category = normalize_category_name(category_name)
    return category == "vehicle" or category.startswith("vehicle.")


def vehicle_category_label(category_name: str) -> Optional[str]:
    category = normalize_category_name(category_name)
    if not is_vehicle_category(category):
        return None
    return VEHICLE_CATEGORY_NAMES.get(category, category.split(".")[-1].replace("_", " "))


def category_label(category_name: str) -> str:
    category = normalize_category_name(category_name)
    if not category:
        return "object"
    if category in CATEGORY_LABEL_NAMES:
        return CATEGORY_LABEL_NAMES[category]
    parts = [part for part in category.split(".") if part]
    if len(parts) >= 2 and parts[0] == "human" and parts[1] == "pedestrian":
        return "pedestrian"
    return parts[-1].replace("_", " ") if parts else "object"


def canonical_agent_name(category: Optional[str], color: Optional[str] = None) -> str:
    """Return the stable category-only canonical name.

    ``color`` remains accepted for backward-compatible callers but is ignored by
    the current reference policy.
    """

    return str(category or "object").strip().lower() or "object"


def build_scene_agent_alias_map(
    scene_name: str,
    samples: Iterable[Dict[str, Any]],
    ann_by_token: Dict[str, Dict[str, Any]],
    visual_descriptors: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a category/token index for all annotated instances in a scene.

    This does not assign final VQA target names. Clip-level reference names are
    generated later from the close agents present at the reference frame.
    ``visual_descriptors`` is accepted for compatibility and intentionally
    ignored, because color is not part of the primary naming scheme.
    """

    instance_tokens: List[str] = []
    seen = set()
    for sample in samples:
        for ann_token in sample.get("anns", []):
            ann = ann_by_token.get(ann_token)
            if ann is None:
                continue
            instance_token = ann["instance_token"]
            if instance_token in seen:
                continue
            seen.add(instance_token)
            instance_tokens.append(instance_token)

    references = []
    for instance_token in instance_tokens:
        anns = [
            ann for ann in ann_by_token.values()
            if ann.get("instance_token") == instance_token
        ]
        category_name = anns[0].get("category_name", "") if anns else ""
        category = category_label(category_name)
        if not category:
            continue
        is_vehicle = is_vehicle_category(category_name)
        references.append(
            {
                "instance_token": instance_token,
                "raw_agent_id": risk_agent_id(instance_token),
                "risk_agent_id": risk_agent_id(instance_token),
                "category_name": category_name,
                "category": category,
                "canonical_agent_name": canonical_agent_name(category),
                "is_vehicle": is_vehicle,
                "is_nameable_agent": True,
                "use_color": False,
                "reference_scope": "scene_agent_category_index",
            }
        )

    return {
        "scene_name": scene_name,
        "reference_scope": "scene_agent_category_index",
        "raw_id_type": "Bench2Drive actor id",
        "risk_agent_id_format": "Obstacle <instance_token>",
        "naming_policy": (
            "category-only scene index; final VQA names are generated per clip "
            "from reference-frame close agents using category, relative position, "
            "distance, and rank"
        ),
        "references": references,
        "instance_token_to_reference": {
            row["instance_token"]: row for row in references
        },
        "risk_agent_id_to_reference": {
            row["risk_agent_id"]: row for row in references
        },
    }


def reference_lookup(agent_reference_map: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not agent_reference_map:
        return {}
    lookup = {}
    for row in agent_reference_map.get("references", []):
        for key in ("risk_agent_id", "raw_agent_id", "instance_token"):
            value = row.get(key)
            if value:
                lookup[value] = row
    return lookup


def agent_reference_for_risk_agent(
    raw_agent_id: str,
    agent_reference_map: Optional[Dict[str, Any]],
    clip_agent_reference_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instance_token = instance_token_from_risk_agent_id(raw_agent_id)
    lookup = reference_lookup(clip_agent_reference_map) or {}
    row = lookup.get(raw_agent_id) or lookup.get(instance_token)
    if row is None:
        scene_lookup = reference_lookup(agent_reference_map)
        row = scene_lookup.get(raw_agent_id) or scene_lookup.get(instance_token)

    identity = {
        "raw_agent_id": raw_agent_id,
        "risk_agent_id": raw_agent_id if raw_agent_id.startswith(RISK_AGENT_PREFIX) else risk_agent_id(instance_token),
        "instance_token": instance_token,
        "is_named_agent": False,
        "is_named_vehicle": False,
    }
    if row is None:
        return identity

    identity.update(
        {
            "is_named_agent": True,
            "is_named_vehicle": bool(row.get("is_vehicle", False)),
            "canonical_agent_name": row.get("canonical_agent_name"),
            "clip_reference_name": row.get("clip_reference_name") or row.get("canonical_agent_name"),
            "target_reference": row.get("target_reference"),
            "category_name": row.get("category_name"),
            "category": row.get("category"),
            "is_vehicle": bool(row.get("is_vehicle", False)),
            "relative_position": row.get("relative_position"),
            "relative_direction": row.get("relative_direction"),
            "distance_meters": row.get("distance_meters"),
            "distance_bucket": row.get("distance_bucket"),
            "near_far": row.get("near_far"),
            "visible_camera": row.get("visible_camera"),
            "visible_cameras": row.get("visible_cameras", []),
            "projected_bbox_area": row.get("projected_bbox_area"),
            "reference_visibility": row.get("reference_visibility"),
            "rank_in_group": row.get("rank_in_group"),
            "rank_label": row.get("rank_label"),
            "position_group_size": row.get("position_group_size"),
            "reference_scope": row.get("reference_scope"),
            "reference_quality": row.get("reference_quality"),
            "skip_vqa": bool(row.get("skip_vqa", False)),
            "skip_reason": row.get("skip_reason"),
            "is_reference_ambiguous": bool(row.get("is_reference_ambiguous", False)),
            "use_color": False,
        }
    )
    return identity


def attach_identity(
    payload: Dict[str, Any],
    raw_agent_id: str,
    agent_reference_map: Optional[Dict[str, Any]],
    clip_agent_reference_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload.update(agent_reference_for_risk_agent(raw_agent_id, agent_reference_map, clip_agent_reference_map))
    return payload


def normalize_relative_direction(relative_direction: Any) -> str:
    direction = str(relative_direction or "unknown").strip().lower()
    aliases = {
        "front left": "front-left",
        "front_right": "front-right",
        "front right": "front-right",
        "rear_left": "rear-left",
        "rear left": "rear-left",
        "rear_right": "rear-right",
        "rear right": "rear-right",
        "back": "behind",
        "rear": "behind",
    }
    return aliases.get(direction, direction)


def relative_position_phrase(relative_direction: Any) -> str:
    return DIRECTION_PHRASES.get(normalize_relative_direction(relative_direction), "near the ego vehicle")


def coarse_direction_phrase(relative_direction: Any) -> str:
    return COARSE_DIRECTION_PHRASES.get(normalize_relative_direction(relative_direction), "nearby")


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


def _axis_distance(values: Dict[str, Any], primary: str) -> Optional[float]:
    if not isinstance(values, dict):
        return None
    for key in (primary, primary.lower()):
        parsed = numeric_or_none(values.get(key))
        if parsed is not None:
            return abs(parsed)
    return None


def distance_meters(agent_data: Dict[str, Any]) -> Optional[float]:
    dtc = agent_data.get("Distance to Collision", {})
    long_d = _axis_distance(dtc, "Longitudinal")
    lat_d = _axis_distance(dtc, "Lateral")
    if long_d is not None and lat_d is not None:
        return math.hypot(long_d, lat_d)
    return long_d if long_d is not None else lat_d


def distance_bucket(distance: Optional[float]) -> Optional[str]:
    if distance is None:
        return None
    edges = [0, 5, 10, 15, 20, 30, 40, 60]
    for lower, upper in zip(edges, edges[1:]):
        if lower <= distance < upper:
            return f"{lower}-{upper} m"
    return "60+ m"


def near_far_label(distance: Optional[float]) -> Optional[str]:
    if distance is None:
        return None
    return "near" if distance < 15.0 else "far"


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def rank_label(rank_index: int, group_size: int) -> str:
    if rank_index == 0:
        return "closest"
    if rank_index == group_size - 1:
        return "farthest"
    return f"{_ordinal(rank_index + 1)} closest"


def build_reference_expression(
    category: str,
    position: str,
    rank: Optional[str],
    bucket: Optional[str],
) -> str:
    rank_part = f"{rank} " if rank else ""
    bucket_part = f" {bucket} away" if bucket else ""
    return f"the {rank_part}{category} {position}{bucket_part} at the last observed frame"


def build_clip_agent_reference_map(
    scene_reference_map: Optional[Dict[str, Any]],
    frame_agents: Dict[str, Dict[str, Any]],
    reference_visibility: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create clip-level target references for reference-frame close agents.

    Stage 4 already filters close obstacles. This function therefore names only
    the close agents present in ``frame_agents`` and makes each reference unique
    within the clip using category + final relative position + distance rank.
    Rank, projected-box area, and visibility are kept as metadata but do not
    filter targets at this stage.
    """

    scene_lookup = reference_lookup(scene_reference_map)
    visibility_lookup = reference_visibility or {}
    rows: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for raw_agent_id, agent_data in frame_agents.items():
        base = scene_lookup.get(raw_agent_id) or scene_lookup.get(instance_token_from_risk_agent_id(raw_agent_id))
        if not base:
            continue
        distance = distance_meters(agent_data)
        direction = normalize_relative_direction(agent_data.get("Relative Direction"))
        position = relative_position_phrase(direction)
        category = canonical_agent_name(base.get("category"))
        visibility = (
            visibility_lookup.get(raw_agent_id)
            or visibility_lookup.get(instance_token_from_risk_agent_id(raw_agent_id))
            or {}
        )
        row = {
            **base,
            "canonical_agent_name": category,
            "relative_direction": direction,
            "relative_position": position,
            "coarse_direction": coarse_direction_phrase(direction),
            "distance_meters": round(distance, 2) if distance is not None else None,
            "distance_bucket": distance_bucket(distance),
            "near_far": near_far_label(distance),
            "reference_scope": "clip_reference_frame_agent",
            "anchor_frame": "reference_frame",
            "anchor_frame_description": "last observed frame",
            "visible_cameras": visibility.get("visible_cameras", []),
            "visible_camera": visibility.get("visible_camera"),
            "projected_bbox_area": visibility.get("projected_bbox_area"),
            "reference_visibility": visibility,
            "use_color": False,
        }
        rows.append(row)
        groups[(category, position)].append(len(rows) - 1)

    for indexes in groups.values():
        indexes.sort(
            key=lambda idx: (
                rows[idx]["distance_meters"] is None,
                rows[idx]["distance_meters"] if rows[idx]["distance_meters"] is not None else float("inf"),
                rows[idx]["risk_agent_id"],
            )
        )
        group_size = len(indexes)
        for rank_index, row_index in enumerate(indexes):
            row = rows[row_index]
            rank = rank_label(rank_index, group_size) if group_size > 1 else None
            rank_number = rank_index + 1
            expression = build_reference_expression(
                row["canonical_agent_name"],
                row["relative_position"],
                rank,
                row["distance_bucket"],
            )
            skip_vqa = False
            row.update(
                {
                    "clip_reference_name": expression,
                    "rank_in_group": rank_number,
                    "rank_label": rank,
                    "position_group_size": group_size,
                    "reference_quality": "high",
                    "skip_vqa": skip_vqa,
                    "skip_reason": None,
                    "is_reference_ambiguous": False,
                    "target_reference": {
                        "primary_expression": expression,
                        "anchor_frame": "reference_frame",
                        "anchor_frame_description": "last observed frame",
                        "category": row.get("category"),
                        "category_name": row.get("category_name"),
                        "final_relative_position": row["relative_direction"],
                        "relative_position_phrase": row["relative_position"],
                        "rank_in_group": rank_number,
                        "rank_label": rank,
                        "position_group_size": group_size,
                        "distance_meters": row["distance_meters"],
                        "distance_bucket": row["distance_bucket"],
                        "visible_camera": row.get("visible_camera"),
                        "visible_cameras": row.get("visible_cameras", []),
                        "projected_bbox_area": row.get("projected_bbox_area"),
                        "use_color": False,
                        "is_unique_in_clip": True,
                        "skip_vqa": skip_vqa,
                        "reference_quality": "high",
                    },
                }
            )

    return {
        "scene_name": (scene_reference_map or {}).get("scene_name"),
        "reference_scope": "clip_reference_frame_agent",
        "naming_policy": (
            "category + final-frame relative position + rank, with optional "
            "distance bucket; color and YOLO are not primary naming inputs"
        ),
        "references": rows,
        "instance_token_to_reference": {
            row["instance_token"]: row for row in rows
        },
        "risk_agent_id_to_reference": {
            row["risk_agent_id"]: row for row in rows
        },
    }
