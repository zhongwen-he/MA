#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scene-local agent alias helpers for annotated video/VQA generation.

The pipeline keeps raw nuScenes identities as durable keys:

    obstacle_id / instance_token
    risk agent id = "Obstacle <instance_token>"

Video overlays and VQA prompts use short scene-local display ids such as A003.
Every display id must be resolved through the scene alias map before joining
with raw nuScenes tables or NuRisk-style risk outputs.
"""

from typing import Any, Dict, Iterable, List, Optional


RISK_AGENT_PREFIX = "Obstacle "


def display_agent_id(index: int) -> str:
    return f"A{index:03d}"


def scene_agent_id(scene_name: str, display_id: str) -> str:
    return f"{scene_name}_{display_id}"


def risk_agent_id(instance_token: str) -> str:
    return f"{RISK_AGENT_PREFIX}{instance_token}"


def instance_token_from_risk_agent_id(agent_id: str) -> str:
    if agent_id.startswith(RISK_AGENT_PREFIX):
        return agent_id[len(RISK_AGENT_PREFIX) :]
    return agent_id


def build_scene_agent_alias_map(
    scene_name: str,
    samples: Iterable[Dict[str, Any]],
    ann_by_token: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one stable alias map for an entire scene.

    Aliases are assigned in first-observed sample/annotation order and are
    reused by every clip from the scene.
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

    aliases = []
    for index, instance_token in enumerate(instance_tokens, start=1):
        display_id = display_agent_id(index)
        aliases.append(
            {
                "agent_id": display_id,
                "display_agent_id": display_id,
                "scene_agent_id": scene_agent_id(scene_name, display_id),
                "instance_token": instance_token,
                "raw_agent_id": risk_agent_id(instance_token),
                "risk_agent_id": risk_agent_id(instance_token),
                "alias_scope": "scene",
            }
        )

    return {
        "scene_name": scene_name,
        "alias_scope": "scene",
        "id_format": "A%03d",
        "raw_id_type": "nuScenes instance_token",
        "risk_agent_id_format": "Obstacle <instance_token>",
        "aliases": aliases,
        "instance_token_to_agent_id": {
            row["instance_token"]: row["display_agent_id"] for row in aliases
        },
        "risk_agent_id_to_agent_id": {
            row["risk_agent_id"]: row["display_agent_id"] for row in aliases
        },
    }


def alias_lookup(agent_alias_map: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not agent_alias_map:
        return {}
    lookup = {}
    for row in agent_alias_map.get("aliases", []):
        lookup[row["risk_agent_id"]] = row
        lookup[row["instance_token"]] = row
        lookup[row["display_agent_id"]] = row
        lookup[row["scene_agent_id"]] = row
    return lookup


def identity_for_risk_agent(
    raw_agent_id: str,
    agent_alias_map: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    instance_token = instance_token_from_risk_agent_id(raw_agent_id)
    lookup = alias_lookup(agent_alias_map)
    row = lookup.get(raw_agent_id) or lookup.get(instance_token)
    if row is None:
        display_id = raw_agent_id
        return {
            "agent_id": display_id,
            "display_agent_id": display_id,
            "scene_agent_id": display_id,
            "raw_agent_id": raw_agent_id,
            "risk_agent_id": raw_agent_id,
            "instance_token": instance_token,
            "alias_scope": "unmapped",
        }
    return {
        "agent_id": row["display_agent_id"],
        "display_agent_id": row["display_agent_id"],
        "scene_agent_id": row["scene_agent_id"],
        "raw_agent_id": row["risk_agent_id"],
        "risk_agent_id": row["risk_agent_id"],
        "instance_token": row["instance_token"],
        "alias_scope": row.get("alias_scope", "scene"),
    }


def attach_identity(
    payload: Dict[str, Any],
    raw_agent_id: str,
    agent_alias_map: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload.update(identity_for_risk_agent(raw_agent_id, agent_alias_map))
    return payload
