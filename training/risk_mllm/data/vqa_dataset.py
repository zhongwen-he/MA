#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset reader for clean multi-view NuRisk-style VQA JSON files.

This reader intentionally does not decode videos. It returns resolved video
paths plus question/answer text so a downstream trainer can choose its own
video processor.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class NuRiskVQADataset:
    """Lightweight reader for Stage 7/8 clean VQA samples."""

    def __init__(
        self,
        json_path: str,
        vqa_root: Optional[str] = None,
        answer_label_policy: str = "full",
    ) -> None:
        self.json_path = Path(json_path).expanduser().resolve()
        self.vqa_root = Path(vqa_root).expanduser().resolve() if vqa_root else self.json_path.parent
        self.answer_label_policy = answer_label_policy
        self.entries = self._load_entries(self.json_path)

    @staticmethod
    def _load_entries(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data["entries"]
        raise ValueError(f"Unsupported VQA JSON format: {path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        conversations = entry.get("conversations", [])
        if len(conversations) < 2:
            raise ValueError(f"Entry {entry.get('id', index)} has fewer than two conversation turns")

        video_paths = [self._resolve_video_path(path) for path in entry.get("video", [])]
        answer_text = conversations[1].get("value", "")
        try:
            answer_json = json.loads(answer_text)
        except json.JSONDecodeError:
            answer_json = None
        explanations = self._first_path_dict(answer_json, ("explanations",)) or {}
        reasoning_summary = answer_json.get("reasoning_summary", "") if isinstance(answer_json, dict) else ""
        risk_targets = self._extract_risk_targets(answer_json)
        cot_steps = self._build_cot_steps(answer_json, entry)
        training_answer = self._build_training_answer(answer_json, answer_text)

        return {
            "id": entry.get("id"),
            "scene": entry.get("scene"),
            "clip_id": entry.get("clip_id"),
            "agent_id": entry.get("agent_id"),
            "target_vehicle": entry.get("target_vehicle"),
            "canonical_agent_name": entry.get("canonical_agent_name"),
            "clip_reference_name": entry.get("clip_reference_name"),
            "video_paths": video_paths,
            "question": conversations[0].get("value", ""),
            "answer": training_answer,
            "answer_json": answer_json,
            "explanations": explanations,
            "reasoning_summary": reasoning_summary,
            "cot_steps": cot_steps,
            "risk_targets": risk_targets,
        }

    def _resolve_video_path(self, path: str) -> str:
        video_path = Path(path)
        if video_path.is_absolute():
            return str(video_path)
        resolved_path = (self.vqa_root / video_path).resolve()
        if resolved_path.exists():
            return str(resolved_path)

        parts = video_path.parts
        if parts and parts[0] == "videos":
            local_video_path = (self.vqa_root / "video_clip_dataset" / video_path).resolve()
            if local_video_path.exists():
                return str(local_video_path)

            sibling_video_path = (self.vqa_root.parent / "video_clip_dataset" / video_path).resolve()
            if sibling_video_path.exists():
                return str(sibling_video_path)

            raw_video_path = (self.vqa_root / "raw_video_clip_dataset_v1" / video_path).resolve()
            if raw_video_path.exists():
                return str(raw_video_path)

        return str(resolved_path)

    def _extract_risk_targets(self, answer_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(answer_json, dict):
            return {}

        targets = {
            "current_risk_score": self._get_path(answer_json, "current_risk", "risk_score"),
            "future_worst_risk_score": self._first_path(
                answer_json,
                ("future_worst_risk", "risk_score"),
                ("predicted_future_worst_risk", "risk_score"),
            ),
            "risk_trend": self._first_path(
                answer_json,
                ("risk_change", "trend"),
                ("risk_change_analysis", "trend"),
            ),
            "mitigation_action": self._first_path(
                answer_json,
                ("ego_meta_action", "longitudinal"),
                ("scene_consistent_ego_mitigation", "ego_meta_action", "longitudinal"),
            ),
            "lateral_mitigation_action": self._first_path(
                answer_json,
                ("ego_meta_action", "lateral"),
                ("scene_consistent_ego_mitigation", "ego_meta_action", "lateral"),
            ),
            "recommended_ego_speed_mps": self._first_path(
                answer_json,
                ("quantitative_suggestion", "target_speed_mps"),
                ("scene_consistent_ego_mitigation", "quantitative_suggestion", "target_speed_mps"),
            ),
            "suggested_deceleration_mps2": self._first_path(
                answer_json,
                ("quantitative_suggestion", "target_acceleration_mps2"),
                ("scene_consistent_ego_mitigation", "quantitative_suggestion", "target_acceleration_mps2"),
            ),
            "target_lateral_offset_m": self._first_path(
                answer_json,
                ("quantitative_suggestion", "target_lateral_offset_m"),
                ("scene_consistent_ego_mitigation", "quantitative_suggestion", "target_lateral_offset_m"),
            ),
            "current_ttc_longitudinal": self._first_path(
                answer_json,
                ("current_risk", "time_to_collision", "longitudinal"),
                ("current_risk", "time_to_collision", "Longitudinal"),
            ),
            "current_ttc_lateral": self._first_path(
                answer_json,
                ("current_risk", "time_to_collision", "lateral"),
                ("current_risk", "time_to_collision", "Lateral"),
            ),
            "current_dtc_longitudinal": self._first_path(
                answer_json,
                ("current_risk", "distance_to_collision", "longitudinal"),
                ("current_risk", "distance_to_collision", "Longitudinal"),
            ),
            "current_dtc_lateral": self._first_path(
                answer_json,
                ("current_risk", "distance_to_collision", "lateral"),
                ("current_risk", "distance_to_collision", "Lateral"),
            ),
            "future_ttc": self._first_path(
                answer_json,
                ("future_worst_risk", "time_to_collision", "longitudinal"),
                ("predicted_future_worst_risk", "time_to_collision", "longitudinal"),
            ),
            "future_ttc_lateral": self._first_path(
                answer_json,
                ("future_worst_risk", "time_to_collision", "lateral"),
                ("future_worst_risk", "time_to_collision", "Lateral"),
                ("predicted_future_worst_risk", "time_to_collision", "lateral"),
                ("predicted_future_worst_risk", "time_to_collision", "Lateral"),
            ),
            "future_dtc": self._first_path(
                answer_json,
                ("future_worst_risk", "distance_to_collision", "longitudinal"),
                ("predicted_future_worst_risk", "distance_to_collision", "longitudinal"),
            ),
            "future_dtc_lateral": self._first_path(
                answer_json,
                ("future_worst_risk", "distance_to_collision", "lateral"),
                ("future_worst_risk", "distance_to_collision", "Lateral"),
                ("predicted_future_worst_risk", "distance_to_collision", "lateral"),
                ("predicted_future_worst_risk", "distance_to_collision", "Lateral"),
            ),
        }

        field_candidates = {
            "current_risk_score": (
                "current_risk_score",
                "current_score",
                "observed_risk_score",
                "risk_score_current",
            ),
            "future_worst_risk_score": (
                "future_worst_risk_score",
                "future_worst_score",
                "future_risk_score",
                "predicted_future_risk_score",
                "risk_score_future",
            ),
            "risk_trend": (
                "risk_trend",
                "trend",
                "risk_change",
                "future_risk_trend",
            ),
            "mitigation_action": (
                "mitigation_action",
                "recommended_action",
                "action",
                "ego_action",
            ),
            "lateral_mitigation_action": (
                "lateral_mitigation_action",
                "recommended_lateral_action",
                "lateral_action",
                "ego_lateral_action",
            ),
            "recommended_ego_speed_mps": (
                "recommended_ego_speed_mps",
                "recommended_speed_mps",
                "recommended_speed",
                "target_speed_mps",
            ),
            "suggested_deceleration_mps2": (
                "suggested_deceleration_mps2",
                "recommended_deceleration_mps2",
                "target_deceleration_mps2",
                "deceleration",
            ),
            "target_lateral_offset_m": (
                "target_lateral_offset_m",
                "recommended_lateral_offset_m",
                "lateral_offset_m",
            ),
            "current_ttc_longitudinal": (
                "current_ttc_longitudinal",
                "current_ttc_s",
            ),
            "current_ttc_lateral": (
                "current_ttc_lateral",
                "current_lateral_ttc_s",
            ),
            "current_dtc_longitudinal": (
                "current_dtc_longitudinal",
                "current_dtc_m",
            ),
            "current_dtc_lateral": (
                "current_dtc_lateral",
                "current_lateral_dtc_m",
            ),
            "future_ttc": (
                "future_ttc",
                "future_ttc_s",
                "predicted_ttc",
                "ttc",
            ),
            "future_ttc_lateral": (
                "future_ttc_lateral",
                "future_lateral_ttc_s",
            ),
            "future_dtc": (
                "future_dtc",
                "future_dtc_m",
                "future_dtc_longitudinal",
                "predicted_dtc",
                "dtc",
            ),
            "future_dtc_lateral": (
                "future_dtc_lateral",
                "future_lateral_dtc_m",
            ),
        }

        for target_name, candidates in field_candidates.items():
            if targets.get(target_name) is None:
                targets[target_name] = self._find_first(answer_json, candidates)
        return {key: value for key, value in targets.items() if value is not None}

    def _build_training_answer(self, answer_json: Optional[Dict[str, Any]], fallback_text: str) -> str:
        """Return the inference-time JSON target without explicit CoT text."""

        if not isinstance(answer_json, dict):
            return fallback_text
        if self.answer_label_policy not in {
            "full",
            "head_placeholder_json",
            "stage2_compact_placeholder_json",
        }:
            raise ValueError(f"Unsupported answer_label_policy: {self.answer_label_policy}")

        compact_answer = dict(answer_json)
        for explanation_key in (
            "reasoning_summary",
            "explanations",
            "mitigation_explanation",
            "chain_of_thought",
            "cot_steps",
        ):
            compact_answer.pop(explanation_key, None)
        if self.answer_label_policy == "head_placeholder_json":
            compact_answer = self._replace_head_owned_values_with_null(compact_answer)
        elif self.answer_label_policy == "stage2_compact_placeholder_json":
            compact_answer = self._build_stage2_compact_placeholder_answer(answer_json)
        return json.dumps(compact_answer, ensure_ascii=False, indent=2)

    def _build_stage2_compact_placeholder_answer(self, answer_json: Dict[str, Any]) -> Dict[str, Any]:
        """Build a short Stage-2 JSON target for visual/Q-Former answer alignment.

        Stage 2 should teach the visual interface to support the final answer
        structure, while later risk heads learn the actual numeric values and
        fixed meta-action classes.  Therefore this target deliberately removes
        long/redundant fields and replaces head-owned fields with ``None``.
        """

        current = answer_json.get("current_risk", {})
        future = self._first_path_dict(
            answer_json,
            ("future_worst_risk",),
            ("predicted_future_worst_risk",),
        ) or {}
        change = self._first_path_dict(
            answer_json,
            ("risk_change",),
            ("risk_change_analysis",),
        ) or {}

        target_name = (
            answer_json.get("agent_id")
            or answer_json.get("clip_reference_name")
            or answer_json.get("target_vehicle")
            or answer_json.get("canonical_agent_name")
            or "the target agent"
        )

        compact: Dict[str, Any] = {
            "agent_id": target_name,
            "current_risk": {
                "frame": current.get("frame") if isinstance(current, dict) else None,
                "relative_direction": current.get("relative_direction") if isinstance(current, dict) else None,
                "distance_to_collision": {
                    "longitudinal": None,
                    "lateral": None,
                },
                "time_to_collision": {
                    "longitudinal": None,
                    "lateral": None,
                },
                "risk_score": None,
                "risk_level": None,
            },
            "future_worst_risk": {
                "horizon_seconds": future.get("horizon_seconds") if isinstance(future, dict) else None,
                "available": future.get("available", True) if isinstance(future, dict) else None,
                "relative_direction": future.get("relative_direction") if isinstance(future, dict) else None,
                "distance_to_collision": {
                    "longitudinal": None,
                    "lateral": None,
                },
                "time_to_collision": {
                    "longitudinal": None,
                    "lateral": None,
                },
                "risk_score": None,
                "risk_level": None,
            },
            "risk_change": {
                "current_score": None,
                "future_worst_score": None,
                "delta_score": None,
                "trend": None,
                "future_intervention_required": None,
            },
            "target_agent_role": answer_json.get("target_agent_role"),
            "ego_meta_action": {
                "longitudinal": None,
                "lateral": None,
            },
            "quantitative_suggestion": {
                "duration_s": self._first_path(
                    answer_json,
                    ("quantitative_suggestion", "duration_s"),
                    ("scene_consistent_ego_mitigation", "quantitative_suggestion", "duration_s"),
                ),
                "current_speed_mps": None,
                "target_speed_mps": None,
                "target_acceleration_mps2": None,
                "target_lateral_offset_m": None,
            },
        }

        if compact["future_worst_risk"]["horizon_seconds"] is None:
            compact["future_worst_risk"]["horizon_seconds"] = 0.5
        if compact["quantitative_suggestion"]["duration_s"] is None:
            compact["quantitative_suggestion"]["duration_s"] = compact["future_worst_risk"]["horizon_seconds"]
        return compact

    def _replace_head_owned_values_with_null(self, answer_json: Dict[str, Any]) -> Dict[str, Any]:
        """Mask fields that are intended to be filled by risk heads at inference."""

        masked = json.loads(json.dumps(answer_json, ensure_ascii=False))
        head_owned_paths = (
            ("current_risk", "risk_score"),
            ("current_risk", "risk_level"),
            ("current_risk", "time_to_collision", "longitudinal"),
            ("current_risk", "time_to_collision", "lateral"),
            ("current_risk", "time_to_collision", "Longitudinal"),
            ("current_risk", "time_to_collision", "Lateral"),
            ("current_risk", "distance_to_collision", "longitudinal"),
            ("current_risk", "distance_to_collision", "lateral"),
            ("current_risk", "distance_to_collision", "Longitudinal"),
            ("current_risk", "distance_to_collision", "Lateral"),
            ("future_worst_risk", "risk_score"),
            ("future_worst_risk", "risk_level"),
            ("future_worst_risk", "time_to_collision", "longitudinal"),
            ("future_worst_risk", "time_to_collision", "lateral"),
            ("future_worst_risk", "time_to_collision", "Longitudinal"),
            ("future_worst_risk", "time_to_collision", "Lateral"),
            ("future_worst_risk", "distance_to_collision", "longitudinal"),
            ("future_worst_risk", "distance_to_collision", "lateral"),
            ("future_worst_risk", "distance_to_collision", "Longitudinal"),
            ("future_worst_risk", "distance_to_collision", "Lateral"),
            ("predicted_future_worst_risk", "risk_score"),
            ("predicted_future_worst_risk", "risk_level"),
            ("predicted_future_worst_risk", "time_to_collision", "longitudinal"),
            ("predicted_future_worst_risk", "time_to_collision", "lateral"),
            ("predicted_future_worst_risk", "time_to_collision", "Longitudinal"),
            ("predicted_future_worst_risk", "time_to_collision", "Lateral"),
            ("predicted_future_worst_risk", "distance_to_collision", "longitudinal"),
            ("predicted_future_worst_risk", "distance_to_collision", "lateral"),
            ("predicted_future_worst_risk", "distance_to_collision", "Longitudinal"),
            ("predicted_future_worst_risk", "distance_to_collision", "Lateral"),
            ("risk_change", "current_score"),
            ("risk_change", "future_worst_score"),
            ("risk_change", "delta_score"),
            ("risk_change", "trend"),
            ("risk_change_analysis", "current_score"),
            ("risk_change_analysis", "future_worst_score"),
            ("risk_change_analysis", "delta_score"),
            ("risk_change_analysis", "trend"),
            ("ego_meta_action", "longitudinal"),
            ("ego_meta_action", "lateral"),
            ("scene_consistent_ego_mitigation", "ego_meta_action", "longitudinal"),
            ("scene_consistent_ego_mitigation", "ego_meta_action", "lateral"),
            ("quantitative_suggestion", "target_speed_mps"),
            ("quantitative_suggestion", "target_acceleration_mps2"),
            ("quantitative_suggestion", "target_lateral_offset_m"),
            ("scene_consistent_ego_mitigation", "quantitative_suggestion", "target_speed_mps"),
            ("scene_consistent_ego_mitigation", "quantitative_suggestion", "target_acceleration_mps2"),
            ("scene_consistent_ego_mitigation", "quantitative_suggestion", "target_lateral_offset_m"),
        )
        for path in head_owned_paths:
            self._set_path_if_present(masked, path, None)
        return masked

    def _set_path_if_present(self, data: Dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        cursor: Any = data
        for key in path[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                return
            cursor = cursor[key]
        if isinstance(cursor, dict) and path[-1] in cursor:
            cursor[path[-1]] = value

    def _build_cot_steps(
        self,
        answer_json: Optional[Dict[str, Any]],
        entry: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Construct four SIM-CoT style risk reasoning targets from structured JSON."""

        if not isinstance(answer_json, dict):
            return ["", "", "", ""]
        entry = entry or {}

        current = answer_json.get("current_risk", {})
        future = self._first_path_dict(
            answer_json,
            ("predicted_future_worst_risk",),
            ("future_worst_risk",),
        ) or {}
        change = self._first_path_dict(
            answer_json,
            ("risk_change_analysis",),
            ("risk_change",),
        ) or {}
        meta_action = self._first_path_dict(
            answer_json,
            ("scene_consistent_ego_mitigation", "ego_meta_action"),
            ("ego_meta_action",),
        ) or {}
        quantitative = self._first_path_dict(
            answer_json,
            ("scene_consistent_ego_mitigation", "quantitative_suggestion"),
            ("quantitative_suggestion",),
        ) or {}
        future_intervention_required = self._first_path(
            answer_json,
            ("risk_change_analysis", "is_future_intervention_required"),
            ("risk_change_analysis", "future_intervention_required"),
            ("risk_change", "is_future_intervention_required"),
            ("risk_change", "future_intervention_required"),
        )
        action_rationale = self._extract_action_rationale(answer_json)

        target_name = (
            answer_json.get("clip_reference_name")
            or answer_json.get("target_vehicle")
            or answer_json.get("agent_id")
            or answer_json.get("canonical_agent_name")
            or entry.get("clip_reference_name")
            or entry.get("target_vehicle")
            or entry.get("agent_id")
            or entry.get("canonical_agent_name")
            or "the target agent"
        )
        current_dtc = current.get("distance_to_collision", {}) if isinstance(current, dict) else {}
        current_ttc = current.get("time_to_collision", {}) if isinstance(current, dict) else {}
        future_dtc = future.get("distance_to_collision", {}) if isinstance(future, dict) else {}
        future_ttc = future.get("time_to_collision", {}) if isinstance(future, dict) else {}

        future_available = future.get("available", True) if isinstance(future, dict) else False
        if future_available:
            future_prefix = (
                f"within {self._fmt(future.get('horizon_seconds'))} seconds, "
                f"the worst future risk is {self._fmt(future.get('risk_level'))} "
                f"with score {self._fmt(future.get('risk_score'))}"
            )
        else:
            future_prefix = "future risk is unavailable within the selected horizon"

        return [
            (
                f"Current risk: {target_name} is {self._fmt(current.get('relative_direction'))}; "
                f"current DTC is longitudinal={self._fmt(current_dtc.get('longitudinal', current_dtc.get('Longitudinal')))} m "
                f"and lateral={self._fmt(current_dtc.get('lateral', current_dtc.get('Lateral')))} m; "
                f"current TTC is longitudinal={self._fmt(current_ttc.get('longitudinal', current_ttc.get('Longitudinal')))} s "
                f"and lateral={self._fmt(current_ttc.get('lateral', current_ttc.get('Lateral')))} s; "
                f"current risk is {self._fmt(current.get('risk_level'))} with score {self._fmt(current.get('risk_score'))}."
            ),
            (
                f"Future risk: {future_prefix}; "
                f"future DTC is longitudinal={self._fmt(future_dtc.get('longitudinal', future_dtc.get('Longitudinal')))} m "
                f"and lateral={self._fmt(future_dtc.get('lateral', future_dtc.get('Lateral')))} m; "
                f"future TTC is longitudinal={self._fmt(future_ttc.get('longitudinal', future_ttc.get('Longitudinal')))} s "
                f"and lateral={self._fmt(future_ttc.get('lateral', future_ttc.get('Lateral')))} s."
            ),
            (
                f"Risk trend: current score is {self._fmt(change.get('current_score'))}, "
                f"future worst score is {self._fmt(change.get('future_worst_score'))}, "
                f"delta score is {self._fmt(change.get('delta_score'))}, "
                f"trend is {self._fmt(change.get('trend'))}, "
                f"and future intervention required is {self._fmt(future_intervention_required)}."
            ),
            (
                f"Mitigation: target role is {self._fmt(answer_json.get('target_agent_role'))}; "
                f"ego action is longitudinal={self._fmt(meta_action.get('longitudinal'))}, "
                f"lateral={self._fmt(meta_action.get('lateral'))}; "
                f"target speed is {self._fmt(quantitative.get('target_speed_mps'))} m/s, "
                f"target acceleration is {self._fmt(quantitative.get('target_acceleration_mps2'))} m/s^2, "
                f"and target lateral offset is {self._fmt(quantitative.get('target_lateral_offset_m'))} m"
                f"{self._fmt_action_rationale(action_rationale)}."
            ),
        ]

    def _extract_action_rationale(self, answer_json: Dict[str, Any]) -> str:
        rationales: List[str] = []
        explanations = answer_json.get("explanations", {})
        if isinstance(explanations, dict):
            self._append_unique_text(rationales, explanations.get("action"))

        self._append_unique_text(rationales, answer_json.get("mitigation_explanation"))

        reasoning_summary = answer_json.get("reasoning_summary")
        if not rationales and isinstance(reasoning_summary, str):
            action_match = re.search(r"(Action:\s*.+)$", reasoning_summary, flags=re.IGNORECASE)
            self._append_unique_text(rationales, action_match.group(1) if action_match else reasoning_summary)

        return " ".join(rationales)

    def _append_unique_text(self, values: List[str], candidate: Any) -> None:
        if not isinstance(candidate, str):
            return
        text = " ".join(candidate.split())
        if text and text not in values:
            values.append(text)

    def _fmt_action_rationale(self, rationale: str) -> str:
        if not rationale:
            return ""
        return f"; action rationale: {rationale.rstrip('. ')}"

    def _get_path(self, data: Dict[str, Any], *path: str) -> Any:
        value: Any = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    def _first_path(self, data: Dict[str, Any], *paths: tuple[str, ...]) -> Any:
        for path in paths:
            value = self._get_path(data, *path)
            if value is not None:
                return value
        return None

    def _first_path_dict(self, data: Dict[str, Any], *paths: tuple[str, ...]) -> Optional[Dict[str, Any]]:
        value = self._first_path(data, *paths)
        return value if isinstance(value, dict) else None

    def _find_first(self, data: Any, candidates: tuple[str, ...]) -> Any:
        normalized_candidates = {self._normalize_key(candidate) for candidate in candidates}
        if isinstance(data, dict):
            for key, value in data.items():
                if self._normalize_key(str(key)) in normalized_candidates:
                    if isinstance(value, dict) and value.get("longitudinal") is not None:
                        return value["longitudinal"]
                    return value
            for value in data.values():
                found = self._find_first(value, candidates)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for item in data:
                found = self._find_first(item, candidates)
                if found is not None:
                    return found
        return None

    def _normalize_key(self, key: str) -> str:
        return re.sub(r"[^a-z0-9]", "", key.lower())

    def _fmt(self, value: Any) -> str:
        if value is None:
            return "unknown"
        if isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)
