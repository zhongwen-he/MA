#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Assemble final structured JSON answers from Qwen drafts and risk heads.

The final answer path is:

    Qwen native LM draft JSON
        + recurrent-latent-CoT risk-head predictions
        -> one structured JSON answer

The CoT decoder is training-only and is intentionally not used here.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor

from risk_mllm.models.risk_latent_mllm import RiskLatentConfig


RISK_SCORE_TO_LEVEL = {
    0: "Collision Risk",
    1: "Extreme Risk",
    2: "High Risk",
    3: "Medium Risk",
    4: "Low Risk",
    5: "Negligible Risk",
}


class StructuredAnswerAssembler:
    """Overwrite head-owned JSON fields with auxiliary-head predictions."""

    def __init__(self, config: RiskLatentConfig) -> None:
        self.config = config

    def assemble_one(
        self,
        draft_answer: str | Mapping[str, Any],
        aux_outputs: Mapping[str, Any],
        batch_index: int = 0,
    ) -> Dict[str, Any]:
        answer = self._parse_draft(draft_answer)
        logits = aux_outputs.get("logits", {})
        predictions = aux_outputs.get("predictions", {})

        current_score = self._class_prediction(logits, "current_risk_score", batch_index)
        future_score = self._class_prediction(logits, "future_worst_risk_score", batch_index)
        trend_id = self._class_prediction(logits, "risk_trend", batch_index)
        longitudinal_action_id = self._class_prediction(logits, "mitigation_action", batch_index)
        lateral_action_id = self._class_prediction(logits, "lateral_mitigation_action", batch_index)

        current_ttc = self._vector_prediction(predictions, "current_ttc", batch_index, size=2)
        current_dtc = self._vector_prediction(predictions, "current_dtc", batch_index, size=2)
        future_ttc = self._vector_prediction(predictions, "future_ttc_vector", batch_index, size=2)
        future_dtc = self._vector_prediction(predictions, "future_dtc_vector", batch_index, size=2)
        if future_ttc is None:
            scalar_future_ttc = self._scalar_prediction(predictions, "future_ttc", batch_index)
            future_ttc = [scalar_future_ttc, None]
        if future_dtc is None:
            scalar_future_dtc = self._scalar_prediction(predictions, "future_dtc", batch_index)
            future_dtc = [scalar_future_dtc, None]

        current_ttc = self._apply_finite_logits(logits, "current_ttc_finite", current_ttc, batch_index)
        future_ttc = self._apply_finite_logits(logits, "future_ttc_finite", future_ttc, batch_index)

        self._write_risk_block(answer, ("current_risk",), current_score, current_ttc, current_dtc)
        future_path = ("predicted_future_worst_risk",) if "predicted_future_worst_risk" in answer else ("future_worst_risk",)
        self._write_risk_block(answer, future_path, future_score, future_ttc, future_dtc)

        trend = self._label_from_id(trend_id, self.config.trend_labels)
        self._write_change_block(answer, current_score, future_score, trend)

        longitudinal_action = self._label_from_id(longitudinal_action_id, self.config.action_labels)
        lateral_action = self._label_from_id(lateral_action_id, self.config.lateral_action_labels)
        speed = self._scalar_prediction(predictions, "recommended_ego_speed_mps", batch_index)
        acceleration = self._scalar_prediction(predictions, "suggested_deceleration_mps2", batch_index)
        lateral_offset = self._scalar_prediction(predictions, "target_lateral_offset_m", batch_index)
        self._write_action_block(answer, longitudinal_action, lateral_action, speed, acceleration, lateral_offset)

        return answer

    def _parse_draft(self, draft_answer: str | Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(draft_answer, Mapping):
            return copy.deepcopy(dict(draft_answer))
        try:
            parsed = json.loads(draft_answer)
        except json.JSONDecodeError:
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}

    def _write_risk_block(
        self,
        answer: Dict[str, Any],
        path: Sequence[str],
        score: Optional[int],
        ttc: Optional[Sequence[Any]],
        dtc: Optional[Sequence[Any]],
    ) -> None:
        block = self._ensure_dict_path(answer, path)
        if score is not None:
            block["risk_score"] = score
            block["risk_level"] = RISK_SCORE_TO_LEVEL.get(score, "Unknown Risk")
        if ttc is not None:
            ttc_block = self._ensure_child_dict(block, "time_to_collision")
            self._write_long_lat(ttc_block, ttc)
        if dtc is not None:
            dtc_block = self._ensure_child_dict(block, "distance_to_collision")
            self._write_long_lat(dtc_block, dtc)

    def _write_change_block(
        self,
        answer: Dict[str, Any],
        current_score: Optional[int],
        future_score: Optional[int],
        trend: Optional[str],
    ) -> None:
        key = "risk_change_analysis" if "risk_change_analysis" in answer else "risk_change"
        change = self._ensure_dict_path(answer, (key,))
        if current_score is not None:
            change["current_score"] = current_score
        if future_score is not None:
            change["future_worst_score"] = future_score
        if current_score is not None and future_score is not None:
            change["delta_score"] = future_score - current_score
        if trend is not None:
            change["trend"] = trend

    def _write_action_block(
        self,
        answer: Dict[str, Any],
        longitudinal_action: Optional[str],
        lateral_action: Optional[str],
        speed: Optional[float],
        acceleration: Optional[float],
        lateral_offset: Optional[float],
    ) -> None:
        action_paths = [("ego_meta_action",)]
        if "scene_consistent_ego_mitigation" in answer:
            action_paths.append(("scene_consistent_ego_mitigation", "ego_meta_action"))
        for path in action_paths:
            action = self._ensure_dict_path(answer, path)
            if longitudinal_action is not None:
                action["longitudinal"] = longitudinal_action
            if lateral_action is not None:
                action["lateral"] = lateral_action

        quantitative_paths = [("quantitative_suggestion",)]
        if "scene_consistent_ego_mitigation" in answer:
            quantitative_paths.append(("scene_consistent_ego_mitigation", "quantitative_suggestion"))
        for path in quantitative_paths:
            quantitative = self._ensure_dict_path(answer, path)
            if speed is not None:
                quantitative["target_speed_mps"] = speed
            if acceleration is not None:
                quantitative["target_acceleration_mps2"] = acceleration
            if lateral_offset is not None:
                quantitative["target_lateral_offset_m"] = lateral_offset

    def _write_long_lat(self, block: Dict[str, Any], values: Sequence[Any]) -> None:
        if len(values) > 0 and values[0] is not None:
            block["longitudinal"] = values[0]
        if len(values) > 1 and values[1] is not None:
            block["lateral"] = values[1]

    def _class_prediction(
        self,
        logits: Mapping[str, Any],
        key: str,
        batch_index: int,
    ) -> Optional[int]:
        tensor = logits.get(key)
        if not isinstance(tensor, Tensor):
            return None
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if batch_index >= tensor.size(0):
            return None
        return int(torch.argmax(tensor[batch_index]).detach().cpu())

    def _scalar_prediction(
        self,
        predictions: Mapping[str, Any],
        key: str,
        batch_index: int,
    ) -> Optional[float]:
        tensor = predictions.get(key)
        if not isinstance(tensor, Tensor):
            return None
        flat = tensor.detach().float().cpu()
        if flat.dim() == 0:
            value = float(flat)
        elif batch_index < flat.size(0):
            value = float(flat[batch_index].reshape(-1)[0])
        else:
            return None
        return self._json_number(value)

    def _vector_prediction(
        self,
        predictions: Mapping[str, Any],
        key: str,
        batch_index: int,
        size: int,
    ) -> Optional[list[Any]]:
        tensor = predictions.get(key)
        if not isinstance(tensor, Tensor):
            return None
        values = tensor.detach().float().cpu()
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if batch_index >= values.size(0):
            return None
        row = values[batch_index].reshape(-1)
        return [self._json_number(float(row[index])) for index in range(min(size, row.numel()))]

    def _apply_finite_logits(
        self,
        logits: Mapping[str, Any],
        key: str,
        values: Optional[list[Any]],
        batch_index: int,
    ) -> Optional[list[Any]]:
        if values is None:
            return None
        tensor = logits.get(key)
        if not isinstance(tensor, Tensor):
            return values
        finite_logits = tensor.detach().float().cpu()
        if finite_logits.dim() == 2:
            finite_logits = finite_logits.unsqueeze(0)
        if batch_index >= finite_logits.size(0):
            return values
        classes = torch.argmax(finite_logits[batch_index], dim=-1).tolist()
        output = list(values)
        for index, finite_class in enumerate(classes[: len(output)]):
            if int(finite_class) == 0:
                output[index] = "Infinity"
        return output

    def _label_from_id(self, index: Optional[int], labels: Sequence[str]) -> Optional[str]:
        if index is None or index < 0 or index >= len(labels):
            return None
        return labels[index]

    def _ensure_dict_path(self, root: Dict[str, Any], path: Sequence[str]) -> Dict[str, Any]:
        cursor = root
        for key in path:
            child = cursor.get(key)
            if not isinstance(child, dict):
                child = {}
                cursor[key] = child
            cursor = child
        return cursor

    def _ensure_child_dict(self, root: Dict[str, Any], key: str) -> Dict[str, Any]:
        child = root.get(key)
        if not isinstance(child, dict):
            child = {}
            root[key] = child
        return child

    def _json_number(self, value: float) -> Any:
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if math.isnan(value):
            return None
        rounded = round(value, 2)
        return 0.0 if rounded == -0.0 else rounded
