#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run generation validation for a LLaMA-Factory Qwen3-VL LoRA adapter."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--validation-json", default="training/llamafactory/data/nurisk_validation.json")
    parser.add_argument("--output-dir", default="training/outputs/validation/stage1_qwen3vl_json_sft_fields_validate20")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--nframes", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.validation_json, args.limit)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    predictions = []
    for index, row in enumerate(rows):
        label_text = row["messages"][1]["content"]
        label_json = parse_json(label_text)
        prompt_messages = build_messages(row, args)

        text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            videos=row["videos"],
            padding=True,
            return_tensors="pt",
            do_sample_frames=True,
            fps=1.0,
            max_frames=args.nframes,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        inputs.pop("video_metadata", None)
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, prompt_len:]
        prediction_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        prediction_json = parse_json(prediction_text)
        prediction_record = {
            "index": index,
            "prompt": row["messages"][0]["content"],
            "label": label_text,
            "prediction": prediction_text,
            "label_fields": extract_fields(label_json),
            "prediction_fields": extract_fields(prediction_json),
            "json_parse_ok": prediction_json is not None,
            "matches": compare_fields(extract_fields(label_json), extract_fields(prediction_json)),
        }
        predictions.append(prediction_record)
        print(f"[{index + 1}/{len(rows)}] parse={prediction_record['json_parse_ok']} matches={prediction_record['matches']}")

    metrics = summarize(predictions)
    write_json(output_dir / "predictions.json", predictions)
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def load_rows(path: str, limit: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows[:limit]


def build_messages(row: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    text = strip_video_markers(row["messages"][0]["content"])
    content = []
    for index, video_path in enumerate(row["videos"]):
        content.append(
            {
                "type": "video",
                "video": video_path,
                "nframes": args.nframes,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            }
        )
        content.append({"type": "text", "text": f"View {index + 1}: {infer_camera_name(video_path, index)}.\n"})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def strip_video_markers(text: str) -> str:
    lines = []
    for line in str(text).splitlines():
        if line.strip() == "<video>":
            continue
        if re.match(r"^View \d+:", line.strip()):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def infer_camera_name(path: str, index: int) -> str:
    upper_path = path.upper()
    for camera_name in CAMERA_ORDER:
        if camera_name in upper_path:
            return camera_name
    return f"CAMERA_{index + 1}"


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = str(text).strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_fields(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    return {
        "current_risk_score": get_path(obj, "current_risk", "risk_score"),
        "future_worst_risk_score": first_path(
            obj,
            ("future_worst_risk", "risk_score"),
            ("predicted_future_worst_risk", "risk_score"),
        ),
        "risk_trend": first_path(obj, ("risk_change", "trend"), ("risk_change_analysis", "trend")),
        "target_agent_role": obj.get("target_agent_role"),
        "longitudinal_action": first_path(
            obj,
            ("ego_meta_action", "longitudinal"),
            ("scene_consistent_ego_mitigation", "ego_meta_action", "longitudinal"),
        ),
        "lateral_action": first_path(
            obj,
            ("ego_meta_action", "lateral"),
            ("scene_consistent_ego_mitigation", "ego_meta_action", "lateral"),
        ),
    }


def get_path(obj: Dict[str, Any], *path: str) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_path(obj: Dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = get_path(obj, *path)
        if value is not None:
            return value
    return None


def compare_fields(label_fields: Dict[str, Any], prediction_fields: Dict[str, Any]) -> Dict[str, bool]:
    keys = (
        "current_risk_score",
        "future_worst_risk_score",
        "risk_trend",
        "target_agent_role",
        "longitudinal_action",
        "lateral_action",
    )
    return {key: label_fields.get(key) == prediction_fields.get(key) for key in keys}


def summarize(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    parse_ok = sum(1 for item in predictions if item["json_parse_ok"])
    field_keys = (
        "current_risk_score",
        "future_worst_risk_score",
        "risk_trend",
        "target_agent_role",
        "longitudinal_action",
        "lateral_action",
    )
    metrics = {
        "num_samples": total,
        "json_parse_rate": safe_div(parse_ok, total),
        "field_accuracy": {},
    }
    for key in field_keys:
        metrics["field_accuracy"][key] = safe_div(
            sum(1 for item in predictions if item["matches"].get(key)),
            total,
        )
    return metrics


def safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
