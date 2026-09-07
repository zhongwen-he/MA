#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 8: create train/validation splits for clean Stage 7 VQA conversations.

Stage 7 now emits one VQA sample per target agent, while all target-agent
samples from the same clip share one scene-consistent ego meta-action label.
The default split unit is therefore clip, so samples with the same clip_id stay
in the same split and the shared scene-level mitigation label is not duplicated
across train and validation.

The script validates that training samples expose video paths as inputs while
keeping nuScenes tokens and raw agent ids out of the conversation target.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import DEFAULT_DATAROOT, mkdir, output_root


CURRENT_TASK_NAME = "current risk, future risk, target-agent role, and scene-consistent quantitative ego meta-action mitigation"
REQUIRED_ANSWER_FIELDS = {
    "agent_id",
    "observed_timestep_count",
    "final_timestep",
    "per_timestep_distances",
    "current_risk",
    "predicted_future_worst_risk",
    "risk_change_analysis",
    "target_agent_role",
    "scene_consistent_ego_mitigation",
    "reasoning_summary",
}
FORBIDDEN_ANSWER_KEYS = {
    "agent_identity",
    "raw_agent_id",
    "risk_agent_id",
    "instance_token",
    "sample_annotation_token",
    "observed_clip",
    "scene_risk_context",
    "task",
    "explanations",
    "mitigation_explanation",
}
FORBIDDEN_ENTRY_KEYS = {
    "agent_identity",
    "raw_agent_id",
    "risk_agent_id",
    "instance_token",
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


def split_entries(
    entries: Sequence[Dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(entries)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_point = int(len(shuffled) * train_ratio)
    return shuffled[:split_point], shuffled[split_point:]


def group_by_key(entries: Sequence[Dict[str, Any]], key: str) -> List[List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(str(entry.get(key, "")), []).append(entry)
    return list(groups.values())


def split_groups(
    groups: Sequence[List[Dict[str, Any]]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(groups)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_point = int(len(shuffled) * train_ratio)
    train_groups = shuffled[:split_point]
    validation_groups = shuffled[split_point:]
    return flatten(train_groups), flatten(validation_groups)


def flatten(groups: Iterable[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [entry for group in groups for entry in group]


def split_dataset(
    entries: Sequence[Dict[str, Any]],
    train_ratio: float,
    seed: int,
    split_unit: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if split_unit == "entry":
        return split_entries(entries, train_ratio, seed)
    if split_unit == "clip":
        return split_groups(group_by_key(entries, "clip_id"), train_ratio, seed)
    if split_unit == "scene":
        return split_groups(group_by_key(entries, "scene"), train_ratio, seed)
    raise ValueError(f"Unsupported split unit: {split_unit}")


def conversation_answer_text(entry: Dict[str, Any]) -> str:
    conversations = entry.get("conversations", [])
    if len(conversations) < 2:
        return ""
    return str(conversations[1].get("value", ""))


def parse_answer(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    answer_text = conversation_answer_text(entry)
    if not answer_text:
        return None
    try:
        parsed = json.loads(answer_text)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def mitigation_status(answer: Optional[Dict[str, Any]]) -> str:
    if not answer:
        return "missing_answer"
    mitigation = answer.get("scene_consistent_ego_mitigation", {})
    return str(mitigation.get("mitigation_status", "missing_status"))


def target_agent_role(answer: Optional[Dict[str, Any]]) -> str:
    if not answer:
        return "missing_answer"
    return str(answer.get("target_agent_role", "missing_role"))


def future_risk_score(answer: Optional[Dict[str, Any]]) -> str:
    if not answer:
        return "missing_answer"
    future = answer.get("predicted_future_worst_risk", {})
    if not future.get("available", False):
        return "unavailable"
    return str(future.get("risk_score", "missing_score"))


def current_risk_score(answer: Optional[Dict[str, Any]]) -> str:
    if not answer:
        return "missing_answer"
    current = answer.get("current_risk", {})
    return str(current.get("risk_score", "missing_score"))


def entry_answer_cache(entries: Sequence[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    return {str(entry.get("id", index)): parse_answer(entry) for index, entry in enumerate(entries)}


def answer_for_entry(
    entry: Dict[str, Any],
    cache: Dict[str, Optional[Dict[str, Any]]],
    fallback_index: int,
) -> Optional[Dict[str, Any]]:
    return cache.get(str(entry.get("id", fallback_index)))


def count_values(entries: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(entry.get(key, "")) for entry in entries).items()))


def count_answer_values(
    entries: Sequence[Dict[str, Any]],
    answers: Dict[str, Optional[Dict[str, Any]]],
    value_fn,
) -> Dict[str, int]:
    values = []
    for index, entry in enumerate(entries):
        values.append(value_fn(answer_for_entry(entry, answers, index)))
    return dict(sorted(Counter(values).items()))


def validate_new_answer_schema(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    invalid_json = []
    missing_required_fields = []
    missing_video_inputs = []
    missing_scene_mitigation = []
    legacy_mitigation_fields = []
    forbidden_answer_fields = []
    forbidden_entry_fields = []
    incomplete_distance_sequences = []

    for index, entry in enumerate(entries):
        entry_id = str(entry.get("id", index))
        forbidden_entry = sorted(FORBIDDEN_ENTRY_KEYS & set(entry.keys()))
        if forbidden_entry:
            forbidden_entry_fields.append({"id": entry_id, "fields": forbidden_entry})

        video = entry.get("video")
        if not isinstance(video, list) or not video:
            missing_video_inputs.append(entry_id)

        answer = parse_answer(entry)
        if answer is None:
            invalid_json.append(entry_id)
            continue

        missing = sorted(REQUIRED_ANSWER_FIELDS - set(answer.keys()))
        if missing:
            missing_required_fields.append({"id": entry_id, "missing": missing})

        if not answer.get("scene_consistent_ego_mitigation"):
            missing_scene_mitigation.append(entry_id)

        forbidden_answer = sorted(find_forbidden_keys(answer, FORBIDDEN_ANSWER_KEYS))
        if forbidden_answer:
            forbidden_answer_fields.append({"id": entry_id, "fields": forbidden_answer})

        distances = answer.get("per_timestep_distances", [])
        expected_count = answer.get("observed_timestep_count")
        if not isinstance(distances, list) or len(distances) != expected_count:
            incomplete_distance_sequences.append(
                {"id": entry_id, "observed_timestep_count": expected_count, "distance_count": len(distances) if isinstance(distances, list) else None}
            )

        if "quantitative_mitigation_suggestion" in answer:
            legacy_mitigation_fields.append(entry_id)

    return {
        "invalid_answer_json_count": len(invalid_json),
        "missing_required_fields_count": len(missing_required_fields),
        "missing_video_inputs_count": len(missing_video_inputs),
        "missing_scene_consistent_mitigation_count": len(missing_scene_mitigation),
        "legacy_quantitative_mitigation_field_count": len(legacy_mitigation_fields),
        "forbidden_answer_field_count": len(forbidden_answer_fields),
        "forbidden_entry_field_count": len(forbidden_entry_fields),
        "incomplete_distance_sequence_count": len(incomplete_distance_sequences),
        "invalid_answer_json_examples": invalid_json[:10],
        "missing_required_fields_examples": missing_required_fields[:10],
        "missing_video_inputs_examples": missing_video_inputs[:10],
        "missing_scene_consistent_mitigation_examples": missing_scene_mitigation[:10],
        "legacy_quantitative_mitigation_field_examples": legacy_mitigation_fields[:10],
        "forbidden_answer_field_examples": forbidden_answer_fields[:10],
        "forbidden_entry_field_examples": forbidden_entry_fields[:10],
        "incomplete_distance_sequence_examples": incomplete_distance_sequences[:10],
    }


def find_forbidden_keys(value: Any, forbidden_keys: set) -> set:
    found = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in forbidden_keys:
                found.add(key)
            found.update(find_forbidden_keys(nested, forbidden_keys))
    elif isinstance(value, list):
        for nested in value:
            found.update(find_forbidden_keys(nested, forbidden_keys))
    return found


def clip_mitigation_consistency(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    statuses_by_clip: Dict[str, set] = defaultdict(set)
    mitigation_payloads_by_clip: Dict[str, set] = defaultdict(set)

    for entry in entries:
        clip_id = str(entry.get("clip_id", ""))
        answer = parse_answer(entry)
        status = mitigation_status(answer)
        mitigation = (answer or {}).get("scene_consistent_ego_mitigation", {})
        statuses_by_clip[clip_id].add(status)
        mitigation_payloads_by_clip[clip_id].add(json.dumps(mitigation, sort_keys=True, ensure_ascii=False))

    status_violations = {
        clip_id: sorted(values)
        for clip_id, values in statuses_by_clip.items()
        if len(values) > 1
    }
    payload_violations = {
        clip_id: len(values)
        for clip_id, values in mitigation_payloads_by_clip.items()
        if len(values) > 1
    }
    return {
        "clips_checked": len(statuses_by_clip),
        "status_consistency_violation_count": len(status_violations),
        "payload_consistency_violation_count": len(payload_violations),
        "status_consistency_violation_examples": dict(list(status_violations.items())[:10]),
        "payload_consistency_violation_examples": dict(list(payload_violations.items())[:10]),
    }


def make_stats(
    train_entries: Sequence[Dict[str, Any]],
    validation_entries: Sequence[Dict[str, Any]],
    train_ratio: float,
    seed: int,
    split_unit: str,
    source_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    all_entries = list(train_entries) + list(validation_entries)
    train_clips = {entry.get("clip_id") for entry in train_entries}
    validation_clips = {entry.get("clip_id") for entry in validation_entries}
    train_scenes = {entry.get("scene") for entry in train_entries}
    validation_scenes = {entry.get("scene") for entry in validation_entries}
    answers = entry_answer_cache(all_entries)
    train_answers = entry_answer_cache(train_entries)
    validation_answers = entry_answer_cache(validation_entries)

    return {
        "task": CURRENT_TASK_NAME,
        "source_task": source_metadata.get("task"),
        "total_samples": len(all_entries),
        "train_samples": len(train_entries),
        "validation_samples": len(validation_entries),
        "train_ratio": train_ratio,
        "seed": seed,
        "split_unit": split_unit,
        "split_policy_note": "clip split keeps all target-agent VQA samples from the same clip in one split so they share the same scene-consistent mitigation label without train/validation leakage.",
        "total_scenes": len({entry.get("scene") for entry in all_entries}),
        "train_scenes": len(train_scenes),
        "validation_scenes": len(validation_scenes),
        "total_clips": len({entry.get("clip_id") for entry in all_entries}),
        "train_clips": len(train_clips),
        "validation_clips": len(validation_clips),
        "overlapping_clips": len(train_clips & validation_clips),
        "overlapping_scenes": len(train_scenes & validation_scenes),
        "samples_by_scene_train": count_values(train_entries, "scene"),
        "samples_by_scene_validation": count_values(validation_entries, "scene"),
        "samples_by_target_agent_role_train": count_answer_values(train_entries, train_answers, target_agent_role),
        "samples_by_target_agent_role_validation": count_answer_values(validation_entries, validation_answers, target_agent_role),
        "samples_by_mitigation_status_train": count_answer_values(train_entries, train_answers, mitigation_status),
        "samples_by_mitigation_status_validation": count_answer_values(validation_entries, validation_answers, mitigation_status),
        "samples_by_current_risk_score_train": count_answer_values(train_entries, train_answers, current_risk_score),
        "samples_by_current_risk_score_validation": count_answer_values(validation_entries, validation_answers, current_risk_score),
        "samples_by_future_risk_score_train": count_answer_values(train_entries, train_answers, future_risk_score),
        "samples_by_future_risk_score_validation": count_answer_values(validation_entries, validation_answers, future_risk_score),
        "schema_validation": validate_new_answer_schema(all_entries),
        "clip_mitigation_consistency": clip_mitigation_consistency(all_entries),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--input-dir", default=None, help="Default: <dataroot>/nurisk_style")
    parser.add_argument("--input-file", default=None, help="Default: <input-dir>/qwen_future_vqa_dataset.json")
    parser.add_argument("--output-dir", default=None, help="Default: <input-dir>/dataset_splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-unit",
        choices=["entry", "clip", "scene"],
        default="clip",
        help="Default clip keeps all target-agent samples from the same clip in one split. Use entry for legacy NuRisk-style sample-level shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    root = output_root(args.dataroot, args.input_dir)
    input_path = Path(args.input_file).expanduser().resolve() if args.input_file else root / "qwen_future_vqa_dataset.json"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "dataset_splits"

    dataset = load_json(input_path)
    entries = dataset.get("entries", [])
    if not entries:
        raise ValueError(f"No entries found in {input_path}")

    train_entries, validation_entries = split_dataset(
        entries,
        args.train_ratio,
        args.seed,
        args.split_unit,
    )
    stats = make_stats(
        train_entries,
        validation_entries,
        args.train_ratio,
        args.seed,
        args.split_unit,
        dataset.get("metadata", {}),
    )

    write_json(output_dir / "train.json", train_entries)
    write_json(output_dir / "validation.json", validation_entries)
    write_json(output_dir / "dataset_stats.json", stats)

    print("Stage 8 done.")
    print(f"Input: {input_path}")
    print(f"Split unit: {args.split_unit}")
    print(f"Train samples: {len(train_entries)}")
    print(f"Validation samples: {len(validation_entries)}")
    print(f"Output dir: {output_dir}")
    if stats["overlapping_clips"]:
        print(f"Warning: {stats['overlapping_clips']} clip_id values appear in both splits.")
    schema = stats["schema_validation"]
    if schema["invalid_answer_json_count"] or schema["missing_required_fields_count"]:
        print("Warning: some entries do not match the current Stage 7 answer schema; see dataset_stats.json.")
    if schema["forbidden_answer_field_count"] or schema["forbidden_entry_field_count"]:
        print("Warning: raw/debug fields were found in training entries; see dataset_stats.json.")
    consistency = stats["clip_mitigation_consistency"]
    if consistency["payload_consistency_violation_count"]:
        print("Warning: some clips have inconsistent scene-consistent mitigation payloads; see dataset_stats.json.")


if __name__ == "__main__":
    main()
