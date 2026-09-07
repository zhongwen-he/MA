#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export NuRisk VQA splits to LLaMA-Factory multimodal ShareGPT format.

This prepares the mature LLaMA-Factory SFT pipeline for the heavy Qwen3-VL LoRA
stage. The exported answer is the short JSON produced by ``NuRiskVQADataset``.
By default the LLaMA-Factory files contain only the columns consumed by SFT
(``messages`` and ``videos``), so auxiliary risk metadata cannot break Arrow
schema inference.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from risk_mllm.data import NuRiskVQADataset


DEFAULT_CAMERA_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--validation-json", required=True)
    parser.add_argument("--vqa-root", required=True)
    parser.add_argument("--output-dir", default="training/llamafactory/data")
    parser.add_argument("--train-name", default="nurisk_train")
    parser.add_argument("--validation-name", default="nurisk_validation")
    parser.add_argument("--relative-video-paths", action="store_true")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Export at most this many training samples. Useful for small LLaMA-Factory runs.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Export at most this many validation samples. Useful for small LLaMA-Factory runs.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Also export id/scene/risk_targets/cot_steps for debugging. Do not use for LLaMA-Factory SFT.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when max-train-samples or max-validation-samples caps a split.",
    )
    parser.add_argument(
        "--sequential-sampling",
        action="store_true",
        help="Use the original first-N behavior for capped exports. Default is reproducible random sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = NuRiskVQADataset(args.train_json, args.vqa_root)
    validation_dataset = NuRiskVQADataset(args.validation_json, args.vqa_root)

    train_file = output_dir / f"{args.train_name}.json"
    validation_file = output_dir / f"{args.validation_name}.json"
    write_json(
        train_file,
        list(
            convert_dataset(
                train_dataset,
                output_dir,
                args.relative_video_paths,
                args.include_metadata,
                args.max_train_samples,
                args.seed,
                args.sequential_sampling,
            )
        ),
    )
    write_json(
        validation_file,
        list(
            convert_dataset(
                validation_dataset,
                output_dir,
                args.relative_video_paths,
                args.include_metadata,
                args.max_validation_samples,
                args.seed + 1,
                args.sequential_sampling,
            )
        ),
    )

    data_info = {
        args.train_name: build_dataset_info(train_file.name),
        args.validation_name: build_dataset_info(validation_file.name),
    }
    write_json(output_dir / "dataset_info.json", data_info)
    print(
        json.dumps(
            {
                "train_file": str(train_file),
                "validation_file": str(validation_file),
                "dataset_info": str(output_dir / "dataset_info.json"),
                "source_train_samples": len(train_dataset),
                "source_validation_samples": len(validation_dataset),
                "exported_train_samples": count_json_rows(train_file),
                "exported_validation_samples": count_json_rows(validation_file),
                "sampling": "sequential" if args.sequential_sampling else "random",
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def convert_dataset(
    dataset: NuRiskVQADataset,
    output_dir: Path,
    relative_video_paths: bool,
    include_metadata: bool,
    max_samples: int | None,
    seed: int,
    sequential_sampling: bool,
) -> Iterable[Dict[str, Any]]:
    indices = build_sample_indices(len(dataset), max_samples, seed, sequential_sampling)
    for index in indices:
        sample = dataset[index]
        video_paths = [format_video_path(path, output_dir, relative_video_paths) for path in sample["video_paths"]]
        instruction = build_multivideo_instruction(sample["question"], sample["video_paths"])
        row = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": sample["answer"]},
            ],
            "videos": video_paths,
        }
        if include_metadata:
            row.update(
                {
                    "id": sample.get("id"),
                    "scene": sample.get("scene"),
                    "clip_id": sample.get("clip_id"),
                    "agent_id": sample.get("agent_id"),
                    "target_vehicle": sample.get("target_vehicle"),
                    "canonical_agent_name": sample.get("canonical_agent_name"),
                    "clip_reference_name": sample.get("clip_reference_name"),
                    "risk_targets": sample.get("risk_targets", {}),
                    "cot_steps": sample.get("cot_steps", []),
                    "explanations": sample.get("explanations", {}),
                    "reasoning_summary": sample.get("reasoning_summary", ""),
                }
            )
        yield row


def build_sample_indices(
    num_samples: int,
    max_samples: int | None,
    seed: int,
    sequential_sampling: bool,
) -> List[int]:
    limit = num_samples if max_samples is None else min(max_samples, num_samples)
    if max_samples is None or sequential_sampling or limit >= num_samples:
        return list(range(limit))
    return random.Random(seed).sample(range(num_samples), limit)


def build_multivideo_instruction(question: str, video_paths: List[str]) -> str:
    body = str(question).replace("<video>", "").strip()
    camera_names = [infer_camera_name(path, index) for index, path in enumerate(video_paths)]
    video_lines = [f"<video>\nView {index + 1}: {camera_name}." for index, camera_name in enumerate(camera_names)]
    return "\n".join(video_lines + [body])


def infer_camera_name(path: str, index: int) -> str:
    upper_path = str(path).upper()
    for camera_name in DEFAULT_CAMERA_ORDER:
        if camera_name in upper_path:
            return camera_name
    return f"CAMERA_{index + 1}"


def format_video_path(path: str, output_dir: Path, relative_video_paths: bool) -> str:
    video_path = Path(path).expanduser().resolve()
    if not relative_video_paths:
        return str(video_path)
    try:
        return str(video_path.relative_to(output_dir))
    except ValueError:
        return str(video_path)


def build_dataset_info(file_name: str) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "videos": "videos",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_json_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return len(json.load(f))


if __name__ == "__main__":
    main()
