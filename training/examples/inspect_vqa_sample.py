#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Print one clean VQA training sample."""

import argparse
import json

from risk_mllm.data.vqa_dataset import NuRiskVQADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", default="data/sets/nuscenes_mini/nurisk_style/dataset_splits/train.json")
    parser.add_argument("--vqa-root", default="data/sets/nuscenes_mini/nurisk_style")
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = NuRiskVQADataset(args.json_path, args.vqa_root)
    sample = dataset[args.index]
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
