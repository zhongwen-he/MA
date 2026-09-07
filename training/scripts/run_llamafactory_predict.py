#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run LLaMA-Factory prediction with a Qwen3-VL video_metadata compatibility patch."""

import argparse
from pathlib import Path

import yaml
from transformers.generation.utils import GenerationMixin

from llamafactory.train.tuner import run_exp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a LLaMA-Factory YAML prediction config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(Path(args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    original_validate = GenerationMixin._validate_model_kwargs

    def validate_without_video_metadata(self, model_kwargs):
        model_kwargs.pop("video_metadata", None)
        return original_validate(self, model_kwargs)

    GenerationMixin._validate_model_kwargs = validate_without_video_metadata
    run_exp(args=config)


if __name__ == "__main__":
    main()
