#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Download Stage-2 pretrained assets and extract BLIP-2 Q-Former weights.

This script is designed for the 8GB-GPU / low-RAM workstation case:

1. Download Qwen3-VL-2B-Instruct into ``models/Qwen3-VL-2B-Instruct`` if needed.
2. Download only the BLIP-2 checkpoint shard(s) that contain ``qformer.*`` and
   ``query_tokens``.
3. Save a small extracted checkpoint for Stage 2:

   ``training/pretrained/blip2_qformer/Salesforce_blip2-opt-2.7b/qformer_query.pt``

The training code can load that file through ``MultiViewQFormerConfig`` without
calling ``Blip2Model.from_pretrained`` and without instantiating OPT-2.7B.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_QWEN_REPO = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_BLIP2_REPO = "Salesforce/blip2-opt-2.7b"
DEFAULT_QWEN_DIR = "models/Qwen3-VL-2B-Instruct"
DEFAULT_QFORMER_OUTPUT = "training/pretrained/blip2_qformer/Salesforce_blip2-opt-2.7b/qformer_query.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-repo", default=DEFAULT_QWEN_REPO)
    parser.add_argument("--qwen-dir", default=DEFAULT_QWEN_DIR)
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--blip2-repo", default=DEFAULT_BLIP2_REPO)
    parser.add_argument(
        "--blip2-local-dir",
        default=None,
        help=(
            "Optional local directory containing config.json, "
            "model.safetensors.index.json, and required safetensors shard(s). "
            "When set, BLIP-2 Q-Former extraction is fully offline."
        ),
    )
    parser.add_argument("--qformer-output", default=DEFAULT_QFORMER_OUTPUT)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing extracted Q-Former file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_qwen:
        download_qwen(args.qwen_repo, Path(args.qwen_dir), args.cache_dir)
    extract_blip2_qformer(
        repo_id=args.blip2_repo,
        output_path=Path(args.qformer_output),
        cache_dir=args.cache_dir,
        force=bool(args.force),
        local_dir=Path(args.blip2_local_dir) if args.blip2_local_dir else None,
    )


def download_qwen(repo_id: str, output_dir: Path, cache_dir: str | None) -> None:
    output_dir = output_dir.expanduser()
    if (output_dir / "config.json").is_file():
        print(json.dumps({"event": "qwen_exists", "path": str(output_dir)}, ensure_ascii=False))
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        cache_dir=cache_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(json.dumps({"event": "qwen_downloaded", "repo": repo_id, "path": downloaded_path}, ensure_ascii=False))


def extract_blip2_qformer(
    repo_id: str,
    output_path: Path,
    cache_dir: str | None,
    force: bool,
    local_dir: Path | None = None,
) -> None:
    output_path = output_path.expanduser()
    if output_path.is_file() and not force:
        print(json.dumps({"event": "qformer_exists", "path": str(output_path)}, ensure_ascii=False))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if local_dir is not None:
        local_dir = local_dir.expanduser()
        config_path = str(local_dir / "config.json")
        index_path = str(local_dir / "model.safetensors.index.json")
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"Missing local BLIP-2 config.json: {config_path}")
        if not Path(index_path).is_file():
            raise FileNotFoundError(f"Missing local BLIP-2 model.safetensors.index.json: {index_path}")
    else:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=cache_dir, resume_download=True)
        index_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            cache_dir=cache_dir,
            resume_download=True,
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    selected_keys = [
        key
        for key in weight_map
        if key == "query_tokens" or key.startswith("qformer.")
    ]
    if not selected_keys:
        raise RuntimeError(f"No qformer/query_tokens keys found in {repo_id} index: {index_path}")

    shard_names = sorted({weight_map[key] for key in selected_keys})
    print(json.dumps({
        "event": "qformer_shards_required",
        "repo": repo_id,
        "num_tensors": len(selected_keys),
        "shards": shard_names,
    }, ensure_ascii=False))

    if local_dir is not None:
        shard_paths = {}
        for shard_name in shard_names:
            shard_path = local_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing local BLIP-2 shard {shard_name}: {shard_path}")
            shard_paths[shard_name] = str(shard_path)
    else:
        shard_paths = {
            shard_name: hf_hub_download(
                repo_id=repo_id,
                filename=shard_name,
                cache_dir=cache_dir,
                resume_download=True,
            )
            for shard_name in shard_names
        }

    from safetensors import safe_open
    import torch

    qformer_state_dict: dict[str, torch.Tensor] = {}
    query_tokens = None
    for key in selected_keys:
        shard_path = shard_paths[weight_map[key]]
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key).contiguous()
        if key == "query_tokens":
            query_tokens = tensor
        else:
            qformer_state_dict[key.removeprefix("qformer.")] = tensor

    if query_tokens is None:
        raise RuntimeError("BLIP-2 checkpoint did not contain query_tokens")

    with open(config_path, "r", encoding="utf-8") as f:
        blip2_config = json.load(f)
    qformer_config = _load_full_qformer_config_dict(config_path)
    payload: dict[str, Any] = {
        "source_repo": repo_id,
        "source_config_path": str(config_path),
        "source_index_path": str(index_path),
        "source_shards": shard_names,
        "num_query_tokens": int(blip2_config.get("num_query_tokens", query_tokens.shape[1])),
        "qformer_config": qformer_config,
        "query_tokens": query_tokens.cpu(),
        "qformer_state_dict": {key: value.cpu() for key, value in qformer_state_dict.items()},
    }

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    shutil.move(str(tmp_path), str(output_path))
    print(json.dumps({
        "event": "qformer_extracted",
        "path": str(output_path),
        "query_tokens_shape": list(query_tokens.shape),
        "qformer_tensors": len(qformer_state_dict),
        "qformer_hidden_size": qformer_config.get("hidden_size"),
        "qformer_encoder_hidden_size": qformer_config.get("encoder_hidden_size"),
        "qformer_num_hidden_layers": qformer_config.get("num_hidden_layers"),
        "size_mb": round(output_path.stat().st_size / 1024 / 1024, 1),
    }, ensure_ascii=False))


def _load_full_qformer_config_dict(config_path: str) -> dict[str, Any]:
    """Load BLIP-2 Q-Former config with Transformers defaults expanded.

    Salesforce/blip2-opt-2.7b stores only a partial ``qformer_config`` in
    config.json.  Using ``Blip2Config`` here expands it to the actual default
    Q-Former architecture fields used by Hugging Face.
    """

    try:
        from transformers import Blip2Config
    except ImportError as exc:
        raise RuntimeError("transformers is required to expand BLIP-2 Q-Former config") from exc

    blip2_config = Blip2Config.from_json_file(str(config_path))
    qformer_config = blip2_config.qformer_config.to_dict()
    required_keys = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "cross_attention_frequency",
        "encoder_hidden_size",
    ]
    missing = [key for key in required_keys if key not in qformer_config]
    if missing:
        raise RuntimeError(f"Expanded BLIP-2 Q-Former config missing keys {missing}: {config_path}")
    return qformer_config


if __name__ == "__main__":
    main()
