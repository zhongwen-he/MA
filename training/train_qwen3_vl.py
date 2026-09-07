#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal Qwen3-VL multi-camera risk-latent training entrypoint."""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from risk_mllm.data import NuRiskVQADataset, Qwen3VLMultiCameraCollator
from risk_mllm.models import (
    MultiViewQFormerConfig,
    Qwen3VLMultiCameraRiskMLLM,
    RiskLatentConfig,
    build_qwen3_vl_multicamera_model,
    freeze_qwen_video_encoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/qwen3_vl_training_test.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--forward-only", action="store_true", help="Run one no-grad batch and print smoke-test outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train_cfg = config.get("training", {})
    output_dir = Path(args.output_dir or train_cfg.get("output_dir", "training/outputs/qwen3_vl_training_test"))
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, qwen_model = load_qwen3_vl(config.get("qwen3_vl", {}))
    if train_cfg.get("freeze_video_encoder", True):
        freeze_qwen_video_encoder(qwen_model)

    qwen_model = configure_llm_training(qwen_model, train_cfg)
    model = build_model(qwen_model, config)
    risk_adapter_path = train_cfg.get("risk_adapter_name_or_path")
    if risk_adapter_path:
        load_risk_adapter_checkpoint(model, risk_adapter_path)
    configure_adapter_training(model, train_cfg)
    adapter_device = infer_adapter_device(qwen_model)
    model.qformer.to(adapter_device)
    model.aux_heads.to(adapter_device)
    if model.cot_decoder is not None:
        model.cot_decoder.to(adapter_device)
    model.risk_latent_tokens.data = model.risk_latent_tokens.data.to(adapter_device)

    train_dataset = build_dataset(config["data"], split="train")
    validation_dataset = build_dataset(config["data"], split="validation")
    train_dataset = maybe_subsample_dataset(
        train_dataset,
        train_cfg.get("max_train_samples"),
        int(train_cfg.get("random_seed", 42)),
        name="train",
    )
    validation_dataset = maybe_subsample_dataset(
        validation_dataset,
        train_cfg.get("max_validation_samples"),
        int(train_cfg.get("random_seed", 42)) + 1,
        name="validation",
    )
    print(json.dumps({"train_samples": len(train_dataset), "validation_samples": len(validation_dataset)}))
    collator = Qwen3VLMultiCameraCollator(
        processor=processor,
        risk_config=model.risk_config,
        max_question_length=int(train_cfg.get("max_question_length", 1024)),
        max_answer_length=int(train_cfg.get("max_answer_length", 1024)),
        max_cot_step_length=int(train_cfg.get("max_cot_step_length", model.risk_config.max_cot_step_length)),
        video_processor_kwargs=config.get("video_processor", {"fps": 1.0}),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 1))),
        shuffle=bool(train_cfg.get("shuffle_validation", False)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        collate_fn=collator,
    )

    if args.forward_only:
        run_forward_only_smoke(model, train_loader, adapter_device, train_cfg)
        return

    optimizer = build_optimizer(model, train_cfg)
    print(json.dumps({"trainable_parameters": count_trainable_parameters(model)}))

    max_steps = args.max_steps or train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    log_every = int(train_cfg.get("log_every", 1))
    save_every = int(train_cfg.get("save_every", 50))

    max_bad_batches = int(train_cfg.get("max_bad_batches", 100))
    set_train_mode(model, train_cfg)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(train_cfg.get("num_epochs", 1))):
        bad_batches = 0
        pending_micro_steps = 0
        for step, batch in enumerate_valid_batches(train_loader, max_bad_batches=max_bad_batches, phase="train"):
            batch = move_batch_to_device(batch, adapter_device)
            batch = prepare_batch_for_stage(batch, train_cfg)
            outputs = model(**batch)
            loss = outputs["loss"] / grad_accum
            loss.backward()
            pending_micro_steps += 1
            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_micro_steps = 0
                global_step += 1
                if global_step % log_every == 0:
                    log_train_step(epoch, global_step, outputs)
                if global_step % save_every == 0:
                    save_checkpoint(output_dir / f"step-{global_step}", model, processor, config, train_cfg)
                    prune_step_checkpoints(output_dir, int(train_cfg.get("save_total_limit", 0)))
                if max_steps is not None and global_step >= max_steps:
                    break
        if pending_micro_steps > 0 and (max_steps is None or global_step < max_steps):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            log_train_step(epoch, global_step, outputs)
        evaluate(
            model,
            validation_loader,
            adapter_device,
            max_batches=int(train_cfg.get("eval_max_batches", 2)),
            max_bad_batches=max_bad_batches,
            train_cfg=train_cfg,
        )
        if max_steps is not None and global_step >= max_steps:
            break

    save_checkpoint(output_dir / "final", model, processor, config, train_cfg)


def build_dataset(data_cfg: Dict[str, Any], split: str) -> NuRiskVQADataset | ConcatDataset:
    datasets_key = f"{split}_datasets"
    if datasets_key in data_cfg:
        datasets = [
            NuRiskVQADataset(
                dataset_cfg["json"],
                dataset_cfg.get("vqa_root", data_cfg.get("vqa_root")),
                answer_label_policy=dataset_cfg.get(
                    "answer_label_policy",
                    data_cfg.get("answer_label_policy", "full"),
                ),
            )
            for dataset_cfg in data_cfg[datasets_key]
        ]
        if not datasets:
            raise ValueError(f"data.{datasets_key} must not be empty")
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    json_key = "train_json" if split == "train" else "validation_json"
    if json_key not in data_cfg:
        raise KeyError(f"Missing data.{json_key} or data.{datasets_key}")
    return NuRiskVQADataset(
        data_cfg[json_key],
        data_cfg.get("vqa_root"),
        answer_label_policy=data_cfg.get("answer_label_policy", "full"),
    )


def maybe_subsample_dataset(
    dataset: Dataset,
    max_samples: Any,
    seed: int,
    name: str,
) -> Dataset:
    if max_samples is None:
        return dataset
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError(f"{name} max_samples must be positive, got {max_samples}")
    if max_samples >= len(dataset):
        return dataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def enumerate_valid_batches(loader: DataLoader, max_bad_batches: int, phase: str):
    iterator = iter(loader)
    good_batches = 0
    bad_batches = 0
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            bad_batches += 1
            print(json.dumps({
                "phase": phase,
                "event": "skip_bad_batch",
                "bad_batches": bad_batches,
                "error": str(exc),
            }, ensure_ascii=False))
            if bad_batches > max_bad_batches:
                raise RuntimeError(f"Exceeded max_bad_batches={max_bad_batches} during {phase}") from exc
            continue
        good_batches += 1
        yield good_batches, batch


def build_model(qwen_model: torch.nn.Module, config: Dict[str, Any]) -> Qwen3VLMultiCameraRiskMLLM:
    qformer_config = MultiViewQFormerConfig(**config.get("model", {}))
    risk_config = RiskLatentConfig(**normalize_risk_config(config.get("risk_latent", {})))
    return build_qwen3_vl_multicamera_model(
        qwen_model=qwen_model,
        qformer_config=qformer_config,
        risk_config=risk_config,
    )


def load_qwen3_vl(config: Dict[str, Any]) -> tuple[Any, torch.nn.Module]:
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen3-VL requires a recent transformers build. Install the latest transformers from source "
            "or a release that includes Qwen3VLForConditionalGeneration."
        ) from exc

    model_name = resolve_model_path(
        config.get("model_name_or_path", "Qwen/Qwen3-VL-2B-Instruct"),
        bool(config.get("local_files_only", False)),
    )
    local_files_only = bool(config.get("local_files_only", False))
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)

    model_kwargs = {
        "local_files_only": local_files_only,
    }
    if config.get("device_map", "auto") is not None:
        model_kwargs["device_map"] = config.get("device_map", "auto")
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]
    dtype = config.get("dtype", "auto")
    try:
        qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=dtype, **model_kwargs)
    except TypeError:
        torch_dtype = dtype_to_torch(dtype)
        qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )
    adapter_path = config.get("adapter_name_or_path")
    if adapter_path:
        qwen_model = load_peft_adapter(
            qwen_model,
            adapter_path=adapter_path,
            is_trainable=bool(config.get("adapter_is_trainable", False)),
        )
    return processor, qwen_model


def resolve_model_path(model_name_or_path: str, local_files_only: bool) -> str:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return str(path.resolve())
    if not local_files_only:
        return model_name_or_path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("local_files_only repo-id resolution requires huggingface_hub") from exc
    try:
        return snapshot_download(model_name_or_path, local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            f"Model {model_name_or_path!r} is not available in the local Hugging Face cache. "
            "Download it first or set qwen3_vl.model_name_or_path to a local checkpoint directory."
        ) from exc


def configure_llm_training(qwen_model: torch.nn.Module, train_cfg: Dict[str, Any]) -> torch.nn.Module:
    mode = train_cfg.get("train_llm", "lora_only")
    if hasattr(qwen_model, "peft_config"):
        if mode == "full":
            return qwen_model
        if mode == "frozen":
            for parameter in qwen_model.parameters():
                parameter.requires_grad = False
            return qwen_model
        if mode == "lora_only":
            for name, parameter in qwen_model.named_parameters():
                parameter.requires_grad = _is_lora_parameter(name) or "modules_to_save" in name
            return qwen_model
        raise ValueError(f"Unsupported train_llm mode: {mode}")

    if mode == "full":
        return qwen_model
    for parameter in qwen_model.parameters():
        parameter.requires_grad = False
    if mode == "frozen":
        return qwen_model
    if mode != "lora_only":
        raise ValueError(f"Unsupported train_llm mode: {mode}")

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("train_llm=lora_only requires the peft package") from exc

    lora_cfg = train_cfg.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    return get_peft_model(qwen_model, peft_config)


def load_peft_adapter(qwen_model: torch.nn.Module, adapter_path: str, is_trainable: bool = False) -> torch.nn.Module:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading a LLaMA-Factory LoRA adapter requires the peft package") from exc

    adapter_path = str(Path(adapter_path).expanduser().resolve())
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"LoRA adapter path does not exist: {adapter_path}")
    return PeftModel.from_pretrained(qwen_model, adapter_path, is_trainable=is_trainable)


def configure_adapter_training(model: Qwen3VLMultiCameraRiskMLLM, train_cfg: Dict[str, Any]) -> None:
    if not train_cfg.get("train_qformer", True):
        for parameter in model.qformer.parameters():
            parameter.requires_grad = False
    else:
        last_n_layers = train_cfg.get("qformer_train_last_n_layers")
        if last_n_layers is not None:
            configure_qformer_last_n_layers(model, int(last_n_layers))
    if train_cfg.get("train_projector", True):
        for parameter in model.qformer.projector.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.qformer.projector.parameters():
            parameter.requires_grad = False

    model.risk_latent_tokens.requires_grad = bool(train_cfg.get("train_risk_latents", True))
    for parameter in model.aux_heads.parameters():
        parameter.requires_grad = bool(train_cfg.get("train_risk_aux_heads", True))
    if model.cot_decoder is not None:
        for parameter in model.cot_decoder.parameters():
            parameter.requires_grad = bool(train_cfg.get("train_cot_decoder", True))


def configure_qformer_last_n_layers(model: Qwen3VLMultiCameraRiskMLLM, last_n_layers: int) -> None:
    """Train only the last N BLIP-2 Q-Former encoder layers plus lightweight adapters."""

    if last_n_layers < 0:
        raise ValueError(f"qformer_train_last_n_layers must be >= 0, got {last_n_layers}")
    qformer_model = getattr(model.qformer, "qformer", None)
    encoder = getattr(qformer_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError("Cannot locate BLIP-2 Q-Former encoder layers at model.qformer.qformer.encoder.layer")

    for parameter in qformer_model.parameters():
        parameter.requires_grad = False

    if last_n_layers > 0:
        total_layers = len(layers)
        start = max(0, total_layers - last_n_layers)
        for layer in layers[start:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    if hasattr(model.qformer, "query_tokens"):
        model.qformer.query_tokens.requires_grad = not bool(train_cfg_bool(model, "freeze_query_tokens", False))


def train_cfg_bool(model: Qwen3VLMultiCameraRiskMLLM, name: str, default: bool) -> bool:
    """Read a freeze flag stored on the Q-Former config without coupling callers to it."""

    config = getattr(model.qformer, "config", None)
    return bool(getattr(config, name, default))


def set_train_mode(model: Qwen3VLMultiCameraRiskMLLM, train_cfg: Dict[str, Any]) -> None:
    """Put trainable adapters in train mode while keeping a frozen Qwen stable."""

    model.train()
    if train_cfg.get("train_llm", "lora_only") == "frozen":
        model.qwen_model.eval()


def prepare_batch_for_stage(batch: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unused supervision fields according to the active training stage."""

    batch = dict(batch)
    if not bool(train_cfg.get("compute_json_loss", True)):
        for key in (
            "answer_input_ids",
            "answer_attention_mask",
            "answer_labels",
        ):
            batch.pop(key, None)
    if not bool(train_cfg.get("compute_risk_loss", True)):
        batch.pop("risk_targets", None)
    if not bool(train_cfg.get("compute_cot_loss", True)):
        for key in (
            "cot_step_input_ids",
            "cot_step_attention_mask",
            "cot_step_labels",
        ):
            batch.pop(key, None)
    return batch


def build_optimizer(model: Qwen3VLMultiCameraRiskMLLM, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    lora_learning_rate = train_cfg.get("lora_learning_rate")
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    if lora_learning_rate is None:
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    lora_learning_rate = float(lora_learning_rate)
    lora_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_lora_parameter(name):
            lora_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    parameter_groups = []
    if other_parameters:
        parameter_groups.append({"params": other_parameters, "lr": learning_rate, "weight_decay": weight_decay})
    if lora_parameters:
        parameter_groups.append({"params": lora_parameters, "lr": lora_learning_rate, "weight_decay": weight_decay})
    if not parameter_groups:
        raise ValueError("No trainable parameters found for optimizer")
    return torch.optim.AdamW(parameter_groups)


def _is_lora_parameter(name: str) -> bool:
    return "lora_" in name or ".lora_" in name


def load_risk_adapter_checkpoint(model: Qwen3VLMultiCameraRiskMLLM, checkpoint_path: str) -> None:
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        path = path / "adapter.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Risk adapter checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    try:
        model.qformer.load_state_dict(checkpoint["qformer"])
    except RuntimeError as exc:
        raise RuntimeError(
            "The risk adapter Q-Former is incompatible with the current Hugging Face "
            "Blip2QFormerModel backend. Checkpoints produced by the former custom "
            "CrossAttentionBlock implementation cannot be migrated automatically; "
            "reuse the Stage-1 Qwen adapter and restart training from Stage 2a."
        ) from exc
    model.risk_latent_tokens.data.copy_(checkpoint["risk_latent_tokens"].to(dtype=model.risk_latent_tokens.dtype))
    model.aux_heads.load_state_dict(checkpoint["aux_heads"], strict=False)
    cot_state = checkpoint.get("cot_decoder")
    if model.cot_decoder is not None and cot_state is not None:
        model.cot_decoder.load_state_dict(cot_state)


@torch.no_grad()
def run_forward_only_smoke(
    model: Qwen3VLMultiCameraRiskMLLM,
    loader: DataLoader,
    device: torch.device,
    train_cfg: Dict[str, Any],
) -> None:
    model.eval()
    batch = None
    for _, candidate in enumerate_valid_batches(loader, max_bad_batches=100, phase="forward_only"):
        batch = candidate
        break
    if batch is None:
        raise RuntimeError("No valid batch found for forward-only smoke test")
    batch = move_batch_to_device(batch, device)
    batch = prepare_batch_for_stage(batch, train_cfg)
    outputs = model(**batch)
    payload = {
        "mode": "forward_only",
        "loss": float(outputs["loss"].detach().cpu()),
        "answer_loss": float(outputs["answer_loss"].detach().cpu()),
        "weighted_answer_loss": float(outputs["weighted_answer_loss"].detach().cpu()),
        "aux_loss": float(outputs["aux_loss"].detach().cpu()),
        "weighted_aux_loss": float(outputs["weighted_aux_loss"].detach().cpu()),
        "cot_loss": float(outputs["cot_loss"].detach().cpu()),
        "weighted_cot_loss": float(outputs["weighted_cot_loss"].detach().cpu()),
        "fused_visual_tokens_shape": list(outputs["fused_visual_tokens"].shape),
        "latent_hidden_states_shape": list(outputs["latent_hidden_states"].shape),
        "reasoning_states_shape": list(outputs["reasoning_states"].shape),
        "risk_head_states_shape": list(outputs["risk_head_states"].shape),
        "latent_range": list(outputs["latent_range"]),
        "attention_mask_shape": list(outputs["attention_mask"].shape),
        "labels_shape": list(outputs["labels"].shape) if outputs["labels"] is not None else None,
        "sample_ids": batch.get("sample_ids"),
        "scenes": batch.get("scenes"),
    }
    json_loss = outputs.get("json_loss")
    if json_loss is not None:
        payload["json_loss"] = float(json_loss.detach().cpu())
    payload.update(flatten_loss_components(outputs))
    print(json.dumps(payload, ensure_ascii=False))


@torch.no_grad()
def evaluate(
    model: Qwen3VLMultiCameraRiskMLLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    max_bad_batches: int = 100,
    train_cfg: Dict[str, Any] | None = None,
) -> None:
    train_cfg = train_cfg or {}
    model.eval()
    losses = []
    for index, batch in enumerate_valid_batches(loader, max_bad_batches=max_bad_batches, phase="eval"):
        if index > max_batches:
            break
        batch = move_batch_to_device(batch, device)
        batch = prepare_batch_for_stage(batch, train_cfg)
        outputs = model(**batch)
        losses.append(float(outputs["loss"].detach().cpu()))
    if losses:
        print(json.dumps({"eval_loss": sum(losses) / len(losses), "eval_batches": len(losses)}))
    set_train_mode(model, train_cfg)


def save_checkpoint(
    path: Path,
    model: Qwen3VLMultiCameraRiskMLLM,
    processor: Any,
    config: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "qformer": model.qformer.state_dict(),
            "risk_latent_tokens": model.risk_latent_tokens.detach().cpu(),
            "aux_heads": model.aux_heads.state_dict(),
            "cot_decoder": model.cot_decoder.state_dict() if model.cot_decoder is not None else None,
        },
        path / "adapter.pt",
    )
    if bool(train_cfg.get("save_qwen_adapter", True)) and hasattr(model.qwen_model, "save_pretrained"):
        model.qwen_model.save_pretrained(path / "qwen_lora_or_model")
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(path / "processor")
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def prune_step_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return

    step_dirs = []
    for path in output_dir.glob("step-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        step_dirs.append((step, path))

    step_dirs.sort()
    excess = len(step_dirs) - save_total_limit
    if excess <= 0:
        return
    for _, path in step_dirs[:excess]:
        shutil.rmtree(path)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                inner_key: inner_value.to(device) if isinstance(inner_value, Tensor) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def infer_adapter_device(qwen_model: torch.nn.Module) -> torch.device:
    try:
        return next(qwen_model.get_input_embeddings().parameters()).device
    except StopIteration:
        return next(qwen_model.parameters()).device


def log_train_step(epoch: int, global_step: int, outputs: Dict[str, Any]) -> None:
    payload = {
        "epoch": epoch,
        "step": global_step,
        "loss": float(outputs["loss"].detach().cpu()),
        "answer_loss": float(outputs["answer_loss"].detach().cpu()),
        "weighted_answer_loss": float(outputs["weighted_answer_loss"].detach().cpu()),
        "aux_loss": float(outputs["aux_loss"].detach().cpu()),
        "weighted_aux_loss": float(outputs["weighted_aux_loss"].detach().cpu()),
        "cot_loss": float(outputs["cot_loss"].detach().cpu()),
        "weighted_cot_loss": float(outputs["weighted_cot_loss"].detach().cpu()),
    }
    json_loss = outputs.get("json_loss")
    if json_loss is not None:
        payload["json_loss"] = float(json_loss.detach().cpu())
    payload.update(flatten_loss_components(outputs))
    print(json.dumps(payload))


def flatten_loss_components(outputs: Dict[str, Any]) -> Dict[str, float]:
    aux_outputs = outputs.get("aux_outputs", {})
    losses = aux_outputs.get("losses", {}) if isinstance(aux_outputs, dict) else {}
    flattened = {}
    for name, value in losses.items():
        if isinstance(value, Tensor):
            flattened[f"aux_{name}"] = float(value.detach().cpu())
    return flattened


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def normalize_risk_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    for key in ("trend_labels", "action_labels", "lateral_action_labels"):
        if key in normalized and isinstance(normalized[key], list):
            normalized[key] = tuple(normalized[key])
    return normalized


def dtype_to_torch(dtype: str) -> torch.dtype | str:
    if dtype in ("auto", None):
        return "auto"
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Reading YAML configs requires PyYAML") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
