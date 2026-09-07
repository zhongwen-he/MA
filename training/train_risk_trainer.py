#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Trainer-based entrypoint for Stage 2/3 risk-latent training."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from torch.utils.data import Subset
from transformers import TrainingArguments

from risk_mllm.data import Qwen3VLMultiCameraCollator
from risk_mllm.trainer import RiskMLLMTrainer
from train_qwen3_vl import (
    build_dataset,
    build_model,
    configure_adapter_training,
    configure_llm_training,
    count_trainable_parameters,
    load_qwen3_vl,
    load_risk_adapter_checkpoint,
    load_yaml,
    maybe_subsample_dataset,
    prepare_batch_for_stage,
    set_train_mode,
    freeze_qwen_video_encoder,
    infer_adapter_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps for smoke tests.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Override training.max_train_samples.")
    parser.add_argument("--max-validation-samples", type=int, default=None, help="Override training.max_validation_samples.")
    parser.add_argument("--eval-max-batches", type=int, default=None, help="Override training.eval_max_batches.")
    parser.add_argument("--save-steps", type=int, default=None, help="Override training.save_every.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override training.eval_every.")
    parser.add_argument("--log-steps", type=int, default=None, help="Override training.log_every.")
    parser.add_argument("--eval-only", action="store_true", help="Load the model/adapter and run validation loss only.")
    parser.add_argument(
        "--risk-adapter-checkpoint",
        default=None,
        help="Override training.risk_adapter_name_or_path, e.g. a Stage-2 final/ or checkpoint directory.",
    )
    parser.add_argument(
        "--eval-output-dir",
        default=None,
        help="Directory for eval-only metrics. Defaults to <output_dir>/eval_only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train_cfg = config.get("training", {})
    apply_cli_overrides(train_cfg, args)
    output_dir = Path(args.output_dir or train_cfg.get("output_dir", "training/outputs/risk_trainer"))
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, qwen_model = load_qwen3_vl(config.get("qwen3_vl", {}))
    if train_cfg.get("freeze_video_encoder", True):
        freeze_qwen_video_encoder(qwen_model)
    qwen_model = configure_llm_training(qwen_model, train_cfg)
    model = build_model(qwen_model, config)

    risk_adapter_path = args.risk_adapter_checkpoint or train_cfg.get("risk_adapter_name_or_path")
    if risk_adapter_path:
        load_risk_adapter_checkpoint(model, risk_adapter_path)
    configure_adapter_training(model, train_cfg)
    adapter_device = infer_adapter_device(qwen_model)
    model.qformer.to(adapter_device)
    model.aux_heads.to(adapter_device)
    if model.cot_decoder is not None:
        model.cot_decoder.to(adapter_device)
    model.risk_latent_tokens.data = model.risk_latent_tokens.data.to(adapter_device)
    set_train_mode(model, train_cfg)

    train_dataset = build_dataset(config["data"], split="train")
    eval_dataset = build_dataset(config["data"], split="validation")
    train_dataset = maybe_subsample_dataset(
        train_dataset,
        train_cfg.get("max_train_samples"),
        int(train_cfg.get("random_seed", 42)),
        name="train",
    )
    eval_dataset = maybe_subsample_dataset(
        eval_dataset,
        train_cfg.get("max_validation_samples"),
        int(train_cfg.get("random_seed", 42)) + 1,
        name="validation",
    )
    eval_dataset = maybe_limit_eval_dataset(eval_dataset, train_cfg, int(train_cfg.get("random_seed", 42)) + 2)
    print(json.dumps({"train_samples": len(train_dataset), "eval_samples": len(eval_dataset)}))
    print(json.dumps({"trainable_parameters": count_trainable_parameters(model)}))

    collator = Qwen3VLMultiCameraCollator(
        processor=processor,
        risk_config=model.risk_config,
        max_question_length=int(train_cfg.get("max_question_length", 1024)),
        max_answer_length=int(train_cfg.get("max_answer_length", 1024)),
        max_cot_step_length=int(train_cfg.get("max_cot_step_length", model.risk_config.max_cot_step_length)),
        video_processor_kwargs=config.get("video_processor", {"fps": 1.0}),
    )
    data_collator = StageBatchCollator(collator, train_cfg)

    if args.forward_only:
        run_forward_only(model, train_dataset, data_collator)
        return

    training_args = build_training_arguments(output_dir, train_cfg)
    trainer = RiskMLLMTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        raw_config=config,
        train_cfg=train_cfg,
    )
    if args.eval_only:
        eval_metrics = trainer.evaluate()
        eval_output_dir = Path(args.eval_output_dir or output_dir / "eval_only")
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        with (eval_output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
        print(json.dumps({"eval_metrics_path": str(eval_output_dir / "eval_metrics.json")}, ensure_ascii=False))
        return
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    if bool(train_cfg.get("skip_final_eval", False)):
        trainer.save_metrics("all", train_result.metrics)
        return
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    all_metrics = {**train_result.metrics, **eval_metrics}
    trainer.save_metrics("all", all_metrics)


class StageBatchCollator:
    def __init__(self, collator: Qwen3VLMultiCameraCollator, train_cfg: Dict[str, Any]) -> None:
        self.collator = collator
        self.train_cfg = train_cfg

    def __call__(self, samples: list[Dict[str, Any]]) -> Dict[str, Any]:
        return prepare_batch_for_stage(self.collator(samples), self.train_cfg)


def apply_cli_overrides(train_cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        "max_steps": args.max_steps,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "eval_max_batches": args.eval_max_batches,
        "save_every": args.save_steps,
        "eval_every": args.eval_steps,
        "log_every": args.log_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            train_cfg[key] = value


def build_training_arguments(output_dir: Path, train_cfg: Dict[str, Any]) -> TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "do_train": True,
        "do_eval": train_cfg.get("eval_strategy", "steps") != "no",
        "per_device_train_batch_size": int(train_cfg.get("batch_size", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 1))),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "num_train_epochs": int(train_cfg.get("num_epochs", 1)),
        "max_steps": int(train_cfg["max_steps"]) if train_cfg.get("max_steps") is not None else -1,
        "logging_steps": int(train_cfg.get("log_every", 10)),
        "save_steps": int(train_cfg.get("save_every", 200)),
        "eval_steps": int(train_cfg.get("eval_every", train_cfg.get("save_every", 200))),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "remove_unused_columns": False,
        "dataloader_num_workers": int(train_cfg.get("num_workers", 0)),
        "dataloader_pin_memory": bool(train_cfg.get("pin_memory", False)),
        "fp16": bool(train_cfg.get("fp16", True)),
        "bf16": bool(train_cfg.get("bf16", False)),
        "report_to": train_cfg.get("report_to", []),
        "logging_dir": str(output_dir / "runs"),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
    }
    if train_cfg.get("eval_strategy", "steps") == "no":
        kwargs["eval_strategy"] = "no"
    elif train_cfg.get("eval_strategy", "steps") == "steps":
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["eval_strategy"] = "epoch"
    if train_cfg.get("save_strategy", "steps") == "steps":
        kwargs["save_strategy"] = "steps"
    else:
        kwargs["save_strategy"] = "epoch"

    # Keep compatibility with older Transformers naming.
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def maybe_limit_eval_dataset(dataset: Any, train_cfg: Dict[str, Any], seed: int) -> Any:
    eval_max_batches = train_cfg.get("eval_max_batches")
    if eval_max_batches is None:
        return dataset
    max_examples = int(eval_max_batches) * int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 1)))
    if max_examples <= 0 or max_examples >= len(dataset):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_examples].tolist()
    return Subset(dataset, indices)


@torch.no_grad()
def run_forward_only(
    model: torch.nn.Module,
    train_dataset: Any,
    data_collator: StageBatchCollator,
) -> None:
    model.eval()
    batch = data_collator([train_dataset[0]])
    device = infer_adapter_device(model.qwen_model)
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = {
                inner_key: inner_value.to(device) if isinstance(inner_value, torch.Tensor) else inner_value
                for inner_key, inner_value in value.items()
            }
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
        "labels_shape": list(outputs["labels"].shape) if outputs["labels"] is not None else None,
        "latent_hidden_states_shape": list(outputs["latent_hidden_states"].shape),
        "reasoning_states_shape": list(outputs["reasoning_states"].shape),
        "risk_head_states_shape": list(outputs["risk_head_states"].shape),
    }
    json_loss = outputs.get("json_loss")
    if json_loss is not None:
        payload["json_loss"] = float(json_loss.detach().cpu())
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
