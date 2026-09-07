#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate Stage-3 risk latent auxiliary heads on validation samples."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from risk_mllm.data import Qwen3VLMultiCameraCollator
from train_qwen3_vl import (
    build_dataset,
    build_model,
    freeze_qwen_video_encoder,
    infer_adapter_device,
    load_qwen3_vl,
    load_risk_adapter_checkpoint,
    load_yaml,
    maybe_subsample_dataset,
    prepare_batch_for_stage,
)


CLASSIFICATION_FIELDS = {
    "current_risk_score": {
        "logit_key": "current_risk_score",
        "target_key": "current_risk_score",
        "labels": [0, 1, 2, 3, 4, 5],
    },
    "future_worst_risk_score": {
        "logit_key": "future_worst_risk_score",
        "target_key": "future_worst_risk_score",
        "labels": [0, 1, 2, 3, 4, 5],
    },
    "risk_trend": {
        "logit_key": "risk_trend",
        "target_key": "risk_trend",
        "labels_from_config": "trend_labels",
    },
    "mitigation_action": {
        "logit_key": "mitigation_action",
        "target_key": "mitigation_action",
        "labels_from_config": "action_labels",
    },
    "lateral_mitigation_action": {
        "logit_key": "lateral_mitigation_action",
        "target_key": "lateral_mitigation_action",
        "labels_from_config": "lateral_action_labels",
    },
}

REGRESSION_FIELDS = {
    "recommended_ego_speed_mps": {
        "prediction_key": "recommended_ego_speed_mps",
        "target_key": "recommended_ego_speed_mps",
    },
    "suggested_deceleration_mps2": {
        "prediction_key": "suggested_deceleration_mps2",
        "target_key": "suggested_deceleration_mps2",
    },
    "target_lateral_offset_m": {
        "prediction_key": "target_lateral_offset_m",
        "target_key": "target_lateral_offset_m",
    },
    "future_ttc": {
        "prediction_key": "future_ttc",
        "target_key": "future_ttc",
    },
    "future_dtc": {
        "prediction_key": "future_dtc",
        "target_key": "future_dtc",
    },
}

VECTOR_REGRESSION_FIELDS = {
    "current_ttc": {
        "prediction_key": "current_ttc",
        "target_key": "current_ttc",
        "components": ["longitudinal", "lateral"],
    },
    "current_dtc": {
        "prediction_key": "current_dtc",
        "target_key": "current_dtc",
        "components": ["longitudinal", "lateral"],
    },
    "future_ttc_vector": {
        "prediction_key": "future_ttc_vector",
        "target_key": "future_ttc_vector",
        "components": ["longitudinal", "lateral"],
    },
    "future_dtc_vector": {
        "prediction_key": "future_dtc_vector",
        "target_key": "future_dtc_vector",
        "components": ["longitudinal", "lateral"],
    },
}

FINITE_CLASSIFICATION_FIELDS = {
    "current_ttc_finite": {
        "logit_key": "current_ttc_finite",
        "target_key": "current_ttc_finite",
        "components": ["longitudinal", "lateral"],
    },
    "future_ttc_finite": {
        "logit_key": "future_ttc_finite",
        "target_key": "future_ttc_finite",
        "components": ["longitudinal", "lateral"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        default="training/outputs/stage3_risk_latent_warmup_15k/final",
        help="Stage-3 final/checkpoint directory containing adapter.pt and qwen_lora_or_model.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config JSON/YAML. Defaults to <checkpoint-dir>/config.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/outputs/stage3_risk_latent_warmup_15k/eval",
        help="Directory for metrics and per-sample predictions.",
    )
    parser.add_argument("--split", choices=("validation", "train"), default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print_json({"event": "load_config", "checkpoint_dir": str(checkpoint_dir)})
    config = load_config(Path(args.config).expanduser() if args.config else checkpoint_dir / "config.json")
    train_cfg = config.setdefault("training", {})
    train_cfg["compute_json_loss"] = True
    train_cfg["compute_risk_loss"] = True
    train_cfg["compute_cot_loss"] = False

    qwen_cfg = config.setdefault("qwen3_vl", {})
    stage3_qwen_lora = checkpoint_dir / "qwen_lora_or_model"
    if stage3_qwen_lora.is_dir():
        qwen_cfg["adapter_name_or_path"] = str(stage3_qwen_lora)
    qwen_cfg["adapter_is_trainable"] = False

    print_json({"event": "load_qwen3_vl", "adapter": qwen_cfg.get("adapter_name_or_path")})
    processor, qwen_model = load_qwen3_vl(qwen_cfg)
    freeze_qwen_video_encoder(qwen_model)
    for parameter in qwen_model.parameters():
        parameter.requires_grad = False

    print_json({"event": "build_model"})
    model = build_model(qwen_model, config)
    print_json({"event": "load_risk_adapter", "path": str(checkpoint_dir)})
    load_risk_adapter_checkpoint(model, str(checkpoint_dir))
    for parameter in model.parameters():
        parameter.requires_grad = False

    device = infer_adapter_device(qwen_model)
    model.qformer.to(device)
    model.aux_heads.to(device)
    if model.cot_decoder is not None:
        model.cot_decoder.to(device)
    model.risk_latent_tokens.data = model.risk_latent_tokens.data.to(device)
    model.eval()

    print_json({"event": "build_dataset", "split": args.split})
    dataset = build_dataset(config["data"], split=args.split)
    max_samples = args.max_samples
    if max_samples is None and args.split == "validation":
        max_samples = train_cfg.get("max_validation_samples")
    dataset = maybe_subsample_dataset(
        dataset,
        max_samples,
        int(train_cfg.get("random_seed", 42)) + 17,
        name=args.split,
    )
    print_json({"event": "dataset_ready", "samples": len(dataset)})

    batch_size = args.batch_size or int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 1)))
    collator = Qwen3VLMultiCameraCollator(
        processor=processor,
        risk_config=model.risk_config,
        max_question_length=int(train_cfg.get("max_question_length", 256)),
        max_answer_length=int(train_cfg.get("max_answer_length", 384)),
        max_cot_step_length=int(train_cfg.get("max_cot_step_length", model.risk_config.max_cot_step_length)),
        video_processor_kwargs=config.get("video_processor", {"fps": 1.0}),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collator,
    )

    accum = MetricsAccumulator(model.risk_config)
    predictions_path = output_dir / f"{args.split}_head_predictions.jsonl"
    total_loss = 0.0
    total_batches = 0
    total_samples = 0

    with predictions_path.open("w", encoding="utf-8") as pred_file:
        bad_batches = 0
        iterator = iter(loader)
        batch_index = 0
        while True:
            print_json({"event": "fetch_batch", "next_batch": batch_index + 1})
            try:
                batch = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                bad_batches += 1
                print_json({"event": "skip_bad_batch", "bad_batches": bad_batches, "error": str(exc)})
                if bad_batches > 20:
                    raise RuntimeError("Exceeded 20 bad batches during evaluation") from exc
                continue
            batch_index += 1
            if args.max_batches is not None and batch_index > args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            batch = prepare_batch_for_stage(batch, train_cfg)
            with torch.inference_mode():
                outputs = model(**batch)
            batch_size_actual = int(next(iter(batch["risk_targets"].values())).shape[0])
            total_batches += 1
            total_samples += batch_size_actual
            total_loss += float(outputs["loss"].detach().float().cpu()) * batch_size_actual
            accum.update(outputs["aux_outputs"], batch["risk_targets"])
            for record in build_prediction_records(outputs["aux_outputs"], batch["risk_targets"], batch):
                pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print_json({"event": "batch_done", "batch": batch_index, "samples": total_samples})

    metrics = accum.compute()
    metrics["eval_loss"] = total_loss / max(total_samples, 1)
    metrics["samples"] = total_samples
    metrics["batches"] = total_batches
    metrics["checkpoint_dir"] = str(checkpoint_dir)
    metrics["split"] = args.split

    write_json(output_dir / f"{args.split}_head_metrics.json", metrics)
    write_json(output_dir / "eval_run_config.json", vars(args))
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


class MetricsAccumulator:
    def __init__(self, risk_config: Any) -> None:
        self.risk_config = risk_config
        self.classification: Dict[str, Dict[str, list[int]]] = {
            name: {"pred": [], "target": []} for name in CLASSIFICATION_FIELDS
        }
        self.finite_classification: Dict[str, Dict[str, list[int]]] = {}
        for name, spec in FINITE_CLASSIFICATION_FIELDS.items():
            for component in spec["components"]:
                self.finite_classification[f"{name}_{component}"] = {"pred": [], "target": []}
        self.regression: Dict[str, Dict[str, list[float]]] = {
            name: {"pred": [], "target": []} for name in REGRESSION_FIELDS
        }
        for name, spec in VECTOR_REGRESSION_FIELDS.items():
            for component in spec["components"]:
                self.regression[f"{name}_{component}"] = {"pred": [], "target": []}

    def update(self, aux_outputs: Dict[str, Any], risk_targets: Dict[str, Tensor]) -> None:
        logits = aux_outputs.get("logits", {})
        predictions = aux_outputs.get("predictions", {})
        for name, spec in CLASSIFICATION_FIELDS.items():
            pred = argmax_predictions(logits.get(spec["logit_key"]))
            target = tensor_to_long_list(risk_targets.get(spec["target_key"]))
            append_valid_classes(self.classification[name], pred, target, self.risk_config.ignore_index)

        for name, spec in FINITE_CLASSIFICATION_FIELDS.items():
            pred_tensor = logits.get(spec["logit_key"])
            target_tensor = risk_targets.get(spec["target_key"])
            if not isinstance(pred_tensor, Tensor) or not isinstance(target_tensor, Tensor):
                continue
            pred = torch.argmax(pred_tensor.detach().float().cpu(), dim=-1)
            target = target_tensor.detach().long().cpu()
            for index, component in enumerate(spec["components"]):
                append_valid_classes(
                    self.finite_classification[f"{name}_{component}"],
                    pred[:, index].tolist(),
                    target[:, index].tolist(),
                    self.risk_config.ignore_index,
                )

        for name, spec in REGRESSION_FIELDS.items():
            pred = tensor_to_float_list(predictions.get(spec["prediction_key"]))
            target = tensor_to_float_list(risk_targets.get(spec["target_key"]))
            append_valid_regression(self.regression[name], pred, target)

        for name, spec in VECTOR_REGRESSION_FIELDS.items():
            pred_tensor = predictions.get(spec["prediction_key"])
            target_tensor = risk_targets.get(spec["target_key"])
            if not isinstance(pred_tensor, Tensor) or not isinstance(target_tensor, Tensor):
                continue
            pred = pred_tensor.detach().float().cpu()
            target = target_tensor.detach().float().cpu()
            for index, component in enumerate(spec["components"]):
                append_valid_regression(
                    self.regression[f"{name}_{component}"],
                    pred[:, index].tolist(),
                    target[:, index].tolist(),
                )

    def compute(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "classification": {},
            "finite_classification": {},
            "regression": {},
        }
        for name, values in self.classification.items():
            labels = labels_for_field(name, CLASSIFICATION_FIELDS[name], self.risk_config)
            metrics["classification"][name] = classification_metrics(values["pred"], values["target"], labels)
        for name, values in self.finite_classification.items():
            metrics["finite_classification"][name] = classification_metrics(values["pred"], values["target"], [0, 1])
        for name, values in self.regression.items():
            metrics["regression"][name] = regression_metrics(values["pred"], values["target"])

        cur_pred = self.classification["current_risk_score"]["pred"]
        cur_target = self.classification["current_risk_score"]["target"]
        fut_pred = self.classification["future_worst_risk_score"]["pred"]
        fut_target = self.classification["future_worst_risk_score"]["target"]
        metrics["high_risk"] = high_risk_metrics(cur_pred, cur_target, fut_pred, fut_target)
        return metrics


def labels_for_field(name: str, spec: Dict[str, Any], risk_config: Any) -> list[Any]:
    del name
    if "labels" in spec:
        return list(spec["labels"])
    return list(getattr(risk_config, spec["labels_from_config"]))


def classification_metrics(pred: list[int], target: list[int], labels: list[Any]) -> Dict[str, Any]:
    if not target:
        return {"count": 0}
    correct = sum(int(p == t) for p, t in zip(pred, target))
    label_ids = list(range(len(labels))) if labels and not isinstance(labels[0], int) else [int(v) for v in labels]
    confusion = {str(label): {str(other): 0 for other in label_ids} for label in label_ids}
    for p, t in zip(pred, target):
        confusion.setdefault(str(t), {str(other): 0 for other in label_ids})
        confusion[str(t)][str(p)] = confusion[str(t)].get(str(p), 0) + 1

    f1_values = []
    per_class = {}
    for label in label_ids:
        tp = sum(1 for p, t in zip(pred, target) if p == label and t == label)
        fp = sum(1 for p, t in zip(pred, target) if p == label and t != label)
        fn = sum(1 for p, t in zip(pred, target) if p != label and t == label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        support = sum(1 for t in target if t == label)
        f1_values.append(f1)
        per_class[str(label)] = {
            "label": labels[label] if label < len(labels) else label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    output = {
        "count": len(target),
        "accuracy": correct / len(target),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
    if all(isinstance(value, int) for value in labels):
        output["ordinal_mae"] = sum(abs(p - t) for p, t in zip(pred, target)) / len(target)
    return output


def high_risk_metrics(
    current_pred: list[int],
    current_target: list[int],
    future_pred: list[int],
    future_target: list[int],
) -> Dict[str, Any]:
    count = min(len(current_pred), len(current_target), len(future_pred), len(future_target))
    if count == 0:
        return {"count": 0}
    pred = [min(current_pred[i], future_pred[i]) <= 2 for i in range(count)]
    target = [min(current_target[i], future_target[i]) <= 2 for i in range(count)]
    tp = sum(1 for p, t in zip(pred, target) if p and t)
    fp = sum(1 for p, t in zip(pred, target) if p and not t)
    fn = sum(1 for p, t in zip(pred, target) if not p and t)
    tn = sum(1 for p, t in zip(pred, target) if not p and not t)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "count": count,
        "accuracy": safe_div(tp + tn, count),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "false_negative_rate": safe_div(fn, fn + tp),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def regression_metrics(pred: list[float], target: list[float]) -> Dict[str, Any]:
    if not target:
        return {"count": 0}
    errors = [p - t for p, t in zip(pred, target)]
    abs_errors = [abs(value) for value in errors]
    squared_errors = [value * value for value in errors]
    return {
        "count": len(target),
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
        "mean_error": sum(errors) / len(errors),
    }


def build_prediction_records(
    aux_outputs: Dict[str, Any],
    risk_targets: Dict[str, Tensor],
    batch: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    logits = aux_outputs.get("logits", {})
    predictions = aux_outputs.get("predictions", {})
    batch_size = int(next(iter(risk_targets.values())).shape[0])
    sample_ids = batch.get("sample_ids", [None] * batch_size)
    scenes = batch.get("scenes", [None] * batch_size)
    for index in range(batch_size):
        record = {
            "sample_id": sample_ids[index] if index < len(sample_ids) else None,
            "scene": scenes[index] if index < len(scenes) else None,
            "classification": {},
            "regression": {},
        }
        for name, spec in CLASSIFICATION_FIELDS.items():
            pred = class_prediction_at(logits.get(spec["logit_key"]), index)
            target = tensor_value_at(risk_targets.get(spec["target_key"]), index)
            record["classification"][name] = {"prediction": pred, "target": target}
        for name, spec in REGRESSION_FIELDS.items():
            pred = tensor_value_at(predictions.get(spec["prediction_key"]), index)
            target = tensor_value_at(risk_targets.get(spec["target_key"]), index)
            record["regression"][name] = {"prediction": pred, "target": target}
        for name, spec in VECTOR_REGRESSION_FIELDS.items():
            pred = tensor_row_at(predictions.get(spec["prediction_key"]), index)
            target = tensor_row_at(risk_targets.get(spec["target_key"]), index)
            record["regression"][name] = {"prediction": pred, "target": target}
        yield record


def argmax_predictions(tensor: Any) -> list[int]:
    if not isinstance(tensor, Tensor):
        return []
    values = tensor.detach().float().cpu()
    if values.dim() == 1:
        values = values.unsqueeze(0)
    return torch.argmax(values, dim=-1).tolist()


def class_prediction_at(tensor: Any, index: int) -> Optional[int]:
    values = argmax_predictions(tensor)
    return values[index] if index < len(values) else None


def tensor_to_long_list(tensor: Any) -> list[int]:
    if not isinstance(tensor, Tensor):
        return []
    return tensor.detach().long().cpu().view(-1).tolist()


def tensor_to_float_list(tensor: Any) -> list[float]:
    if not isinstance(tensor, Tensor):
        return []
    return tensor.detach().float().cpu().view(-1).tolist()


def tensor_value_at(tensor: Any, index: int) -> Optional[float]:
    values = tensor_to_float_list(tensor)
    if index >= len(values):
        return None
    value = values[index]
    return value if math.isfinite(value) else None


def tensor_row_at(tensor: Any, index: int) -> Optional[list[Optional[float]]]:
    if not isinstance(tensor, Tensor):
        return None
    values = tensor.detach().float().cpu()
    if values.dim() == 1:
        values = values.unsqueeze(0)
    if index >= values.size(0):
        return None
    output = []
    for value in values[index].view(-1).tolist():
        output.append(value if math.isfinite(value) else None)
    return output


def append_valid_classes(store: Dict[str, list[int]], pred: list[int], target: list[int], ignore_index: int) -> None:
    for pred_value, target_value in zip(pred, target):
        if target_value == ignore_index or target_value < 0:
            continue
        store["pred"].append(int(pred_value))
        store["target"].append(int(target_value))


def append_valid_regression(store: Dict[str, list[float]], pred: list[float], target: list[float]) -> None:
    for pred_value, target_value in zip(pred, target):
        if not math.isfinite(float(target_value)) or not math.isfinite(float(pred_value)):
            continue
        store["pred"].append(float(pred_value))
        store["target"].append(float(target_value))


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


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_yaml(str(path))
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
