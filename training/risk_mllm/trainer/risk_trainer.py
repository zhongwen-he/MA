#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hugging Face Trainer adapter for project-specific risk-latent training."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from torch import Tensor
from transformers import Trainer


class RiskMLLMTrainer(Trainer):
    """Trainer that keeps HF's training engine but saves project adapters.

    The wrapped model already computes the full training objective in
    ``forward``. This class adapts Hugging Face Trainer to:

    - accept custom multimodal batch fields such as ``risk_targets`` and
      ``cot_step_labels``;
    - log JSON/risk/CoT loss components;
    - save only the project risk adapter by default instead of a full Qwen
      model copy.
    """

    def __init__(
        self,
        *args: Any,
        processor: Optional[Any] = None,
        raw_config: Optional[Dict[str, Any]] = None,
        train_cfg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.raw_config = raw_config or {}
        self.train_cfg = train_cfg or {}
        self._last_loss_components: Dict[str, float] = {}

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **_: Any,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        outputs = model(**inputs)
        loss = outputs["loss"]
        self._last_loss_components = self._flatten_loss_components(outputs)
        if return_outputs:
            return loss, self._lightweight_outputs(outputs)
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"].detach()
        return loss, None, None

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer

        learning_rate = float(self.train_cfg.get("learning_rate", self.args.learning_rate))
        lora_learning_rate = self.train_cfg.get("lora_learning_rate")
        weight_decay = float(self.train_cfg.get("weight_decay", self.args.weight_decay))
        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not named_parameters:
            raise ValueError("No trainable parameters found for optimizer")

        parameter_groups = self._build_project_parameter_groups(
            named_parameters=named_parameters,
            base_learning_rate=learning_rate,
            base_weight_decay=weight_decay,
        )
        if parameter_groups:
            self.optimizer = torch.optim.AdamW(parameter_groups)
            return self.optimizer

        lora_parameters = []
        other_parameters = []
        for name, parameter in named_parameters:
            if self._is_lora_parameter(name):
                lora_parameters.append(parameter)
            else:
                other_parameters.append(parameter)

        parameter_groups = []
        if other_parameters:
            parameter_groups.append({"params": other_parameters, "lr": learning_rate, "weight_decay": weight_decay})
        if lora_parameters:
            parameter_groups.append(
                {
                    "params": lora_parameters,
                    "lr": float(lora_learning_rate),
                    "weight_decay": weight_decay,
                }
            )
        self.optimizer = torch.optim.AdamW(parameter_groups)
        return self.optimizer

    def _build_project_parameter_groups(
        self,
        named_parameters: list[tuple[str, torch.nn.Parameter]],
        base_learning_rate: float,
        base_weight_decay: float,
    ) -> list[dict[str, Any]]:
        """Build optional differential LR groups for Stage-2+ adapters."""

        group_specs = [
            ("qformer.query_tokens", "query_tokens_learning_rate"),
            ("qformer.qformer.", "qformer_learning_rate"),
            ("qformer.visual_adapter.", "visual_adapter_learning_rate"),
            ("qformer.projector.", "projector_learning_rate"),
            ("qformer.camera_embedding.", "camera_embedding_learning_rate"),
            ("qformer.temporal_embedding.", "temporal_embedding_learning_rate"),
            ("risk_latent_tokens", "risk_latent_learning_rate"),
            ("aux_heads.", "aux_heads_learning_rate"),
            ("cot_decoder.", "cot_decoder_learning_rate"),
        ]
        configured = {key for _, key in group_specs if self.train_cfg.get(key) is not None}
        if not configured:
            return []

        grouped: dict[str, list[torch.nn.Parameter]] = {key: [] for _, key in group_specs}
        default_parameters: list[torch.nn.Parameter] = []
        for name, parameter in named_parameters:
            assigned = False
            for prefix, lr_key in group_specs:
                if name == prefix or name.startswith(prefix):
                    grouped[lr_key].append(parameter)
                    assigned = True
                    break
            if not assigned:
                default_parameters.append(parameter)

        parameter_groups: list[dict[str, Any]] = []
        if default_parameters:
            parameter_groups.append(
                {
                    "params": default_parameters,
                    "lr": base_learning_rate,
                    "weight_decay": base_weight_decay,
                }
            )
        for _, lr_key in group_specs:
            parameters = grouped[lr_key]
            if not parameters:
                continue
            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": float(self.train_cfg.get(lr_key, base_learning_rate)),
                    "weight_decay": base_weight_decay,
                }
            )
        return parameter_groups

    def log(self, logs: Dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._last_loss_components:
            logs = {**logs, **self._last_loss_components}
        super().log(logs, *args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        del _internal_call
        output_path = Path(output_dir or self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._save_risk_adapter(output_path)

    def _issue_warnings_after_load(self, load_result: Any) -> None:
        if not hasattr(self.model, "_keys_to_ignore_on_save"):
            self.model._keys_to_ignore_on_save = None
        if not hasattr(self.model, "_keys_to_ignore_on_load_missing"):
            self.model._keys_to_ignore_on_load_missing = None
        if not hasattr(self.model, "_keys_to_ignore_on_load_unexpected"):
            self.model._keys_to_ignore_on_load_unexpected = None
        # The checkpoint intentionally contains only project adapter weights,
        # not the frozen Qwen backbone. Missing base-model keys are expected.
        del load_result

    def _load_scaler(self, resume_from_checkpoint: str) -> None:
        if getattr(self.accelerator, "scaler", None) is None:
            return
        super()._load_scaler(resume_from_checkpoint)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, Any]] = None) -> None:
        del state_dict
        self.save_model(output_dir)

    def _save_risk_adapter(self, output_path: Path) -> None:
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        adapter_state = self._adapter_state_dict(model)
        torch.save(
            {
                "qformer": model.qformer.state_dict(),
                "risk_latent_tokens": model.risk_latent_tokens.detach().cpu(),
                "aux_heads": model.aux_heads.state_dict(),
                "cot_decoder": model.cot_decoder.state_dict() if model.cot_decoder is not None else None,
            },
            output_path / "adapter.pt",
        )
        torch.save(adapter_state, output_path / "pytorch_model.bin")
        if bool(self.train_cfg.get("save_qwen_adapter", True)) and hasattr(model.qwen_model, "save_pretrained"):
            model.qwen_model.save_pretrained(output_path / "qwen_lora_or_model")
        if self.processor is not None and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(output_path / "processor")
        if self.raw_config:
            with (output_path / "config.json").open("w", encoding="utf-8") as f:
                json.dump(self.raw_config, f, ensure_ascii=False, indent=2)

    def _adapter_state_dict(self, model: torch.nn.Module) -> Dict[str, Tensor]:
        state_dict: Dict[str, Tensor] = {
            "risk_latent_tokens": model.risk_latent_tokens.detach().cpu(),
        }
        for prefix, module in (
            ("qformer", model.qformer),
            ("aux_heads", model.aux_heads),
            ("cot_decoder", model.cot_decoder),
        ):
            if module is None:
                continue
            for name, tensor in module.state_dict().items():
                state_dict[f"{prefix}.{name}"] = tensor.detach().cpu()
        return state_dict

    def _flatten_loss_components(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        components: Dict[str, float] = {}
        for key in (
            "json_loss",
            "answer_loss",
            "weighted_answer_loss",
            "aux_loss",
            "weighted_aux_loss",
            "cot_loss",
            "weighted_cot_loss",
        ):
            value = outputs.get(key)
            if isinstance(value, Tensor):
                components[key] = float(value.detach().float().cpu())

        aux_outputs = outputs.get("aux_outputs", {})
        losses = aux_outputs.get("losses", {}) if isinstance(aux_outputs, dict) else {}
        for name, value in losses.items():
            if isinstance(value, Tensor):
                components[f"aux_{name}"] = float(value.detach().float().cpu())
        return components

    def _lightweight_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        keep = {
            "loss": outputs.get("loss"),
            "json_loss": outputs.get("json_loss"),
            "answer_loss": outputs.get("answer_loss"),
            "weighted_answer_loss": outputs.get("weighted_answer_loss"),
            "aux_loss": outputs.get("aux_loss"),
            "weighted_aux_loss": outputs.get("weighted_aux_loss"),
            "cot_loss": outputs.get("cot_loss"),
            "weighted_cot_loss": outputs.get("weighted_cot_loss"),
        }
        return {key: value for key, value in keep.items() if value is not None}

    def _is_lora_parameter(self, name: str) -> bool:
        return "lora_" in name or ".lora_" in name
