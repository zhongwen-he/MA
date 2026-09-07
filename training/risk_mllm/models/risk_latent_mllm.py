#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""LLM-side risk latent tokens, recurrent latent CoT, and risk supervision.

The Q-Former stays a visual fusion adapter. This module inserts learnable risk
latent tokens into the LLM context after fused multi-view visual tokens and
before answer tokens:

    [question tokens] [32 visual tokens] [8 risk latent tokens] [answer tokens]

Stage 3 can supervise those hidden states directly as a risk representation
warm-up. Stage 4/5 should instead generate recurrent implicit CoT states
``z_1 ... z_K`` from the risk representation and supervise the risk heads from
those step states, so the latent CoT is part of the main risk-reasoning path
rather than an auxiliary branch.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class RiskLatentConfig:
    llm_hidden_dim: int = 4096
    num_risk_latents: int = 8
    risk_score_classes: int = 6
    trend_labels: Tuple[str, ...] = ("improving", "stable", "worsening")
    action_labels: Tuple[str, ...] = (
        "maintain_speed",
        "accelerate",
        "mild_decelerate",
        "moderate_decelerate",
        "strong_decelerate",
        "stop",
        "stopping",
    )
    lateral_action_labels: Tuple[str, ...] = (
        "maintain_lane",
        "lateral_shift_left",
        "lateral_shift_right",
    )
    dropout: float = 0.1
    latent_init_std: float = 0.02
    ignore_index: int = -100
    answer_loss_weight: float = 1.0
    risk_loss_weight: float = 1.0
    current_risk_loss_weight: float = 0.5
    future_risk_loss_weight: float = 0.7
    trend_loss_weight: float = 0.3
    action_loss_weight: float = 0.5
    lateral_action_loss_weight: float = 0.3
    speed_loss_weight: float = 0.2
    deceleration_loss_weight: float = 0.2
    current_ttc_loss_weight: float = 0.0
    current_dtc_loss_weight: float = 0.0
    future_ttc_loss_weight: float = 0.2
    future_dtc_loss_weight: float = 0.2
    future_ttc_vector_loss_weight: float = -1.0
    future_dtc_vector_loss_weight: float = -1.0
    lateral_offset_loss_weight: float = 0.0
    ttc_loss_mode: str = "raw"
    ttc_finite_loss_weight: float = 0.0
    risk_head_source: str = "risk_latents"
    enable_recurrent_latent_cot: bool = False
    enable_latent_cot: bool = True
    num_cot_steps: int = 4
    cot_loss_weight: float = 0.3
    cot_decoder_hidden_dim: int = 512
    cot_decoder_layers: int = 2
    cot_decoder_heads: int = 8
    cot_decoder_dropout: float = 0.1
    max_cot_step_length: int = 128
    max_speed_mps: float = 60.0
    max_abs_acceleration_mps2: float = 12.0
    max_ttc_seconds: float = 60.0
    max_abs_dtc_meters: float = 200.0


class RiskAuxiliaryHeads(nn.Module):
    """Predict structured risk labels from H_R or recurrent CoT states.

    ``state_source="risk_latents"`` is the Stage 3 warm-up path:
    current/future/action/trend are pooled from the 8 risk-latent hidden states.

    ``state_source="recurrent_cot"`` is the Stage 4/5 path:
    z1 supervises current risk geometry, z2 future risk geometry, z3 trend, and
    z4 ego mitigation. This is the path that makes latent CoT serve the main
    risk objective.
    """

    def __init__(self, config: RiskLatentConfig) -> None:
        super().__init__()
        if config.num_risk_latents < 4:
            raise ValueError("Risk auxiliary heads expect at least 4 risk latent tokens")
        if config.num_cot_steps < 4:
            raise ValueError("Risk auxiliary heads expect at least 4 recurrent CoT steps")

        self.config = config
        hidden_dim = config.llm_hidden_dim
        self.current_risk_head = self._classification_head(hidden_dim, config.risk_score_classes)
        self.future_risk_head = self._classification_head(hidden_dim, config.risk_score_classes)
        self.trend_head = self._classification_head(hidden_dim, len(config.trend_labels))
        self.action_head = self._classification_head(hidden_dim, len(config.action_labels))
        self.lateral_action_head = self._classification_head(hidden_dim, len(config.lateral_action_labels))
        self.speed_head = self._regression_head(hidden_dim)
        self.deceleration_head = self._regression_head(hidden_dim)
        self.current_ttc_head = self._vector_regression_head(hidden_dim, 2)
        self.current_ttc_finite_head = self._vector_classification_head(hidden_dim, 2, 2)
        self.current_dtc_head = self._vector_regression_head(hidden_dim, 2)
        self.future_ttc_head = self._regression_head(hidden_dim)
        self.future_dtc_head = self._regression_head(hidden_dim)
        self.future_ttc_vector_head = self._vector_regression_head(hidden_dim, 2)
        self.future_ttc_finite_head = self._vector_classification_head(hidden_dim, 2, 2)
        self.future_dtc_vector_head = self._vector_regression_head(hidden_dim, 2)
        self.lateral_offset_head = self._regression_head(hidden_dim)

    def forward(
        self,
        latent_hidden_states: Tensor,
        targets: Optional[Dict[str, Tensor]] = None,
        state_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        if latent_hidden_states.dim() != 3:
            raise ValueError(f"Expected latent_hidden_states [B, Z, D], got rank {latent_hidden_states.dim()}")

        head_dtype = next(self.current_risk_head.parameters()).dtype
        latent_hidden_states = latent_hidden_states.to(dtype=head_dtype)

        source = state_source or self.config.risk_head_source
        h_cur, h_fut, h_trend, h_act = self._select_task_states(latent_hidden_states, source)

        raw_current_ttc = self.current_ttc_head(h_cur)
        raw_future_ttc = self.future_ttc_head(h_fut).squeeze(-1)
        raw_future_ttc_vector = self.future_ttc_vector_head(h_fut)
        logits = {
            "current_risk_score": self.current_risk_head(h_cur),
            "future_worst_risk_score": self.future_risk_head(h_fut),
            "risk_trend": self.trend_head(h_trend),
            "mitigation_action": self.action_head(h_act),
            "lateral_mitigation_action": self.lateral_action_head(h_act),
            "current_ttc_finite": self.current_ttc_finite_head(h_cur),
            "future_ttc_finite": self.future_ttc_finite_head(h_fut),
        }
        predictions = {
            "recommended_ego_speed_mps": self.speed_head(h_act).squeeze(-1),
            "suggested_deceleration_mps2": self.deceleration_head(h_act).squeeze(-1),
            "current_ttc": self._ttc_prediction(raw_current_ttc),
            "current_dtc": self.current_dtc_head(h_cur),
            "future_ttc": self._ttc_prediction(raw_future_ttc),
            "future_dtc": self.future_dtc_head(h_fut).squeeze(-1),
            "future_ttc_vector": self._ttc_prediction(raw_future_ttc_vector),
            "future_dtc_vector": self.future_dtc_vector_head(h_fut),
            "target_lateral_offset_m": self.lateral_offset_head(h_act).squeeze(-1),
        }

        losses: Dict[str, Tensor] = {}
        total_aux_loss = latent_hidden_states.new_zeros(())
        if targets:
            loss = self._classification_loss(
                logits["current_risk_score"],
                targets.get("current_risk_score"),
                self.config.risk_score_classes,
            )
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "current_risk_loss",
                loss,
                self.config.current_risk_loss_weight,
            )

            loss = self._classification_loss(
                logits["future_worst_risk_score"],
                targets.get("future_worst_risk_score"),
                self.config.risk_score_classes,
            )
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "future_risk_loss",
                loss,
                self.config.future_risk_loss_weight,
            )

            loss = self._classification_loss(
                logits["risk_trend"],
                targets.get("risk_trend"),
                len(self.config.trend_labels),
            )
            total_aux_loss = self._add_loss(losses, total_aux_loss, "trend_loss", loss, self.config.trend_loss_weight)

            loss = self._classification_loss(
                logits["mitigation_action"],
                targets.get("mitigation_action"),
                len(self.config.action_labels),
            )
            total_aux_loss = self._add_loss(losses, total_aux_loss, "action_loss", loss, self.config.action_loss_weight)

            loss = self._classification_loss(
                logits["lateral_mitigation_action"],
                targets.get("lateral_mitigation_action"),
                len(self.config.lateral_action_labels),
            )
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "lateral_action_loss",
                loss,
                self.config.lateral_action_loss_weight,
            )

            loss = self._regression_loss(
                predictions["recommended_ego_speed_mps"],
                targets.get("recommended_ego_speed_mps"),
            )
            total_aux_loss = self._add_loss(losses, total_aux_loss, "speed_loss", loss, self.config.speed_loss_weight)

            loss = self._regression_loss(
                predictions["suggested_deceleration_mps2"],
                targets.get("suggested_deceleration_mps2"),
            )
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "deceleration_loss",
                loss,
                self.config.deceleration_loss_weight,
            )

            if self.config.ttc_loss_mode == "finite_log":
                loss = self._vector_classification_loss(
                    logits["current_ttc_finite"],
                    targets.get("current_ttc_finite"),
                    2,
                )
                total_aux_loss = self._add_loss(
                    losses,
                    total_aux_loss,
                    "current_ttc_finite_loss",
                    loss,
                    self.config.ttc_finite_loss_weight,
                )
                loss = self._ttc_log_regression_loss(raw_current_ttc, targets.get("current_ttc_log"))
            else:
                loss = self._regression_loss(predictions["current_ttc"], targets.get("current_ttc"))
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "current_ttc_loss",
                loss,
                self.config.current_ttc_loss_weight,
            )

            loss = self._regression_loss(predictions["current_dtc"], targets.get("current_dtc"))
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "current_dtc_loss",
                loss,
                self.config.current_dtc_loss_weight,
            )

            if self.config.ttc_loss_mode == "finite_log":
                loss = self._ttc_log_regression_loss(raw_future_ttc, targets.get("future_ttc_log"))
            else:
                loss = self._regression_loss(predictions["future_ttc"], targets.get("future_ttc"))
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "future_ttc_loss",
                loss,
                self.config.future_ttc_loss_weight,
            )

            loss = self._regression_loss(predictions["future_dtc"], targets.get("future_dtc"))
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "future_dtc_loss",
                loss,
                self.config.future_dtc_loss_weight,
            )

            if self.config.ttc_loss_mode == "finite_log":
                loss = self._vector_classification_loss(
                    logits["future_ttc_finite"],
                    targets.get("future_ttc_finite"),
                    2,
                )
                total_aux_loss = self._add_loss(
                    losses,
                    total_aux_loss,
                    "future_ttc_finite_loss",
                    loss,
                    self.config.ttc_finite_loss_weight,
                )
                loss = self._ttc_log_regression_loss(raw_future_ttc_vector, targets.get("future_ttc_log_vector"))
            else:
                loss = self._regression_loss(predictions["future_ttc_vector"], targets.get("future_ttc_vector"))
            future_ttc_vector_weight = (
                self.config.future_ttc_loss_weight
                if self.config.future_ttc_vector_loss_weight < 0.0
                else self.config.future_ttc_vector_loss_weight
            )
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "future_ttc_vector_loss",
                loss,
                future_ttc_vector_weight,
            )

            loss = self._regression_loss(predictions["future_dtc_vector"], targets.get("future_dtc_vector"))
            future_dtc_vector_weight = (
                self.config.future_dtc_loss_weight
                if self.config.future_dtc_vector_loss_weight < 0.0
                else self.config.future_dtc_vector_loss_weight
            )
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "future_dtc_vector_loss",
                loss,
                future_dtc_vector_weight,
            )

            loss = self._regression_loss(
                predictions["target_lateral_offset_m"],
                targets.get("target_lateral_offset_m"),
            )
            total_aux_loss = self._add_loss(
                losses,
                total_aux_loss,
                "lateral_offset_loss",
                loss,
                self.config.lateral_offset_loss_weight,
            )

        return {
            "logits": logits,
            "predictions": predictions,
            "losses": losses,
            "total_aux_loss": total_aux_loss,
            "state_source": source,
        }

    def _select_task_states(self, hidden_states: Tensor, state_source: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if state_source == "recurrent_cot":
            if hidden_states.size(1) < 4:
                raise ValueError(
                    "state_source='recurrent_cot' expects at least 4 step states "
                    f"[z1,z2,z3,z4], got {hidden_states.size(1)}"
                )
            return hidden_states[:, 0, :], hidden_states[:, 1, :], hidden_states[:, 2, :], hidden_states[:, 3, :]

        if state_source != "risk_latents":
            raise ValueError(f"Unsupported risk_head_source: {state_source}")

        if hidden_states.size(1) >= 8:
            h_cur = hidden_states[:, 0:2, :].mean(dim=1)
            h_fut = hidden_states[:, 2:4, :].mean(dim=1)
            h_act = hidden_states[:, 4:6, :].mean(dim=1)
            h_trend = hidden_states[:, 6:8, :].mean(dim=1)
            return h_cur, h_fut, h_trend, h_act

        if hidden_states.size(1) >= 4:
            return hidden_states[:, 0, :], hidden_states[:, 1, :], hidden_states[:, 2, :], hidden_states[:, 3, :]

        raise ValueError(f"state_source='risk_latents' expects at least 4 states, got {hidden_states.size(1)}")

    def _classification_head(self, hidden_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _regression_head(self, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _vector_regression_head(self, hidden_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _vector_classification_head(self, hidden_dim: int, out_dim: int, num_classes: int) -> nn.Module:
        return _VectorClassificationHead(hidden_dim, out_dim, num_classes, self.config.dropout)

    def _classification_loss(self, logits: Tensor, target: Optional[Tensor], num_classes: int) -> Optional[Tensor]:
        if target is None:
            return None
        target = target.to(device=logits.device).long().view(-1)
        if target.numel() != logits.size(0):
            raise ValueError(f"Target batch size {target.numel()} does not match logits batch size {logits.size(0)}")

        valid = (target >= 0) & (target < num_classes) & (target != self.config.ignore_index)
        if not valid.any():
            return None
        return F.cross_entropy(logits[valid], target[valid])

    def _regression_loss(self, prediction: Tensor, target: Optional[Tensor]) -> Optional[Tensor]:
        if target is None:
            return None
        target = target.to(device=prediction.device, dtype=prediction.dtype).view(-1)
        if target.numel() != prediction.numel():
            raise ValueError(
                f"Target batch size {target.numel()} does not match prediction batch size {prediction.numel()}"
            )
        valid = torch.isfinite(target)
        if not valid.any():
            return None
        return F.smooth_l1_loss(prediction.view(-1)[valid], target[valid])

    def _vector_classification_loss(
        self,
        logits: Tensor,
        target: Optional[Tensor],
        num_classes: int,
    ) -> Optional[Tensor]:
        if target is None:
            return None
        target = target.to(device=logits.device).long().view(-1)
        logits = logits.view(-1, num_classes)
        if target.numel() != logits.size(0):
            raise ValueError(f"Target size {target.numel()} does not match logits size {logits.size(0)}")
        valid = (target >= 0) & (target < num_classes) & (target != self.config.ignore_index)
        if not valid.any():
            return None
        return F.cross_entropy(logits[valid], target[valid])

    def _ttc_prediction(self, raw_prediction: Tensor) -> Tensor:
        if self.config.ttc_loss_mode == "finite_log":
            return F.softplus(raw_prediction)
        return raw_prediction

    def _ttc_log_regression_loss(self, raw_prediction: Tensor, target: Optional[Tensor]) -> Optional[Tensor]:
        if target is None:
            return None
        target = target.to(device=raw_prediction.device, dtype=raw_prediction.dtype).view(-1)
        prediction = raw_prediction.view(-1)
        if target.numel() != prediction.numel():
            raise ValueError(
                f"Target batch size {target.numel()} does not match prediction batch size {prediction.numel()}"
            )
        valid = torch.isfinite(target)
        if not valid.any():
            return None
        log_scale = math.log1p(float(self.config.max_ttc_seconds))
        predicted_seconds = F.softplus(prediction[valid])
        predicted_log = torch.log1p(predicted_seconds) / log_scale
        return F.smooth_l1_loss(predicted_log, target[valid])

    def _add_loss(
        self,
        losses: Dict[str, Tensor],
        total_loss: Tensor,
        name: str,
        loss: Optional[Tensor],
        weight: float,
    ) -> Tensor:
        if loss is None or weight == 0.0:
            return total_loss
        weighted = loss * weight
        losses[name] = loss
        losses[f"weighted_{name}"] = weighted
        return total_loss + weighted


class _VectorClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim * num_classes),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.net(hidden_states).view(hidden_states.size(0), self.out_dim, self.num_classes)


class RiskCoTAuxiliaryDecoder(nn.Module):
    """Training-only step decoder for SIM-CoT-style latent supervision.

    In the intended Stage 4B/5 path, input states are recurrent latent CoT
    states ``[z1,z2,z3,z4]`` and each ``z_k`` conditions generation of the
    corresponding textual GT reasoning step ``s_k``. If more states are passed,
    they are pooled only as a compatibility fallback.
    """

    def __init__(self, config: RiskLatentConfig, vocab_size: int, pad_token_id: Optional[int] = None) -> None:
        super().__init__()
        if config.num_cot_steps <= 0:
            raise ValueError("num_cot_steps must be positive")

        self.config = config
        self.vocab_size = int(vocab_size)
        self.pad_token_id = pad_token_id
        hidden_dim = int(config.cot_decoder_hidden_dim)
        num_heads = int(config.cot_decoder_heads)
        if hidden_dim % num_heads != 0:
            raise ValueError(f"cot_decoder_hidden_dim={hidden_dim} must be divisible by cot_decoder_heads={num_heads}")

        self.latent_projector = nn.Sequential(
            nn.LayerNorm(config.llm_hidden_dim),
            nn.Linear(config.llm_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_embedding = nn.Embedding(
            self.vocab_size,
            hidden_dim,
            padding_idx=pad_token_id if pad_token_id is not None and pad_token_id >= 0 else None,
        )
        self.position_embedding = nn.Embedding(config.max_cot_step_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=config.cot_decoder_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=int(config.cot_decoder_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, self.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.pad_token_id is not None and 0 <= self.pad_token_id < self.vocab_size:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_token_id].zero_()

    def forward(
        self,
        latent_hidden_states: Tensor,
        cot_step_input_ids: Optional[Tensor] = None,
        cot_step_attention_mask: Optional[Tensor] = None,
        cot_step_labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        if cot_step_input_ids is None or cot_step_labels is None:
            return self._empty_output(latent_hidden_states)
        if latent_hidden_states.dim() != 3:
            raise ValueError(f"Expected latent_hidden_states [B, Z, D], got rank {latent_hidden_states.dim()}")
        if cot_step_input_ids.dim() != 3:
            raise ValueError(f"Expected cot_step_input_ids [B, S, L], got rank {cot_step_input_ids.dim()}")

        num_steps = int(self.config.num_cot_steps)
        batch_size, steps, step_length = cot_step_input_ids.shape
        if steps != num_steps:
            raise ValueError(f"Expected {num_steps} CoT steps, got {steps}")
        if step_length > int(self.config.max_cot_step_length):
            raise ValueError(
                f"CoT step length {step_length} exceeds max_cot_step_length={self.config.max_cot_step_length}"
            )

        device = latent_hidden_states.device
        cot_step_input_ids = cot_step_input_ids.to(device=device).long()
        cot_step_labels = cot_step_labels.to(device=device).long()
        if cot_step_attention_mask is None:
            cot_step_attention_mask = (cot_step_labels != self.config.ignore_index).long()
        else:
            cot_step_attention_mask = cot_step_attention_mask.to(device=device).long()

        valid_tokens = cot_step_labels.ne(self.config.ignore_index).sum()
        if valid_tokens.item() == 0:
            return self._empty_output(latent_hidden_states)

        latent_groups = self._pool_latent_groups(latent_hidden_states)
        decoder_dtype = next(self.latent_projector.parameters()).dtype
        prefixes = self.latent_projector(latent_groups.to(dtype=decoder_dtype).reshape(batch_size * steps, -1))
        prefixes = prefixes.unsqueeze(1)

        flat_input_ids = cot_step_input_ids.reshape(batch_size * steps, step_length)
        flat_attention_mask = cot_step_attention_mask.reshape(batch_size * steps, step_length)
        flat_labels = cot_step_labels.reshape(batch_size * steps, step_length)

        if step_length > 1:
            previous_token_ids = flat_input_ids[:, :-1]
            previous_attention = flat_attention_mask[:, :-1]
            previous_embeds = self.token_embedding(previous_token_ids)
            decoder_inputs = torch.cat([prefixes, previous_embeds], dim=1)
            input_attention = torch.cat(
                [
                    torch.ones(prefixes.size(0), 1, dtype=previous_attention.dtype, device=device),
                    previous_attention,
                ],
                dim=1,
            )
        else:
            decoder_inputs = prefixes
            input_attention = torch.ones(prefixes.size(0), 1, dtype=torch.long, device=device)

        positions = torch.arange(step_length, device=device)
        decoder_inputs = decoder_inputs + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(step_length, step_length, dtype=torch.bool, device=device),
            diagonal=1,
        )
        key_padding_mask = input_attention == 0
        decoded = self.decoder(decoder_inputs, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        logits = self.output_head(self.output_norm(decoded))

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            flat_labels.reshape(-1),
            ignore_index=self.config.ignore_index,
            reduction="sum",
        )
        loss = loss / valid_tokens.clamp_min(1).to(dtype=loss.dtype)
        return {
            "loss": loss,
            "valid_tokens": valid_tokens.detach(),
        }

    def _pool_latent_groups(self, latent_hidden_states: Tensor) -> Tensor:
        num_steps = int(self.config.num_cot_steps)
        num_latents = int(latent_hidden_states.size(1))
        if num_latents < num_steps:
            raise ValueError(f"Need at least {num_steps} conditioning states for CoT decoding, got {num_latents}")
        if num_latents == num_steps:
            return latent_hidden_states
        group_size = max(1, num_latents // num_steps)
        groups = []
        for step_index in range(num_steps):
            start = step_index * group_size
            end = num_latents if step_index == num_steps - 1 else min(num_latents, start + group_size)
            groups.append(latent_hidden_states[:, start:end, :].mean(dim=1))
        return torch.stack(groups, dim=1)

    def _empty_output(self, reference: Tensor) -> Dict[str, Any]:
        return {
            "loss": reference.new_zeros(()),
            "valid_tokens": reference.new_zeros((), dtype=torch.long),
        }


class RiskLatentMLLM(nn.Module):
    """Wrap a Q-Former and causal LLM with LLM-context risk latent tokens."""

    def __init__(
        self,
        qformer: nn.Module,
        llm: nn.Module,
        config: RiskLatentConfig,
        aux_heads: Optional[RiskAuxiliaryHeads] = None,
    ) -> None:
        super().__init__()
        self.qformer = qformer
        self.llm = llm
        self.config = config
        self.risk_latent_tokens = nn.Parameter(torch.empty(1, config.num_risk_latents, config.llm_hidden_dim))
        self.aux_heads = aux_heads if aux_heads is not None else RiskAuxiliaryHeads(config)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.risk_latent_tokens, mean=0.0, std=self.config.latent_init_std)

    def forward(
        self,
        visual_embeds: Optional[Tensor] = None,
        visual_attention_mask: Optional[Tensor] = None,
        question_input_ids: Optional[Tensor] = None,
        question_attention_mask: Optional[Tensor] = None,
        answer_input_ids: Optional[Tensor] = None,
        answer_attention_mask: Optional[Tensor] = None,
        answer_labels: Optional[Tensor] = None,
        question_embeds: Optional[Tensor] = None,
        answer_embeds: Optional[Tensor] = None,
        fused_visual_tokens: Optional[Tensor] = None,
        risk_targets: Optional[Dict[str, Tensor]] = None,
        **llm_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run JSON loss and auxiliary risk latent losses.

        Provide either token ids or precomputed embeddings for question/answer.
        Provide either raw visual embeddings for the Q-Former or precomputed
        fused visual tokens. All risk latents are inserted after the visual
        tokens and before answer tokens.
        """

        if fused_visual_tokens is None:
            if visual_embeds is None:
                raise ValueError("Either visual_embeds or fused_visual_tokens must be provided")
            fused_visual_tokens = self.qformer(visual_embeds, visual_attention_mask=visual_attention_mask)

        question_embeds = self._resolve_text_embeds(question_embeds, question_input_ids, "question")
        answer_embeds = self._resolve_text_embeds(answer_embeds, answer_input_ids, "answer")
        if question_embeds is None:
            raise ValueError("question_input_ids or question_embeds must be provided")

        batch_size = fused_visual_tokens.size(0)
        device = question_embeds.device
        dtype = question_embeds.dtype
        fused_visual_tokens = fused_visual_tokens.to(device=device, dtype=dtype)
        if answer_embeds is not None:
            answer_embeds = answer_embeds.to(device=device, dtype=dtype)
        risk_latents = self.risk_latent_tokens.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

        input_parts = [question_embeds, fused_visual_tokens, risk_latents]
        if answer_embeds is not None:
            input_parts.append(answer_embeds)
        inputs_embeds = torch.cat(input_parts, dim=1)

        question_mask = self._resolve_attention_mask(question_attention_mask, question_embeds, device)
        visual_mask = torch.ones(batch_size, fused_visual_tokens.size(1), dtype=question_mask.dtype, device=device)
        latent_mask = torch.ones(batch_size, risk_latents.size(1), dtype=question_mask.dtype, device=device)
        attention_parts = [question_mask, visual_mask, latent_mask]
        if answer_embeds is not None:
            answer_mask = self._resolve_attention_mask(answer_attention_mask, answer_embeds, device)
            attention_parts.append(answer_mask)
        else:
            answer_mask = None
        attention_mask = torch.cat(attention_parts, dim=1)

        labels = self._build_labels(
            question_length=question_embeds.size(1),
            visual_length=fused_visual_tokens.size(1),
            latent_length=risk_latents.size(1),
            answer_input_ids=answer_input_ids,
            answer_labels=answer_labels,
            answer_attention_mask=answer_mask,
            device=device,
        )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **llm_kwargs,
        )

        latent_start = question_embeds.size(1) + fused_visual_tokens.size(1)
        latent_end = latent_start + risk_latents.size(1)
        hidden_states = llm_outputs.hidden_states[-1]
        latent_hidden_states = hidden_states[:, latent_start:latent_end, :]
        aux_outputs = self.aux_heads(
            latent_hidden_states,
            risk_targets,
            state_source="risk_latents",
        )

        json_loss = getattr(llm_outputs, "loss", None)
        answer_loss = json_loss if json_loss is not None else latent_hidden_states.new_zeros(())
        aux_loss = aux_outputs["total_aux_loss"]
        weighted_answer_loss = answer_loss * self.config.answer_loss_weight
        weighted_aux_loss = aux_loss * self.config.risk_loss_weight
        total_loss = weighted_answer_loss + weighted_aux_loss

        return {
            "loss": total_loss,
            "json_loss": json_loss,
            "answer_loss": answer_loss,
            "weighted_answer_loss": weighted_answer_loss,
            "aux_loss": aux_loss,
            "weighted_aux_loss": weighted_aux_loss,
            "aux_outputs": aux_outputs,
            "latent_hidden_states": latent_hidden_states,
            "latent_range": (latent_start, latent_end),
            "fused_visual_tokens": fused_visual_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
            "llm_outputs": llm_outputs,
        }

    def _resolve_text_embeds(
        self,
        embeds: Optional[Tensor],
        input_ids: Optional[Tensor],
        name: str,
    ) -> Optional[Tensor]:
        if embeds is not None:
            return embeds
        if input_ids is None:
            if name == "answer":
                return None
            raise ValueError(f"{name}_input_ids or {name}_embeds must be provided")
        embedding_layer = self.llm.get_input_embeddings()
        return embedding_layer(input_ids)

    def _resolve_attention_mask(self, mask: Optional[Tensor], embeds: Tensor, device: torch.device) -> Tensor:
        if mask is not None:
            return mask.to(device=device)
        return torch.ones(embeds.size(0), embeds.size(1), dtype=torch.long, device=device)

    def _build_labels(
        self,
        question_length: int,
        visual_length: int,
        latent_length: int,
        answer_input_ids: Optional[Tensor],
        answer_labels: Optional[Tensor],
        answer_attention_mask: Optional[Tensor],
        device: torch.device,
    ) -> Optional[Tensor]:
        if answer_labels is None and answer_input_ids is None:
            return None

        labels = answer_labels if answer_labels is not None else answer_input_ids
        if labels is None:
            return None
        labels = labels.to(device=device).long()
        if answer_attention_mask is not None:
            labels = labels.masked_fill(answer_attention_mask.to(device=device) == 0, self.config.ignore_index)

        prefix_length = question_length + visual_length + latent_length
        prefix = torch.full(
            (labels.size(0), prefix_length),
            self.config.ignore_index,
            dtype=labels.dtype,
            device=device,
        )
        return torch.cat([prefix, labels], dim=1)


def build_risk_target_tensors(samples: Sequence[Dict[str, Any]], config: RiskLatentConfig) -> Dict[str, Tensor]:
    """Convert raw dataset risk target dictionaries into batched tensors."""

    risk_targets = [sample.get("risk_targets", sample) for sample in samples]
    return {
        "current_risk_score": _encode_numeric_class_targets(
            [target.get("current_risk_score") for target in risk_targets],
            config.ignore_index,
        ),
        "future_worst_risk_score": _encode_numeric_class_targets(
            [target.get("future_worst_risk_score") for target in risk_targets],
            config.ignore_index,
        ),
        "risk_trend": encode_string_targets(
            [target.get("risk_trend") for target in risk_targets],
            config.trend_labels,
            config.ignore_index,
        ),
        "mitigation_action": encode_string_targets(
            [target.get("mitigation_action") for target in risk_targets],
            config.action_labels,
            config.ignore_index,
        ),
        "lateral_mitigation_action": encode_string_targets(
            [target.get("lateral_mitigation_action") for target in risk_targets],
            config.lateral_action_labels,
            config.ignore_index,
        ),
        "recommended_ego_speed_mps": _encode_regression_targets(
            [target.get("recommended_ego_speed_mps") for target in risk_targets],
            min_value=0.0,
            max_value=config.max_speed_mps,
        ),
        "suggested_deceleration_mps2": _encode_regression_targets(
            [target.get("suggested_deceleration_mps2") for target in risk_targets],
            min_value=-config.max_abs_acceleration_mps2,
            max_value=config.max_abs_acceleration_mps2,
        ),
        "target_lateral_offset_m": _encode_regression_targets(
            [target.get("target_lateral_offset_m") for target in risk_targets],
            min_value=-config.max_abs_dtc_meters,
            max_value=config.max_abs_dtc_meters,
        ),
        "current_ttc": _encode_vector_regression_targets(
            risk_targets,
            ("current_ttc_longitudinal", "current_ttc_lateral"),
            min_value=0.0,
            max_value=config.max_ttc_seconds,
        ),
        "current_ttc_finite": _encode_vector_finite_targets(
            risk_targets,
            ("current_ttc_longitudinal", "current_ttc_lateral"),
            config.ignore_index,
        ),
        "current_ttc_log": _encode_vector_log_ttc_targets(
            risk_targets,
            ("current_ttc_longitudinal", "current_ttc_lateral"),
            max_value=config.max_ttc_seconds,
        ),
        "current_dtc": _encode_vector_regression_targets(
            risk_targets,
            ("current_dtc_longitudinal", "current_dtc_lateral"),
            min_value=-config.max_abs_dtc_meters,
            max_value=config.max_abs_dtc_meters,
        ),
        "future_ttc": _encode_regression_targets(
            [target.get("future_ttc") for target in risk_targets],
            min_value=0.0,
            max_value=config.max_ttc_seconds,
        ),
        "future_ttc_log": _encode_log_ttc_targets(
            [target.get("future_ttc") for target in risk_targets],
            max_value=config.max_ttc_seconds,
        ),
        "future_dtc": _encode_regression_targets(
            [target.get("future_dtc") for target in risk_targets],
            min_value=-config.max_abs_dtc_meters,
            max_value=config.max_abs_dtc_meters,
        ),
        "future_ttc_vector": _encode_vector_regression_targets(
            risk_targets,
            ("future_ttc", "future_ttc_lateral"),
            min_value=0.0,
            max_value=config.max_ttc_seconds,
        ),
        "future_ttc_finite": _encode_vector_finite_targets(
            risk_targets,
            ("future_ttc", "future_ttc_lateral"),
            config.ignore_index,
        ),
        "future_ttc_log_vector": _encode_vector_log_ttc_targets(
            risk_targets,
            ("future_ttc", "future_ttc_lateral"),
            max_value=config.max_ttc_seconds,
        ),
        "future_dtc_vector": _encode_vector_regression_targets(
            risk_targets,
            ("future_dtc", "future_dtc_lateral"),
            min_value=-config.max_abs_dtc_meters,
            max_value=config.max_abs_dtc_meters,
        ),
    }


def encode_string_targets(values: Sequence[Any], label_space: Sequence[str], ignore_index: int = -100) -> Tensor:
    """Map string labels to integer ids for auxiliary classification heads."""

    label_to_id = {label: index for index, label in enumerate(label_space)}
    encoded = []
    for value in values:
        if isinstance(value, str):
            encoded.append(label_to_id.get(value, ignore_index))
        elif value is None:
            encoded.append(ignore_index)
        else:
            encoded.append(int(value))
    return torch.tensor(encoded, dtype=torch.long)


def _encode_numeric_class_targets(values: Sequence[Any], ignore_index: int) -> Tensor:
    encoded = []
    for value in values:
        number = _to_number(value)
        encoded.append(ignore_index if number is None else int(number))
    return torch.tensor(encoded, dtype=torch.long)


def _encode_regression_targets(
    values: Sequence[Any],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Tensor:
    encoded = []
    for value in values:
        number = _to_regression_number(value)
        if number is None:
            encoded.append(float("nan"))
            continue
        if min_value is not None:
            number = max(float(min_value), number)
        if max_value is not None:
            number = min(float(max_value), number)
        encoded.append(float(number))
    return torch.tensor(encoded, dtype=torch.float32)


def _encode_vector_regression_targets(
    targets: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Tensor:
    rows = []
    for target in targets:
        values = []
        for key in keys:
            number = _to_regression_number(target.get(key))
            if number is None:
                values.append(float("nan"))
                continue
            if min_value is not None:
                number = max(float(min_value), number)
            if max_value is not None:
                number = min(float(max_value), number)
            values.append(float(number))
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32)


def _encode_log_ttc_targets(
    values: Sequence[Any],
    max_value: float,
) -> Tensor:
    log_scale = math.log1p(float(max_value))
    encoded = []
    for value in values:
        number = _to_regression_number(value)
        if number is None or not math.isfinite(number):
            encoded.append(float("nan"))
            continue
        number = min(float(max_value), max(0.0, number))
        encoded.append(math.log1p(number) / log_scale)
    return torch.tensor(encoded, dtype=torch.float32)


def _encode_vector_log_ttc_targets(
    targets: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    max_value: float,
) -> Tensor:
    rows = []
    log_scale = math.log1p(float(max_value))
    for target in targets:
        values = []
        for key in keys:
            number = _to_regression_number(target.get(key))
            if number is None or not math.isfinite(number):
                values.append(float("nan"))
                continue
            number = min(float(max_value), max(0.0, number))
            values.append(math.log1p(number) / log_scale)
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32)


def _encode_vector_finite_targets(
    targets: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    ignore_index: int,
) -> Tensor:
    rows = []
    for target in targets:
        values = []
        for key in keys:
            number = _to_regression_number(target.get(key))
            if number is None:
                values.append(ignore_index)
            else:
                values.append(1 if math.isfinite(number) else 0)
        rows.append(values)
    return torch.tensor(rows, dtype=torch.long)


def _to_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _to_regression_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) or math.isinf(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) or math.isinf(number) else None
    return None
