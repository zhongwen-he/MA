#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Qwen3-VL multi-camera adapter for risk-latent training.

Qwen3-VL keeps responsibility for per-video encoding and language modeling.
This wrapper extracts one feature sequence per camera video with
``get_video_features()``, groups those sequences back into samples, and lets the
multi-view Q-Former perform cross-camera fusion before inserting the resulting
visual tokens into the Qwen3-VL language context.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .multiview_qformer import MultiViewQFormerAdapter, MultiViewQFormerConfig
from .risk_latent_mllm import RiskAuxiliaryHeads, RiskCoTAuxiliaryDecoder, RiskLatentConfig


@dataclass
class Qwen3VLMultiCameraConfig:
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    local_files_only: bool = False
    device_map: Optional[str] = "auto"
    dtype: str = "auto"
    freeze_video_encoder: bool = True
    train_llm: str = "lora_only"


class Qwen3VLMultiCameraRiskMLLM(nn.Module):
    """Use Qwen3-VL's video encoder with a multi-camera Q-Former bottleneck."""

    def __init__(
        self,
        qwen_model: nn.Module,
        qformer: MultiViewQFormerAdapter,
        risk_config: RiskLatentConfig,
        aux_heads: Optional[RiskAuxiliaryHeads] = None,
    ) -> None:
        super().__init__()
        self.qwen_model = qwen_model
        self.qformer = qformer
        self.risk_config = risk_config
        self.risk_latent_tokens = nn.Parameter(
            torch.empty(1, risk_config.num_risk_latents, risk_config.llm_hidden_dim)
        )
        self.aux_heads = aux_heads if aux_heads is not None else RiskAuxiliaryHeads(risk_config)
        self.cot_decoder = self._build_cot_decoder() if risk_config.enable_latent_cot else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.risk_latent_tokens, mean=0.0, std=self.risk_config.latent_init_std)

    @property
    def llm(self) -> nn.Module:
        """Compatibility alias for code that expects ``model.llm``."""

        return self.qwen_model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Forward Hugging Face Trainer gradient-checkpointing calls.

        ``Trainer`` calls this method on the top-level model.  The expensive
        transformer blocks live inside the wrapped Qwen model, while the BLIP-2
        Q-Former is also a Transformers module in the Stage-2 path.  Forwarding
        the call keeps the wrapper compatible with ``TrainingArguments`` without
        changing Stage-1 artifacts or checkpoint formats.
        """

        if hasattr(self.qwen_model, "enable_input_require_grads"):
            self.qwen_model.enable_input_require_grads()
        if hasattr(self.qwen_model, "config"):
            self.qwen_model.config.use_cache = False
        if hasattr(self.qwen_model, "gradient_checkpointing_enable"):
            self.qwen_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
            self._disable_frozen_visual_gradient_checkpointing()
        if hasattr(self.qformer, "gradient_checkpointing_enable"):
            self.qformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self) -> None:
        """Forward Hugging Face Trainer gradient-checkpointing disable calls."""

        if hasattr(self.qwen_model, "gradient_checkpointing_disable"):
            self.qwen_model.gradient_checkpointing_disable()
        if hasattr(self.qformer, "gradient_checkpointing_disable"):
            self.qformer.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Expose checkpointing status for Trainer compatibility."""

        return bool(getattr(self.qwen_model, "is_gradient_checkpointing", False)) or bool(
            getattr(self.qformer, "is_gradient_checkpointing", False)
        )

    def forward(
        self,
        pixel_values_videos: Optional[Tensor] = None,
        video_grid_thw: Optional[Tensor] = None,
        camera_counts: Optional[Tensor] = None,
        question_input_ids: Optional[Tensor] = None,
        question_attention_mask: Optional[Tensor] = None,
        answer_input_ids: Optional[Tensor] = None,
        answer_attention_mask: Optional[Tensor] = None,
        answer_labels: Optional[Tensor] = None,
        question_embeds: Optional[Tensor] = None,
        answer_embeds: Optional[Tensor] = None,
        fused_visual_tokens: Optional[Tensor] = None,
        risk_targets: Optional[Dict[str, Tensor]] = None,
        cot_step_input_ids: Optional[Tensor] = None,
        cot_step_attention_mask: Optional[Tensor] = None,
        cot_step_labels: Optional[Tensor] = None,
        **qwen_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run language-model JSON loss plus risk latent auxiliary losses.

        Provide either precomputed ``fused_visual_tokens`` or raw Qwen3-VL video
        processor outputs. ``camera_counts`` defines how the flattened camera
        videos are grouped back into each batch sample.
        """

        qwen_kwargs.pop("sample_ids", None)
        qwen_kwargs.pop("scenes", None)

        if fused_visual_tokens is None:
            if pixel_values_videos is None or video_grid_thw is None or camera_counts is None:
                raise ValueError(
                    "Provide fused_visual_tokens or pixel_values_videos, video_grid_thw, and camera_counts"
                )
            visual_tokens, visual_mask = self.encode_multicamera_videos(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                camera_counts=camera_counts,
                **qwen_kwargs.pop("video_encoder_kwargs", {}),
            )
            fused_visual_tokens = self.qformer(visual_tokens, visual_attention_mask=visual_mask)

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
            resolved_answer_mask = self._resolve_attention_mask(answer_attention_mask, answer_embeds, device)
            attention_parts.append(resolved_answer_mask)
        else:
            resolved_answer_mask = None
        attention_mask = torch.cat(attention_parts, dim=1)

        labels = self._build_labels(
            question_length=question_embeds.size(1),
            visual_length=fused_visual_tokens.size(1),
            latent_length=risk_latents.size(1),
            answer_input_ids=answer_input_ids,
            answer_labels=answer_labels,
            answer_attention_mask=resolved_answer_mask,
            device=device,
        )

        needs_latent_states = self._needs_latent_states(
            risk_targets=risk_targets,
            cot_step_labels=cot_step_labels,
        )
        qwen_outputs = self.qwen_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=needs_latent_states,
            return_dict=True,
            **qwen_kwargs,
        )

        latent_start = question_embeds.size(1) + fused_visual_tokens.size(1)
        latent_end = latent_start + risk_latents.size(1)
        if needs_latent_states:
            hidden_states = qwen_outputs.hidden_states[-1]
            latent_hidden_states = hidden_states[:, latent_start:latent_end, :]
            reasoning_states = self._run_recurrent_latent_cot(
                question_embeds=question_embeds,
                fused_visual_tokens=fused_visual_tokens,
                latent_hidden_states=latent_hidden_states,
                question_attention_mask=question_mask,
                visual_attention_mask=visual_mask,
                qwen_kwargs=qwen_kwargs,
            )
            risk_head_states = reasoning_states if self.risk_config.risk_head_source == "recurrent_cot" else latent_hidden_states
            aux_outputs = self.aux_heads(
                risk_head_states,
                risk_targets,
                state_source=self.risk_config.risk_head_source,
            )
            cot_outputs = self._run_cot_decoder(
                latent_hidden_states=reasoning_states,
                cot_step_input_ids=cot_step_input_ids,
                cot_step_attention_mask=cot_step_attention_mask,
                cot_step_labels=cot_step_labels,
            )
        else:
            latent_hidden_states = risk_latents.new_zeros(risk_latents.shape)
            reasoning_states = latent_hidden_states
            risk_head_states = latent_hidden_states
            zero_loss = fused_visual_tokens.new_zeros(())
            aux_outputs = {
                "total_aux_loss": zero_loss,
                "losses": {},
            }
            cot_outputs = {
                "loss": zero_loss,
                "valid_tokens": torch.zeros((), dtype=torch.long, device=fused_visual_tokens.device),
            }

        json_loss = getattr(qwen_outputs, "loss", None)
        answer_loss = json_loss if json_loss is not None else fused_visual_tokens.new_zeros(())
        aux_loss = aux_outputs["total_aux_loss"]
        weighted_answer_loss = answer_loss * self.risk_config.answer_loss_weight
        weighted_aux_loss = aux_loss * self.risk_config.risk_loss_weight
        weighted_cot_loss = cot_outputs["loss"] * self.risk_config.cot_loss_weight
        total_loss = weighted_answer_loss + weighted_aux_loss + weighted_cot_loss

        return {
            "loss": total_loss,
            "json_loss": json_loss,
            "answer_loss": answer_loss,
            "weighted_answer_loss": weighted_answer_loss,
            "aux_loss": aux_loss,
            "weighted_aux_loss": weighted_aux_loss,
            "cot_loss": cot_outputs["loss"],
            "weighted_cot_loss": weighted_cot_loss,
            "cot_outputs": cot_outputs,
            "aux_outputs": aux_outputs,
            "latent_hidden_states": latent_hidden_states,
            "reasoning_states": reasoning_states,
            "risk_head_states": risk_head_states,
            "latent_range": (latent_start, latent_end),
            "fused_visual_tokens": fused_visual_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
            "qwen_outputs": qwen_outputs,
        }

    def _needs_latent_states(
        self,
        risk_targets: Optional[Dict[str, Tensor]],
        cot_step_labels: Optional[Tensor],
    ) -> bool:
        """Return whether this forward pass needs Qwen hidden states.

        Stage 2 Q-Former answer alignment only uses the Qwen CE loss.  In that
        case returning every Qwen hidden state is unnecessary and very expensive
        on 8GB GPUs.  Later risk/CoT stages still request hidden states because
        their heads are explicitly supervised from the risk latent positions.
        """

        return bool(
            self.risk_config.risk_loss_weight > 0
            or self.risk_config.cot_loss_weight > 0
            or self.risk_config.enable_recurrent_latent_cot
            or risk_targets is not None
            or cot_step_labels is not None
        )

    def _build_cot_decoder(self) -> RiskCoTAuxiliaryDecoder:
        embeddings = self.qwen_model.get_input_embeddings()
        vocab_size = int(getattr(embeddings, "num_embeddings"))
        model_config = getattr(self.qwen_model, "config", None)
        pad_token_id = getattr(model_config, "pad_token_id", None)
        return RiskCoTAuxiliaryDecoder(self.risk_config, vocab_size=vocab_size, pad_token_id=pad_token_id)

    def _run_recurrent_latent_cot(
        self,
        question_embeds: Tensor,
        fused_visual_tokens: Tensor,
        latent_hidden_states: Tensor,
        question_attention_mask: Tensor,
        visual_attention_mask: Tensor,
        qwen_kwargs: Dict[str, Any],
    ) -> Tensor:
        """Generate SIM-CoT-style implicit reasoning states z1..zK.

        The seed context is ``[question, visual, H_R]`` where ``H_R`` are the
        hidden states at the inserted risk-latent positions. At step k, the last
        hidden state of the Qwen forward pass is used as z_k and appended as a
        continuous token before producing z_{k+1}.
        """

        if not self.risk_config.enable_recurrent_latent_cot:
            if self.risk_config.risk_head_source == "recurrent_cot":
                raise ValueError(
                    "risk_head_source='recurrent_cot' requires enable_recurrent_latent_cot=True"
                )
            return latent_hidden_states

        num_steps = int(self.risk_config.num_cot_steps)
        if num_steps <= 0:
            raise ValueError("num_cot_steps must be positive when recurrent latent CoT is enabled")

        batch_size = latent_hidden_states.size(0)
        device = latent_hidden_states.device
        mask_dtype = question_attention_mask.dtype
        latent_attention_mask = torch.ones(
            batch_size,
            latent_hidden_states.size(1),
            dtype=mask_dtype,
            device=device,
        )
        context_embeds = torch.cat(
            [
                question_embeds.to(device=device, dtype=latent_hidden_states.dtype),
                fused_visual_tokens.to(device=device, dtype=latent_hidden_states.dtype),
                latent_hidden_states,
            ],
            dim=1,
        )
        context_attention_mask = torch.cat(
            [
                question_attention_mask.to(device=device),
                visual_attention_mask.to(device=device),
                latent_attention_mask,
            ],
            dim=1,
        )

        recurrence_kwargs = {
            key: value
            for key, value in qwen_kwargs.items()
            if key not in {"labels", "output_hidden_states", "return_dict", "inputs_embeds", "attention_mask"}
        }
        recurrence_kwargs["use_cache"] = False

        reasoning_states = []
        for _ in range(num_steps):
            outputs = self.qwen_model(
                inputs_embeds=context_embeds,
                attention_mask=context_attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **recurrence_kwargs,
            )
            next_state = outputs.hidden_states[-1][:, -1, :]
            reasoning_states.append(next_state)
            context_embeds = torch.cat([context_embeds, next_state.unsqueeze(1)], dim=1)
            context_attention_mask = torch.cat(
                [
                    context_attention_mask,
                    torch.ones(batch_size, 1, dtype=mask_dtype, device=device),
                ],
                dim=1,
            )

        return torch.stack(reasoning_states, dim=1)

    def _run_cot_decoder(
        self,
        latent_hidden_states: Tensor,
        cot_step_input_ids: Optional[Tensor],
        cot_step_attention_mask: Optional[Tensor],
        cot_step_labels: Optional[Tensor],
    ) -> Dict[str, Any]:
        if self.cot_decoder is None:
            return {
                "loss": latent_hidden_states.new_zeros(()),
                "valid_tokens": latent_hidden_states.new_zeros((), dtype=torch.long),
            }
        return self.cot_decoder(
            latent_hidden_states=latent_hidden_states,
            cot_step_input_ids=cot_step_input_ids,
            cot_step_attention_mask=cot_step_attention_mask,
            cot_step_labels=cot_step_labels,
        )

    def encode_multicamera_videos(
        self,
        pixel_values_videos: Tensor,
        video_grid_thw: Tensor,
        camera_counts: Tensor,
        **video_encoder_kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Return padded per-sample camera-video features and attention mask.

        Returns:
            visual_tokens: ``[B, Vmax, Nmax, D]``
            visual_attention_mask: ``[B, Vmax, Nmax]`` with 1 for valid tokens.
        """

        video_features = self._get_qwen_video_features(pixel_values_videos, video_grid_thw, **video_encoder_kwargs)
        counts = [int(value) for value in camera_counts.detach().cpu().tolist()]
        if sum(counts) != len(video_features):
            raise ValueError(f"camera_counts sum {sum(counts)} does not match {len(video_features)} encoded videos")
        if not video_features:
            raise ValueError("No video features were produced")

        batch_size = len(counts)
        max_views = max(counts)
        max_tokens = max(int(features.size(0)) for features in video_features)
        hidden_dim = int(video_features[0].size(-1))
        device = video_features[0].device
        dtype = video_features[0].dtype

        visual_tokens = torch.zeros(batch_size, max_views, max_tokens, hidden_dim, device=device, dtype=dtype)
        visual_attention_mask = torch.zeros(batch_size, max_views, max_tokens, device=device, dtype=torch.long)

        offset = 0
        for batch_index, count in enumerate(counts):
            for view_index in range(count):
                features = video_features[offset + view_index]
                length = int(features.size(0))
                visual_tokens[batch_index, view_index, :length, :] = features
                visual_attention_mask[batch_index, view_index, :length] = 1
            offset += count

        return visual_tokens, visual_attention_mask

    def _get_qwen_video_features(
        self,
        pixel_values_videos: Tensor,
        video_grid_thw: Tensor,
        **video_encoder_kwargs: Any,
    ) -> Sequence[Tensor]:
        visual = get_qwen_visual_module(self.qwen_model)
        visual_is_frozen = not any(parameter.requires_grad for parameter in visual.parameters())
        with torch.set_grad_enabled(not visual_is_frozen):
            outputs = self.qwen_model.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                **video_encoder_kwargs,
            )
        features = getattr(outputs, "pooler_output", None)
        if features is None and isinstance(outputs, (tuple, list)):
            features = outputs[0]
        if isinstance(features, Tensor):
            if video_grid_thw is None:
                raise ValueError("Cannot split flat video features without video_grid_thw")
            merge_size = int(getattr(getattr(self.qwen_model, "visual", None), "spatial_merge_size", 1))
            split_sizes = (video_grid_thw.prod(-1) // (merge_size * merge_size)).detach().cpu().tolist()
            features = torch.split(features, [int(size) for size in split_sizes])
        if not isinstance(features, (tuple, list)):
            raise TypeError("Qwen3-VL get_video_features() did not return a sequence of per-video features")
        return features

    def _disable_frozen_visual_gradient_checkpointing(self) -> None:
        """Do not checkpoint frozen Qwen visual blocks during Stage-2 training.

        Stage 2 only trains the Q-Former/projector.  The Qwen visual encoder is
        used as a frozen feature extractor, so recomputation/checkpointing inside
        that tower only adds memory pressure and can trigger CUDA OOM on 8GB
        GPUs.  The Qwen language model may still use gradient checkpointing
        because answer CE must backpropagate to the inserted visual tokens.
        """

        try:
            visual = get_qwen_visual_module(self.qwen_model)
        except ValueError:
            return
        if any(parameter.requires_grad for parameter in visual.parameters()):
            return
        if hasattr(visual, "gradient_checkpointing_disable"):
            visual.gradient_checkpointing_disable()
        for module in visual.modules():
            if hasattr(module, "gradient_checkpointing"):
                try:
                    module.gradient_checkpointing = False
                except Exception:
                    pass

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
        return self.qwen_model.get_input_embeddings()(input_ids)

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
            labels = labels.masked_fill(answer_attention_mask.to(device=device) == 0, self.risk_config.ignore_index)

        prefix_length = question_length + visual_length + latent_length
        prefix = torch.full(
            (labels.size(0), prefix_length),
            self.risk_config.ignore_index,
            dtype=labels.dtype,
            device=device,
        )
        return torch.cat([prefix, labels], dim=1)


def infer_qwen_hidden_size(qwen_model: nn.Module) -> int:
    """Infer the text hidden size used by Qwen input embeddings."""

    embeddings = qwen_model.get_input_embeddings()
    if not hasattr(embeddings, "embedding_dim"):
        raise ValueError("Could not infer Qwen hidden size from input embeddings")
    return int(embeddings.embedding_dim)


def infer_qwen_visual_feature_size(qwen_model: nn.Module) -> int:
    """Infer the visual feature size emitted toward the Qwen language model."""

    config = getattr(qwen_model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None and isinstance(config, dict):
        vision_config = config.get("vision_config")

    if vision_config is not None:
        if isinstance(vision_config, dict):
            for key in ("out_hidden_size", "hidden_size"):
                value = vision_config.get(key)
                if value is not None:
                    return int(value)
        else:
            for key in ("out_hidden_size", "hidden_size"):
                value = getattr(vision_config, key, None)
                if value is not None:
                    return int(value)

    return infer_qwen_hidden_size(qwen_model)


def build_qwen3_vl_multicamera_model(
    qwen_model: nn.Module,
    qformer_config: Optional[MultiViewQFormerConfig] = None,
    risk_config: Optional[RiskLatentConfig] = None,
) -> Qwen3VLMultiCameraRiskMLLM:
    """Build the wrapper around an already loaded Qwen3-VL model."""

    llm_hidden_size = infer_qwen_hidden_size(qwen_model)
    visual_feature_size = infer_qwen_visual_feature_size(qwen_model)
    if qformer_config is None:
        qformer_config = MultiViewQFormerConfig(
            visual_input_dim=visual_feature_size,
            llm_hidden_dim=llm_hidden_size,
        )
    else:
        qformer_config.visual_input_dim = visual_feature_size
        qformer_config.llm_hidden_dim = llm_hidden_size

    if risk_config is None:
        risk_config = RiskLatentConfig(llm_hidden_dim=llm_hidden_size)
    else:
        risk_config.llm_hidden_dim = llm_hidden_size

    qformer = MultiViewQFormerAdapter(qformer_config)
    return Qwen3VLMultiCameraRiskMLLM(qwen_model=qwen_model, qformer=qformer, risk_config=risk_config)


def get_qwen_visual_module(qwen_model: nn.Module) -> nn.Module:
    """Return Qwen3-VL visual module, also when wrapped by PEFT."""

    candidates = [qwen_model]
    for attr in ("base_model", "model"):
        wrapped = getattr(qwen_model, attr, None)
        if wrapped is not None:
            candidates.append(wrapped)
            inner = getattr(wrapped, "model", None)
            if inner is not None:
                candidates.append(inner)

    for candidate in candidates:
        visual = getattr(candidate, "visual", None)
        if visual is not None:
            return visual

    for _, module in qwen_model.named_modules():
        visual = getattr(module, "visual", None)
        if visual is not None:
            return visual
    raise ValueError("Qwen model does not expose a .visual module to freeze")


def freeze_qwen_video_encoder(qwen_model: nn.Module) -> None:
    """Freeze Qwen3-VL visual/video encoder parameters."""

    visual = get_qwen_visual_module(qwen_model)
    for parameter in visual.parameters():
        parameter.requires_grad = False
