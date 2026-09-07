#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pretrained Hugging Face BLIP-2 Q-Former adapter for multi-view video features.

Architecture:

    visual tokens [B, 6, T, N, D_visual] or [B, L, D_visual]
      -> optional camera / temporal embeddings
      -> flatten multi-view / temporal visual sequence
      -> visual feature adapter:
             Qwen vision hidden dim -> pretrained BLIP-2 encoder hidden dim
      -> pretrained BLIP-2 Q-Former
             pretrained query tokens
             pretrained Q-Former weights
      -> Q-Former output [B, 32, 768]
      -> projector:
             Q-Former hidden dim -> target Qwen LLM hidden dim
      -> fused visual tokens [B, 32, d_llm]

No stride-based token downsampling is applied.

The Q-Former architecture and query embeddings are loaded directly from
the official Salesforce BLIP-2 checkpoint instead of being randomly
initialized.

Object ROI pooling, bbox metadata, explicit cross-camera identity matching,
and hierarchical fusion are deliberately omitted. Cross-view association
is learned implicitly through joint attention over the multi-view visual
token sequence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import gc
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from torch import Tensor, nn
from transformers import Blip2Model, Blip2QFormerConfig, Blip2QFormerModel


@dataclass
class MultiViewQFormerConfig:
    # Dimension produced by the upstream Qwen vision encoder.
    visual_input_dim: int = 2048

    # Hidden dimension expected by the target Qwen language model.
    # Set this to the actual hidden_size of your Qwen LM.
    llm_hidden_dim: int = 4096

    # Multi-view / temporal structure.
    num_views: int = 6
    num_frames: int = 5
    tokens_per_frame: Optional[int] = None

    # Official pretrained BLIP-2 checkpoint.
    pretrained_model_name: str = "Salesforce/blip2-opt-2.7b"
    qformer_checkpoint_path: Optional[str] = (
        "training/pretrained/blip2_qformer/Salesforce_blip2-opt-2.7b/qformer_query.pt"
    )

    # Additional multi-view information added before the Q-Former.
    use_camera_embedding: bool = True
    use_temporal_embedding: bool = True
    use_visual_input_layernorm: bool = True

    # Whether pretrained Q-Former/query weights should remain trainable.
    freeze_qformer: bool = False

    # Whether to freeze pretrained query embeddings separately.
    # Normally keep False if you want them to adapt to multi-view driving.
    freeze_query_tokens: bool = False


class MultiViewQFormerAdapter(nn.Module):
    """Compress multi-view Qwen visual features with a pretrained BLIP-2 Q-Former."""

    def __init__(self, config: MultiViewQFormerConfig) -> None:
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # 1. Load pretrained BLIP-2 Q-Former/query tokens.
        #
        # Prefer a lightweight extracted checkpoint created by:
        #
        #   training/scripts/download_stage2_pretrained.py
        #
        # This avoids loading the full BLIP-2 OPT language model at Stage 2
        # startup. The full Blip2Model fallback remains available for machines
        # with enough CPU RAM, but should not be the default path.
        # ------------------------------------------------------------------
        checkpoint_path = Path(config.qformer_checkpoint_path).expanduser() if config.qformer_checkpoint_path else None
        if checkpoint_path is not None and checkpoint_path.is_file():
            self._load_extracted_qformer_checkpoint(checkpoint_path)
        else:
            self._load_qformer_from_full_blip2(config.pretrained_model_name)

        # Optional sanity checks.
        if self.query_tokens.shape[1] != self.num_queries:
            raise ValueError(
                "Mismatch between pretrained query token tensor and "
                f"config.num_query_tokens: "
                f"{self.query_tokens.shape[1]} vs {self.num_queries}"
            )

        if self.query_tokens.shape[2] != self.qformer_hidden_dim:
            raise ValueError(
                "Mismatch between pretrained query-token hidden size and "
                f"Q-Former hidden size: "
                f"{self.query_tokens.shape[2]} vs "
                f"{self.qformer_hidden_dim}"
            )

        # ------------------------------------------------------------------
        # 2. Multi-view / temporal embeddings.
        #
        # These operate in the ORIGINAL Qwen visual feature space.
        #
        # Example:
        #   Qwen visual token = 2048-d
        #
        # camera/time embeddings are therefore also 2048-d.
        # ------------------------------------------------------------------
        self.camera_embedding = (
            nn.Embedding(
                config.num_views,
                config.visual_input_dim,
            )
            if config.use_camera_embedding
            else None
        )

        self.temporal_embedding = (
            nn.Embedding(
                config.num_frames,
                config.visual_input_dim,
            )
            if config.use_temporal_embedding
            else None
        )

        # ------------------------------------------------------------------
        # 3. Visual feature adapter.
        #
        # Qwen vision feature dimension:
        #       visual_input_dim, e.g. 2048
        #
        # official BLIP-2 Q-Former expects:
        #       encoder_hidden_size, normally 1408
        #
        # Therefore:
        #
        #       2048 -> 1408
        #
        # This layer is newly initialized and should be trained on your data.
        # ------------------------------------------------------------------
        if config.visual_input_dim == self.qformer_encoder_dim:
            self.visual_adapter = (
                nn.LayerNorm(config.visual_input_dim)
                if config.use_visual_input_layernorm
                else nn.Identity()
            )
        else:
            layers = []
            if config.use_visual_input_layernorm:
                layers.append(nn.LayerNorm(config.visual_input_dim))
            layers.append(
                nn.Linear(
                    config.visual_input_dim,
                    self.qformer_encoder_dim,
                )
            )
            self.visual_adapter = nn.Sequential(*layers)

        # ------------------------------------------------------------------
        # 4. Q-Former -> Qwen LM projector.
        #
        # pretrained Q-Former output:
        #       [B, 32, qformer_hidden_dim]
        #       normally [B, 32, 768]
        #
        # target:
        #       [B, 32, llm_hidden_dim]
        #
        # Example:
        #       768 -> 4096
        #
        # This projector is task/model-specific and newly initialized.
        # ------------------------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(
                self.qformer_hidden_dim,
                self.qformer_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.qformer_hidden_dim,
                config.llm_hidden_dim,
            ),
        )

        # ------------------------------------------------------------------
        # 5. Freeze controls.
        # ------------------------------------------------------------------
        if config.freeze_qformer:
            self.qformer.requires_grad_(False)

        if config.freeze_query_tokens:
            self.query_tokens.requires_grad_(False)

        self.reset_new_parameters()

    def _load_extracted_qformer_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        qformer_config_dict = checkpoint.get("qformer_config")
        if not isinstance(qformer_config_dict, dict):
            raise ValueError(f"Missing qformer_config in extracted checkpoint: {checkpoint_path}")
        qformer_config = Blip2QFormerConfig(**qformer_config_dict)
        self.qformer = Blip2QFormerModel(qformer_config)

        qformer_state_dict = checkpoint.get("qformer_state_dict")
        if not isinstance(qformer_state_dict, dict):
            raise ValueError(f"Missing qformer_state_dict in extracted checkpoint: {checkpoint_path}")
        self.qformer.load_state_dict(qformer_state_dict, strict=True)

        query_tokens = checkpoint.get("query_tokens")
        if not isinstance(query_tokens, Tensor):
            raise ValueError(f"Missing query_tokens in extracted checkpoint: {checkpoint_path}")
        self.query_tokens = nn.Parameter(query_tokens.detach().clone())

        self.qformer_hidden_dim = int(qformer_config.hidden_size)
        self.qformer_encoder_dim = int(qformer_config.encoder_hidden_size)
        self.num_queries = int(checkpoint.get("num_query_tokens", self.query_tokens.shape[1]))

    def _load_qformer_from_full_blip2(self, pretrained_model_name: str) -> None:
        pretrained_blip2 = Blip2Model.from_pretrained(
            pretrained_model_name,
            low_cpu_mem_usage=True,
        )

        self.qformer = pretrained_blip2.qformer
        self.query_tokens = nn.Parameter(pretrained_blip2.query_tokens.detach().clone())
        self.qformer_hidden_dim = int(pretrained_blip2.config.qformer_config.hidden_size)
        self.qformer_encoder_dim = int(pretrained_blip2.config.qformer_config.encoder_hidden_size)
        self.num_queries = int(pretrained_blip2.config.num_query_tokens)

        del pretrained_blip2
        gc.collect()

    def reset_new_parameters(self) -> None:
        """Initialize only newly added task-specific modules.

        IMPORTANT:
        pretrained Q-Former weights and pretrained query tokens are NOT reset.
        """

        if self.camera_embedding is not None:
            nn.init.normal_(
                self.camera_embedding.weight,
                mean=0.0,
                std=0.02,
            )

        if self.temporal_embedding is not None:
            nn.init.normal_(
                self.temporal_embedding.weight,
                mean=0.0,
                std=0.02,
            )

        if isinstance(self.visual_adapter, nn.Linear):
            nn.init.xavier_uniform_(self.visual_adapter.weight)
            if self.visual_adapter.bias is not None:
                nn.init.zeros_(self.visual_adapter.bias)
        else:
            for module in self.visual_adapter.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        visual_embeds: Tensor,
        visual_attention_mask: Optional[Tensor] = None,
        return_qformer_tokens: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Return fused visual tokens in the target Qwen LM hidden dimension.

        Args:
            visual_embeds:
                One of:

                [B, V, T, N, D_visual]
                [B, V, N, D_visual]
                [B, L, D_visual]

            visual_attention_mask:
                Optional mask where:

                    1 = valid visual token
                    0 = invalid / padded visual token

            return_qformer_tokens:
                If True, also return the raw pretrained-Q-Former output:

                    [B, num_queries, qformer_hidden_dim]

        Returns:
            llm_tokens:
                [B, num_queries, llm_hidden_dim]

            optionally:

            qformer_tokens:
                [B, num_queries, qformer_hidden_dim]
        """

        # --------------------------------------------------------------
        # Multi-view/video preprocessing.
        #
        # IMPORTANT:
        # No stride downsampling is performed.
        # --------------------------------------------------------------
        visual_tokens, attention_mask = self._prepare_visual_tokens(
            visual_embeds,
            visual_attention_mask,
        )

        # --------------------------------------------------------------
        # Convert Qwen visual feature dimension to the dimension expected
        # by the pretrained BLIP-2 Q-Former.
        #
        # Example:
        #   [B, L, 2048]
        #       ->
        #   [B, L, 1408]
        # --------------------------------------------------------------
        visual_tokens = self.visual_adapter(visual_tokens)

        # Match Q-Former device and dtype.
        qformer_parameter = next(self.qformer.parameters())

        visual_tokens = visual_tokens.to(
            device=qformer_parameter.device,
            dtype=qformer_parameter.dtype,
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(
                device=qformer_parameter.device
            )

        batch_size = visual_tokens.size(0)

        # --------------------------------------------------------------
        # Expand the OFFICIAL PRETRAINED query embeddings across batch.
        #
        # [1, 32, 768]
        #       ->
        # [B, 32, 768]
        # --------------------------------------------------------------
        queries = self.query_tokens.expand(
            batch_size,
            -1,
            -1,
        )

        queries = queries.to(
            device=qformer_parameter.device,
            dtype=qformer_parameter.dtype,
        )

        # --------------------------------------------------------------
        # Official pretrained Q-Former forward.
        #
        # Queries:
        #       [B, 32, 768]
        #
        # Visual memory:
        #       [B, L, 1408]
        #
        # Output:
        #       [B, 32, 768]
        # --------------------------------------------------------------
        qformer_outputs = self.qformer(
            query_embeds=queries,
            encoder_hidden_states=visual_tokens,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        qformer_tokens = qformer_outputs.last_hidden_state

        # --------------------------------------------------------------
        # Project pretrained Q-Former output into Qwen LM input space.
        #
        # Example:
        #       [B, 32, 768]
        #            ->
        #       [B, 32, 4096]
        # --------------------------------------------------------------
        projector_parameter = next(self.projector.parameters())

        qformer_tokens_for_projector = qformer_tokens.to(
            device=projector_parameter.device,
            dtype=projector_parameter.dtype,
        )

        llm_tokens = self.projector(
            qformer_tokens_for_projector
        )

        if return_qformer_tokens:
            return llm_tokens, qformer_tokens

        return llm_tokens

    def _prepare_visual_tokens(
        self,
        visual_embeds: Tensor,
        visual_attention_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Prepare multi-view/video tokens without token downsampling."""

        if visual_embeds.dim() == 5:
            # ----------------------------------------------------------
            # [B, V, T, N, D]
            #
            # Add both:
            #   camera embedding
            #   temporal embedding
            # ----------------------------------------------------------
            visual_embeds = self._add_view_time_embeddings_5d(
                visual_embeds
            )

            batch_size, views, frames, tokens, dim = (
                visual_embeds.shape
            )

            visual_embeds = visual_embeds.reshape(
                batch_size,
                views * frames * tokens,
                dim,
            )

            if (
                visual_attention_mask is not None
                and visual_attention_mask.dim() == 4
            ):
                visual_attention_mask = (
                    visual_attention_mask.reshape(
                        batch_size,
                        views * frames * tokens,
                    )
                )

        elif visual_embeds.dim() == 4:
            # ----------------------------------------------------------
            # [B, V, N, D]
            #
            # Add camera embedding only.
            # ----------------------------------------------------------
            visual_embeds = self._add_view_embeddings_4d(
                visual_embeds
            )

            batch_size, views, tokens, dim = (
                visual_embeds.shape
            )

            visual_embeds = visual_embeds.reshape(
                batch_size,
                views * tokens,
                dim,
            )

            if (
                visual_attention_mask is not None
                and visual_attention_mask.dim() == 3
            ):
                visual_attention_mask = (
                    visual_attention_mask.reshape(
                        batch_size,
                        views * tokens,
                    )
                )

        elif visual_embeds.dim() == 3:
            # ----------------------------------------------------------
            # [B, L, D]
            #
            # If tokens_per_frame is known and L matches the expected
            # multi-view/video layout, reconstruct [B,V,T,N,D] so that
            # camera/time embeddings can still be added.
            # ----------------------------------------------------------
            visual_embeds = (
                self._maybe_reshape_flat_and_add_embeddings(
                    visual_embeds
                )
            )

        else:
            raise ValueError(
                "Expected visual_embeds rank 3, 4, or 5, "
                f"got rank={visual_embeds.dim()}"
            )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Previous implementation:
        #
        #   visual_embeds = visual_embeds[:, ::4, :]
        #
        # has been completely removed.
        #
        # Every visual token is now available to the pretrained Q-Former.
        # --------------------------------------------------------------

        return visual_embeds, visual_attention_mask

    def _add_view_time_embeddings_5d(
        self,
        visual_embeds: Tensor,
    ) -> Tensor:
        """Add camera and temporal identity embeddings."""

        batch_size, views, frames, tokens, dim = (
            visual_embeds.shape
        )

        if dim != self.config.visual_input_dim:
            raise ValueError(
                "visual_embeds last dimension does not match "
                f"visual_input_dim={self.config.visual_input_dim}: "
                f"got {dim}"
            )

        if views > self.config.num_views:
            raise ValueError(
                f"Configured num_views={self.config.num_views}, "
                f"got {views}"
            )

        if frames > self.config.num_frames:
            raise ValueError(
                f"Configured num_frames={self.config.num_frames}, "
                f"got {frames}"
            )

        if self.camera_embedding is not None:
            camera_ids = torch.arange(
                views,
                device=visual_embeds.device,
            )

            camera_embed = self.camera_embedding(
                camera_ids
            ).view(
                1,
                views,
                1,
                1,
                dim,
            )

            visual_embeds = (
                visual_embeds + camera_embed
            )

        if self.temporal_embedding is not None:
            frame_ids = torch.arange(
                frames,
                device=visual_embeds.device,
            )

            temporal_embed = self.temporal_embedding(
                frame_ids
            ).view(
                1,
                1,
                frames,
                1,
                dim,
            )

            visual_embeds = (
                visual_embeds + temporal_embed
            )

        return visual_embeds

    def _add_view_embeddings_4d(
        self,
        visual_embeds: Tensor,
    ) -> Tensor:
        """Add camera identity embeddings to [B,V,N,D] input."""

        batch_size, views, tokens, dim = (
            visual_embeds.shape
        )

        if dim != self.config.visual_input_dim:
            raise ValueError(
                "visual_embeds last dimension does not match "
                f"visual_input_dim={self.config.visual_input_dim}: "
                f"got {dim}"
            )

        if views > self.config.num_views:
            raise ValueError(
                f"Configured num_views={self.config.num_views}, "
                f"got {views}"
            )

        if self.camera_embedding is not None:
            camera_ids = torch.arange(
                views,
                device=visual_embeds.device,
            )

            camera_embed = self.camera_embedding(
                camera_ids
            ).view(
                1,
                views,
                1,
                dim,
            )

            visual_embeds = (
                visual_embeds + camera_embed
            )

        return visual_embeds

    def _maybe_reshape_flat_and_add_embeddings(
        self,
        visual_embeds: Tensor,
    ) -> Tensor:
        """Recover [V,T,N] layout from flat tokens when possible."""

        tokens_per_frame = self.config.tokens_per_frame

        if tokens_per_frame is None:
            return visual_embeds

        batch_size, length, dim = visual_embeds.shape

        if dim != self.config.visual_input_dim:
            raise ValueError(
                "visual_embeds last dimension does not match "
                f"visual_input_dim={self.config.visual_input_dim}: "
                f"got {dim}"
            )

        expected = (
            self.config.num_views
            * self.config.num_frames
            * tokens_per_frame
        )

        if length != expected:
            return visual_embeds

        visual_embeds = visual_embeds.view(
            batch_size,
            self.config.num_views,
            self.config.num_frames,
            tokens_per_frame,
            dim,
        )

        visual_embeds = (
            self._add_view_time_embeddings_5d(
                visual_embeds
            )
        )

        return visual_embeds.reshape(
            batch_size,
            length,
            dim,
        )

    def get_architecture_info(self) -> dict:
        """Return the effective pretrained Q-Former configuration."""

        qcfg = self.qformer.config

        return {
            "pretrained_model_name":
                self.config.pretrained_model_name,

            "visual_input_dim":
                self.config.visual_input_dim,

            "qformer_encoder_hidden_dim":
                self.qformer_encoder_dim,

            "qformer_hidden_dim":
                self.qformer_hidden_dim,

            "num_query_tokens":
                self.num_queries,

            "num_hidden_layers":
                qcfg.num_hidden_layers,

            "num_attention_heads":
                qcfg.num_attention_heads,

            "intermediate_size":
                qcfg.intermediate_size,

            "cross_attention_frequency":
                qcfg.cross_attention_frequency,

            "llm_hidden_dim":
                self.config.llm_hidden_dim,

            "use_camera_embedding":
                self.config.use_camera_embedding,

            "use_temporal_embedding":
                self.config.use_temporal_embedding,

            "freeze_qformer":
                self.config.freeze_qformer,

            "freeze_query_tokens":
                self.config.freeze_query_tokens,
        }
