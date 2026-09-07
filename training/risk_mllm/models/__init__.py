"""Model modules for risk-oriented MLLM training."""

from .multiview_qformer import MultiViewQFormerConfig, MultiViewQFormerAdapter
from .risk_latent_mllm import (
    RiskAuxiliaryHeads,
    RiskCoTAuxiliaryDecoder,
    RiskLatentConfig,
    RiskLatentMLLM,
    build_risk_target_tensors,
    encode_string_targets,
)
from .qwen3_vl_multicamera import (
    Qwen3VLMultiCameraConfig,
    Qwen3VLMultiCameraRiskMLLM,
    build_qwen3_vl_multicamera_model,
    freeze_qwen_video_encoder,
    infer_qwen_hidden_size,
    infer_qwen_visual_feature_size,
)

__all__ = [
    "MultiViewQFormerConfig",
    "MultiViewQFormerAdapter",
    "RiskAuxiliaryHeads",
    "RiskCoTAuxiliaryDecoder",
    "RiskLatentConfig",
    "RiskLatentMLLM",
    "build_risk_target_tensors",
    "encode_string_targets",
    "Qwen3VLMultiCameraConfig",
    "Qwen3VLMultiCameraRiskMLLM",
    "build_qwen3_vl_multicamera_model",
    "freeze_qwen_video_encoder",
    "infer_qwen_hidden_size",
    "infer_qwen_visual_feature_size",
]
