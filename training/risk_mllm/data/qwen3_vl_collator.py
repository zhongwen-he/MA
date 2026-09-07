#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch collation for Qwen3-VL multi-camera risk training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import warnings

import torch
from torch import Tensor
import os

from risk_mllm.models.risk_latent_mllm import RiskLatentConfig, build_risk_target_tensors


@dataclass
class Qwen3VLMultiCameraCollator:
    """Prepare VQA samples for ``Qwen3VLMultiCameraRiskMLLM``.

    The collator treats every camera video as an independent Qwen3-VL video
    input, then records ``camera_counts`` so the model can group the flattened
    video features back into batch samples.
    """

    processor: Any
    risk_config: RiskLatentConfig
    max_question_length: int = 1024
    max_answer_length: int = 1024
    max_cot_step_length: int = 128
    video_processor_kwargs: Dict[str, Any] = field(default_factory=lambda: {"fps": 1.0})
    video_text_prompt: str = "."
    strip_video_marker: bool = True
    preserve_all_video_frames: bool = True
    pad_to_even_frames: bool = True
    corrupt_video_policy: str = "zero"
    fallback_frame_count: int = 6
    fallback_frame_size: Tuple[int, int] = (224, 224)

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        question_batch = self._tokenize_questions(samples)
        answer_batch = self._tokenize_answers(samples)
        cot_step_batch = self._tokenize_cot_steps(samples)
        video_batch = self._process_camera_videos(samples)
        risk_targets = build_risk_target_tensors(samples, self.risk_config)

        return {
            **video_batch,
            "question_input_ids": question_batch["input_ids"],
            "question_attention_mask": question_batch["attention_mask"],
            "answer_input_ids": answer_batch["input_ids"],
            "answer_attention_mask": answer_batch["attention_mask"],
            "answer_labels": answer_batch["input_ids"].clone(),
            "cot_step_input_ids": cot_step_batch["input_ids"],
            "cot_step_attention_mask": cot_step_batch["attention_mask"],
            "cot_step_labels": cot_step_batch["labels"],
            "risk_targets": risk_targets,
            "sample_ids": [sample.get("id") for sample in samples],
            "scenes": [sample.get("scene") for sample in samples],
        }

    def _tokenize_questions(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Tensor]:
        tokenizer = self._tokenizer
        texts = [self._normalize_question(sample.get("question", "")) for sample in samples]
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )

    def _tokenize_answers(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Tensor]:
        tokenizer = self._tokenizer
        eos_token = tokenizer.eos_token or ""
        texts = [str(sample.get("answer", "")) + eos_token for sample in samples]
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_answer_length,
            return_tensors="pt",
        )

    def _tokenize_cot_steps(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Tensor]:
        tokenizer = self._tokenizer
        num_steps = int(getattr(self.risk_config, "num_cot_steps", 4))
        flat_steps: List[str] = []
        for sample in samples:
            steps = list(sample.get("cot_steps", []))
            if len(steps) < num_steps:
                steps.extend([""] * (num_steps - len(steps)))
            flat_steps.extend(str(step) for step in steps[:num_steps])

        batch = tokenizer(
            flat_steps,
            padding=True,
            truncation=True,
            max_length=self.max_cot_step_length,
            return_tensors="pt",
        )
        batch_size = len(samples)
        input_ids = batch["input_ids"].view(batch_size, num_steps, -1)
        attention_mask = batch["attention_mask"].view(batch_size, num_steps, -1)
        labels = input_ids.clone().masked_fill(attention_mask == 0, self.risk_config.ignore_index)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _process_camera_videos(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Tensor]:
        camera_counts = []
        video_outputs = []
        for sample in samples:
            video_paths = list(sample.get("video_paths", []))
            if not video_paths:
                raise ValueError(f"Sample {sample.get('id')} has no video paths")
            camera_counts.append(len(video_paths))
            for video_path in video_paths:
                video_outputs.append(self._process_single_video(video_path))

        pixel_values = [output["pixel_values_videos"] for output in video_outputs]
        grids = [output["video_grid_thw"] for output in video_outputs]
        return {
            "pixel_values_videos": torch.cat(pixel_values, dim=0),
            "video_grid_thw": torch.cat(grids, dim=0).long(),
            "camera_counts": torch.tensor(camera_counts, dtype=torch.long),
        }

    def _process_single_video(self, video_path: str) -> Dict[str, Tensor]:
        if self.preserve_all_video_frames:
            return self._process_single_video_as_keyframes(video_path)
        try:
            return self._process_single_video_with_qwen_utils(video_path)
        except ImportError:
            return self._process_single_video_with_processor(video_path)

    def _process_single_video_as_keyframes(self, video_path: str) -> Dict[str, Tensor]:
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers.video_utils import VideoMetadata

        frames = self._read_all_video_frames(video_path)
        if self.pad_to_even_frames and len(frames) % 2 == 1:
            frames.append(frames[-1].copy())
        sample_fps = float(self.video_processor_kwargs.get("fps", 2.0))
        raw_fps = float(self.video_processor_kwargs.get("raw_fps", sample_fps))
        video_metadata = [
            VideoMetadata(
                total_num_frames=len(frames),
                fps=raw_fps,
                frames_indices=list(range(len(frames))),
            )
        ]

        video_kwargs = {
            key: value
            for key, value in self.video_processor_kwargs.items()
            if key not in {"reader_backend", "fps"}
        }
        video_item = {
            "type": "video",
            "video": frames,
            "sample_fps": sample_fps,
            "raw_fps": raw_fps,
            **video_kwargs,
        }
        message = [
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": self.video_text_prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        video_kwargs = self._normalize_processor_video_kwargs(video_kwargs)
        output = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            return_tensors="pt",
            **video_kwargs,
        )
        return self._extract_video_tensors(output)

    def _read_all_video_frames(self, video_path: str) -> List[Any]:
        try:
            import torchvision
            from PIL import Image
        except ImportError as exc:
            raise ImportError("preserve_all_video_frames=True requires torchvision video decoding") from exc

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        try:
            frames, _, _ = torchvision.io.read_video(str(path), pts_unit="sec", output_format="TCHW")
        except Exception as exc:
            return self._handle_bad_video(video_path, exc)
        if frames.numel() == 0:
            return self._handle_bad_video(video_path, ValueError(f"No frames decoded from video: {video_path}"))

        pil_frames = []
        for frame in frames:
            frame = frame.permute(1, 2, 0).cpu().numpy()
            pil_frames.append(Image.fromarray(frame))
        return pil_frames

    def _handle_bad_video(self, video_path: str, exc: Exception) -> List[Any]:
        if self.corrupt_video_policy != "zero":
            raise exc
        from PIL import Image

        frame_count = max(2, int(self.fallback_frame_count))
        if self.pad_to_even_frames and frame_count % 2 == 1:
            frame_count += 1
        warnings.warn(
            f"Replacing unreadable video with {frame_count} black frames: {video_path} ({exc})",
            RuntimeWarning,
        )
        return [
            Image.new("RGB", self.fallback_frame_size, color=(0, 0, 0))
            for _ in range(frame_count)
        ]

    def _process_single_video_with_qwen_utils(self, video_path: str) -> Dict[str, Tensor]:
        reader_backend = self.video_processor_kwargs.get("reader_backend")
        if reader_backend:
            os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", str(reader_backend))
        from qwen_vl_utils import process_vision_info

        video_kwargs = {
            key: value
            for key, value in self.video_processor_kwargs.items()
            if key != "reader_backend"
        }
        video_item = {"type": "video", "video": video_path, **video_kwargs}
        message = [
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": self.video_text_prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        video_kwargs = self._normalize_processor_video_kwargs(video_kwargs)
        output = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        )
        return self._extract_video_tensors(output)

    def _process_single_video_with_processor(self, video_path: str) -> Dict[str, Tensor]:
        video_kwargs = {
            key: value
            for key, value in self.video_processor_kwargs.items()
            if key != "reader_backend"
        }
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, **video_kwargs},
                    {"type": "text", "text": self.video_text_prompt},
                ],
            }
        ]
        output = self.processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        return self._extract_video_tensors(output)

    def _extract_video_tensors(self, output: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "pixel_values_videos" not in output or "video_grid_thw" not in output:
            raise ValueError(
                "Qwen3-VL processor did not return pixel_values_videos/video_grid_thw. "
                "Check that video decoding dependencies are installed and the video path is valid."
            )
        return {
            "pixel_values_videos": output["pixel_values_videos"],
            "video_grid_thw": output["video_grid_thw"],
        }

    def _normalize_processor_video_kwargs(self, video_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(video_kwargs)
        for key in ("fps", "sample_fps", "raw_fps"):
            value = normalized.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 1:
                normalized[key] = value[0]
        return normalized

    def _normalize_question(self, question: str) -> str:
        if not self.strip_video_marker:
            return question
        return question.replace("<video>", "<multi-camera-video>")

    @property
    def _tokenizer(self) -> Any:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Processor does not expose a tokenizer")
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
