"""Pluggable perception backbones for clip-level captioning."""

from __future__ import annotations

import base64
import importlib.util
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

from ..schemas import BackboneConfig, ClipSpan


class PerceptionBackbone(ABC):
    def __init__(self, config: BackboneConfig):
        self.config = config

    @abstractmethod
    def describe_clip(
        self,
        *,
        video_path: Path,
        clip: ClipSpan,
        question_context: str | None = None,
    ) -> dict[str, Any]:
        """Return a structured observation for one clip span."""


class AnnotationOnlyBackbone(PerceptionBackbone):
    """No model calls; relies on dataset-provided text spans."""

    def describe_clip(
        self,
        *,
        video_path: Path,
        clip: ClipSpan,
        question_context: str | None = None,
    ) -> dict[str, Any]:
        return {
            "text": "",
            "modality": "annotation_only",
            "model": self.config.name,
            "skipped": True,
            "reason": "annotation_only backbone does not run visual captioning",
        }


class OpenRouterVLBackbone(PerceptionBackbone):
    """Caption a clip span with a configurable OpenRouter multimodal model."""

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        if not config.model:
            raise ValueError("OpenRouterVLBackbone requires backbone.model")
        self.api_key = self._load_api_key()

    def _load_api_key(self) -> str:
        env_key = os.environ.get(self.config.api_key_env)
        if env_key:
            return env_key
        keys_path = Path(self.config.keys_py_path) if self.config.keys_py_path else None
        if keys_path and keys_path.exists():
            spec = importlib.util.spec_from_file_location("wrapper_keys", keys_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                key = getattr(module, "OPENROUTER_API_KEY", None)
                if key:
                    return key
        raise RuntimeError(
            f"Missing API key: set {self.config.api_key_env} or provide keys_py_path"
        )

    def _sample_frame_jpegs(self, video_path: Path, clip: ClipSpan, count: int) -> list[str]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenRouterVLBackbone requires opencv-python for frame sampling") from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        frames: list[str] = []
        if count <= 1:
            sample_times = [(clip.start_s + clip.end_s) / 2.0]
        else:
            step = max(clip.end_s - clip.start_s, 0.1) / max(count - 1, 1)
            sample_times = [clip.start_s + i * step for i in range(count)]
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        cap.release()
        return frames

    def describe_clip(
        self,
        *,
        video_path: Path,
        clip: ClipSpan,
        question_context: str | None = None,
    ) -> dict[str, Any]:
        frames = self._sample_frame_jpegs(video_path, clip, self.config.request_frames)
        if not frames:
            return {
                "text": "",
                "modality": "visual_caption",
                "model": self.config.model,
                "skipped": True,
                "reason": "no frames sampled",
            }

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Describe only observable facts in this video clip span. "
                    f"time_span={clip.to_dict()} "
                    f"question_context={question_context or ''}"
                ),
            }
        ]
        for data in frames:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                }
            )

        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "Return one concise factual caption for the clip. No markdown.",
                },
                {"role": "user", "content": user_content},
            ],
        }
        response = requests.post(
            self.config.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        return {
            "text": text,
            "modality": "visual_caption",
            "model": self.config.model,
            "trust_level": "model_labeled",
            "discovery_status": "discovered_runtime",
            "skipped": False,
        }


def build_backbone(config: BackboneConfig) -> PerceptionBackbone:
    name = config.name.lower()
    if name in {"annotation_only", "none", "noop"}:
        return AnnotationOnlyBackbone(config)
    if name in {"openrouter", "openrouter_vl", "vlm"}:
        return OpenRouterVLBackbone(config)
    raise ValueError(f"unsupported backbone: {config.name}")
