#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Qwen3.5-VL adapter for the StreamBridge agent loop.

This adapter keeps StreamBridge's external streaming-model interface while using
the Hugging Face Qwen video-clip path underneath. It does not reuse
StreamBridge's internal image-embedding cache; instead, each response sees the
latest causally visible clip from the source video.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class Qwen35_VL(nn.Module):
    def __init__(
        self,
        ckpt: str,
        video_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict[str, Any] = "auto",
        stream_fps: float = 1.0,
        clip_window_s: float = 4.0,
        video_fps: float = 2.0,
        video_max_frames_per_clip: int = 8,
        system_prompt: str = "You are a helpful streaming video assistant.",
        max_history_turns: int = 6,
    ):
        super().__init__()
        from baseline.model_runtime import attention_kwargs

        self.dtype = dtype
        self.ckpt = ckpt
        self.video_path = video_path
        self.stream_fps = stream_fps
        self.clip_window_s = clip_window_s
        self.video_fps = video_fps
        self.video_max_frames_per_clip = video_max_frames_per_clip
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.language_model = AutoModelForImageTextToText.from_pretrained(
            ckpt,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **attention_kwargs(),
        )
        self.language_model.eval()

        self.frame_count = 0
        self.visible_until_s = 0.0
        self.pending_question: str | None = None
        self.history: list[tuple[str, str]] = []

    @property
    def device(self):
        try:
            return self.language_model.device
        except AttributeError:
            return next(self.language_model.parameters()).device

    def maybe_autocast(self):
        if self.device == torch.device("cpu"):
            return contextlib.nullcontext()
        return torch.cuda.amp.autocast(dtype=self.dtype)

    def receive_one_frame(self, *args, timestamp_s: float | None = None, **kwargs):
        """Record stream progress.

        The activation model consumes the actual frame tensor. Qwen3.5 receives
        source-video clips at response time, so this method only tracks causal
        visibility.
        """
        self.frame_count += 1
        if timestamp_s is None:
            self.visible_until_s = self.frame_count / max(self.stream_fps, 1e-6)
        else:
            self.visible_until_s = max(0.0, float(timestamp_s))

    def receive_user_input(self, text: str):
        self.pending_question = text

    def _clip_bounds(self) -> tuple[float, float]:
        end_s = max(0.0, self.visible_until_s)
        start_s = max(0.0, end_s - self.clip_window_s)
        if end_s <= start_s:
            end_s = start_s + max(1.0 / max(self.video_fps, 1e-6), 0.1)
        return start_s, end_s

    def _build_prompt(self) -> str:
        task = self.pending_question or "Describe the important visible action now."
        lines = [
            self.system_prompt,
            "Use only the provided video clip, which is from the causally visible part of the stream.",
            f"Visible video cutoff: {self.visible_until_s:.2f} seconds.",
        ]
        if self.history:
            lines.append("Recent interaction history:")
            for role, text in self.history[-self.max_history_turns :]:
                lines.append(f"{role}: {text}")
        lines.extend(
            [
                f"Current user request: {task}",
                "Answer concisely with the specific action or information visible now.",
            ]
        )
        return "\n".join(lines)

    def _build_messages(self) -> list[dict[str, Any]]:
        start_s, end_s = self._clip_bounds()
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": self.video_path,
                        "video_start": start_s,
                        "video_end": end_s,
                        "fps": self.video_fps,
                        "max_frames": self.video_max_frames_per_clip,
                    },
                    {"type": "text", "text": self._build_prompt()},
                ],
            }
        ]

    def _read_clip_video(self, start_s: float, end_s: float):
        try:
            import decord

            reader = decord.VideoReader(self.video_path, ctx=decord.cpu(0), num_threads=1)
            source_fps = float(reader.get_avg_fps())
            total_frames = len(reader)
            duration_s = total_frames / max(source_fps, 1e-6)
            safe_end_s = min(max(end_s, 0.0), max(duration_s - (0.5 / max(source_fps, 1e-6)), 0.0))
            safe_start_s = min(max(start_s, 0.0), safe_end_s)
            if safe_end_s <= safe_start_s:
                safe_start_s = max(0.0, safe_end_s - max(1.0 / max(source_fps, 1e-6), 0.1))
            start_idx = min(max(0, int(safe_start_s * source_fps)), max(total_frames - 1, 0))
            end_idx = max(start_idx, int(safe_end_s * source_fps))
            end_idx = min(end_idx, total_frames - 1)
            frame_count = max(2, int((end_s - start_s) * self.video_fps))
            frame_count = min(max(2, self.video_max_frames_per_clip), max(frame_count, 2))
            frame_indices = np.linspace(start_idx, end_idx, frame_count).round().astype(int).tolist()
            frames = reader.get_batch(frame_indices).asnumpy()
        except ModuleNotFoundError:
            import cv2

            capture = cv2.VideoCapture(self.video_path)
            if not capture.isOpened():
                raise RuntimeError(f"could not open video: {self.video_path}")
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or self.video_fps)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_s = total_frames / max(source_fps, 1e-6)
            safe_end_s = min(max(end_s, 0.0), max(duration_s - (0.5 / max(source_fps, 1e-6)), 0.0))
            safe_start_s = min(max(start_s, 0.0), safe_end_s)
            if safe_end_s <= safe_start_s:
                safe_start_s = max(0.0, safe_end_s - max(1.0 / max(source_fps, 1e-6), 0.1))
            start_idx = min(max(0, int(safe_start_s * source_fps)), max(total_frames - 1, 0))
            end_idx = max(start_idx, int(safe_end_s * source_fps))
            end_idx = min(end_idx, total_frames - 1)
            frame_count = max(2, int((end_s - start_s) * self.video_fps))
            frame_count = min(max(2, self.video_max_frames_per_clip), max(frame_count, 2))
            frame_indices = np.linspace(start_idx, end_idx, frame_count).round().astype(int).tolist()
            decoded = []
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    capture.release()
                    raise RuntimeError(f"could not read frame {frame_index} from {self.video_path}")
                decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            capture.release()
            frames = np.stack(decoded, axis=0)

        video = torch.from_numpy(frames).permute(0, 3, 1, 2)
        metadata = {
            "total_num_frames": total_frames,
            "fps": source_fps,
            "frames_indices": frame_indices,
            "video_backend": "streambridge_fallback",
        }
        return video, metadata

    @torch.inference_mode()
    def response(self, **generate_kwargs):
        from transformers.video_utils import VideoMetadata

        max_new_tokens = generate_kwargs.pop("max_new_tokens", 128)
        generate_kwargs.pop("min_length", None)
        generate_kwargs.pop("num_return_sequences", None)
        generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}
        generate_kwargs.setdefault("do_sample", False)

        messages = self._build_messages()
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(generate_kwargs.pop("enable_thinking", False)),
        )
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except Exception as exc:
            start_s, end_s = self._clip_bounds()
            video_tensor, metadata = self._read_clip_video(start_s, end_s)
            metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            image_inputs = None
            video_inputs = [(video_tensor, metadata)]
            video_kwargs = {"do_sample_frames": False}

        videos = []
        video_metadata = []
        for video_input in video_inputs or []:
            if isinstance(video_input, tuple) and len(video_input) == 2:
                video_tensor, metadata = video_input
                frame_indices = metadata.get("frames_indices")
                if hasattr(frame_indices, "tolist"):
                    frame_indices = frame_indices.tolist()
                video_metadata.append(
                    VideoMetadata(
                        total_num_frames=int(metadata.get("total_num_frames") or video_tensor.shape[0]),
                        fps=metadata.get("fps"),
                        frames_indices=[int(index) for index in frame_indices] if frame_indices is not None else None,
                        video_backend=metadata.get("video_backend"),
                    )
                )
                videos.append(video_tensor)
            else:
                videos.append(video_input)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadata or None,
            **video_kwargs,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        output_ids = self.language_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[:, input_len:]
        output_text = self.processor.batch_decode(generated, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        if self.pending_question and output_text:
            self.history.append(("user", self.pending_question))
            self.history.append(("assistant", output_text[0]))
            self.history = self.history[-self.max_history_turns :]

        return output_text

    def reset(self):
        self.frame_count = 0
        self.visible_until_s = 0.0
        self.pending_question = None
        self.history = []
