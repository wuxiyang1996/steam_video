"""Optional raw-video perception tools adapted from the Multi-hop tool idea.

This module is intentionally self-contained and lightweight. It gives the
wrapper a local raw-video backend for smoke tests and hard video_only cases
without importing the full Multi-hop agent runtime or its heavier detector
stack.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import ClipSpan


@dataclass
class VideoToolConfig:
    request_frames: int = 4
    scene_change_threshold: float = 28.0
    max_scene_changes: int = 6


class VideoToolPerceptionBackend:
    """Build clip schemas using local video tools instead of a VLM call."""

    def __init__(self, config: VideoToolConfig | None = None):
        self.config = config or VideoToolConfig()

    def _opencv(self):
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("video_tools backend requires opencv-python") from exc
        return cv2

    def _sample_times(self, clip: ClipSpan, count: int) -> list[float]:
        if count <= 1:
            return [(clip.start_s + clip.end_s) / 2.0]
        span = max(clip.end_s - clip.start_s, 0.1)
        return [clip.start_s + (span * i / max(count - 1, 1)) for i in range(count)]

    def _read_frame(self, video_path: Path, time_s: float) -> tuple[Any | None, dict[str, Any]]:
        cv2 = self._opencv()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, {"ok": False, "time_s": time_s, "error": "video_open_failed"}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(time_s, 0.0) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None, {"ok": False, "time_s": time_s, "error": "frame_read_failed", "fps": fps}
        height, width = frame.shape[:2]
        return frame, {
            "ok": True,
            "time_s": round(time_s, 3),
            "width": int(width),
            "height": int(height),
            "fps": fps,
            "frame_count": frame_count,
        }

    def _mean_frame_diff(self, left: Any, right: Any) -> float:
        cv2 = self._opencv()
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        if left_gray.shape != right_gray.shape:
            right_gray = cv2.resize(right_gray, (left_gray.shape[1], left_gray.shape[0]))
        return float(cv2.absdiff(left_gray, right_gray).mean())

    def _frame_data_uri(self, frame: Any) -> str | None:
        cv2 = self._opencv()
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        data = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"

    def _optional_ocr(self, frame: Any) -> dict[str, Any]:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            return {"available": False, "texts": []}

        cv2 = self._opencv()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.open(io.BytesIO(cv2.imencode(".png", rgb)[1].tobytes()))
        text = pytesseract.image_to_string(image).strip()
        return {"available": True, "texts": [text] if text else []}

    def build_clip_schema(
        self,
        *,
        clip_id: str,
        clip: ClipSpan,
        video_path: Path | None,
        subtitle_context: str | None = None,
        question_context: str | None = None,
    ) -> dict[str, Any]:
        del question_context
        base = {
            "clip_id": clip_id,
            "time_span": clip.to_dict(),
            "granularity": clip.granularity,
            "scene_description": "local video tools inspected sampled frames",
            "observable_facts": [],
            "dialogue_spans": [],
            "entity_mentions": [],
            "events": [],
            "producer": "video_tool_perception_backend",
            "model": "local-video-tools",
        }
        if not video_path or not video_path.exists():
            return {**base, "tool_error": "missing_video_path", "tool_results": []}

        try:
            times = self._sample_times(clip, self.config.request_frames)
            frames: list[tuple[Any, dict[str, Any]]] = []
            results: list[dict[str, Any]] = []
            for time_s in times:
                frame, info = self._read_frame(video_path, time_s)
                results.append({"tool": "get_frame", **info})
                if frame is not None:
                    frames.append((frame, info))

            diffs: list[dict[str, Any]] = []
            for (left, left_info), (right, right_info) in zip(frames, frames[1:]):
                diff = self._mean_frame_diff(left, right)
                diffs.append(
                    {
                        "from_s": left_info["time_s"],
                        "to_s": right_info["time_s"],
                        "mean_absdiff": round(diff, 3),
                    }
                )
            scene_changes = [
                diff for diff in diffs if diff["mean_absdiff"] >= self.config.scene_change_threshold
            ][: self.config.max_scene_changes]
            results.append(
                {
                    "tool": "detect_scene_changes",
                    "threshold": self.config.scene_change_threshold,
                    "changes": scene_changes,
                }
            )

            ocr = self._optional_ocr(frames[0][0]) if frames else {"available": False, "texts": []}
            results.append({"tool": "read_text_in_frame", **ocr})

            facts: list[dict[str, str]] = []
            if frames:
                first_info = frames[0][1]
                facts.append(
                    {
                        "text": (
                            f"Sampled {len(frames)} frames from {clip.start_s:.2f}s to "
                            f"{clip.end_s:.2f}s at approximately {first_info['width']}x{first_info['height']}."
                        ),
                        "modality": "visual",
                    }
                )
            if diffs:
                max_diff = max(diff["mean_absdiff"] for diff in diffs)
                facts.append(
                    {
                        "text": f"Maximum sampled-frame visual change score is {max_diff:.3f}.",
                        "modality": "visual",
                    }
                )
            if subtitle_context:
                facts.append({"text": subtitle_context, "modality": "subtitle"})
            for text in ocr.get("texts", []):
                facts.append({"text": text, "modality": "visual"})

            events = [
                {
                    "description": "local scene-change signal",
                    "time_span": {"start_s": change["from_s"], "end_s": change["to_s"]},
                    "score": change["mean_absdiff"],
                }
                for change in scene_changes
            ]
            payload = {
                **base,
                "scene_description": (
                    "Local video tools sampled frames"
                    + (" and found possible scene changes." if scene_changes else ".")
                ),
                "observable_facts": facts,
                "events": events,
                "tool_results": results,
                "sampled_frame_count": len(frames),
            }
            if frames:
                data_uri = self._frame_data_uri(frames[0][0])
                if data_uri:
                    payload["representative_frame"] = {
                        "time_s": frames[0][1]["time_s"],
                        "image_url": data_uri,
                    }
            return payload
        except Exception as exc:
            return {**base, "tool_error": str(exc), "tool_results": []}
