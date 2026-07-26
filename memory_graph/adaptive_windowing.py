"""Question-independent surprise-driven windowing for visual L1 writes.

The writer consumes a stream of visual representations.  It deliberately does
not inspect the downstream question and it does not ask an LLM to emit a
numeric surprise score.  A learned video encoder can implement
``VisualFeatureProvider``; the OpenCV provider below is only a runnable,
dependency-light fallback for smoke tests and data preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class VisualFeatureSample:
    timestamp_s: float
    vector: tuple[float, ...]
    frame_index: int | None = None

    def __post_init__(self) -> None:
        if self.timestamp_s < 0:
            raise ValueError("feature timestamp must be non-negative")
        if not self.vector:
            raise ValueError("visual feature vector must not be empty")
        if any(not math.isfinite(float(value)) for value in self.vector):
            raise ValueError("visual feature vector must be finite")


@dataclass(frozen=True)
class AdaptiveWindow:
    start_s: float
    end_s: float
    boundary_reason: str
    sample_count: int

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("adaptive window must have a positive duration")
        if self.sample_count < 1:
            raise ValueError("adaptive window must contain a sampled representation")

    def to_dict(self) -> dict[str, object]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "purpose": "surprise_adaptive_coarse_scan",
            "boundary_reason": self.boundary_reason,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class SurpriseWindowConfig:
    sample_period_s: float = 0.5
    min_window_s: float = 2.0
    max_window_s: float = 12.0
    calibration_history: int = 8
    high_surprise_quantile: float = 0.8

    def __post_init__(self) -> None:
        if self.sample_period_s <= 0:
            raise ValueError("sample_period_s must be positive")
        if self.min_window_s <= 0 or self.max_window_s < self.min_window_s:
            raise ValueError("adaptive window duration bounds are invalid")
        if self.calibration_history < 2:
            raise ValueError("calibration_history must be at least two")
        if not 0.5 <= self.high_surprise_quantile < 1.0:
            raise ValueError("high_surprise_quantile must be in [0.5, 1.0)")


class VisualFeatureProvider(Protocol):
    provider_name: str

    def sample(
        self,
        video_path: str | Path,
        *,
        duration_s: float,
        sample_period_s: float,
    ) -> Sequence[VisualFeatureSample]: ...


class AdaptiveWindowProvider(Protocol):
    provider_name: str

    def windows(
        self,
        video_path: str | Path,
        *,
        duration_s: float,
    ) -> list[dict[str, object]]: ...


@dataclass(frozen=True)
class SurpriseWindowWriter:
    """Convert representation changes into bounded-duration write windows.

    The high-surprise boundary is calibrated from the empirical rank of recent
    representation changes.  Stable content therefore grows until
    ``max_window_s``; a non-routine representation change closes a window as
    soon as ``min_window_s`` permits.  The policy is independent of any query.
    """

    config: SurpriseWindowConfig = SurpriseWindowConfig()

    def segment(
        self,
        samples: Sequence[VisualFeatureSample],
        *,
        duration_s: float,
    ) -> tuple[AdaptiveWindow, ...]:
        if duration_s <= 0:
            raise ValueError("video duration must be positive")
        ordered = tuple(sorted(samples, key=lambda value: value.timestamp_s))
        if not ordered:
            raise ValueError("surprise windowing requires visual feature samples")
        dimensions = {len(sample.vector) for sample in ordered}
        if len(dimensions) != 1:
            raise ValueError("visual feature dimensions must match")
        if ordered[-1].timestamp_s > duration_s + 1e-6:
            raise ValueError("visual feature sample exceeds video duration")

        start_s = 0.0
        start_index = 0
        change_history: list[float] = []
        windows: list[AdaptiveWindow] = []
        previous = ordered[0]

        for index, sample in enumerate(ordered[1:], start=1):
            change = _cosine_distance(previous.vector, sample.vector)
            previous = sample

            while sample.timestamp_s - start_s >= self.config.max_window_s:
                end_s = min(duration_s, start_s + self.config.max_window_s)
                windows.append(
                    AdaptiveWindow(
                        start_s=start_s,
                        end_s=end_s,
                        boundary_reason="maximum_duration",
                        sample_count=max(1, index - start_index),
                    )
                )
                start_s = end_s
                start_index = index

            enough_history = len(change_history) >= self.config.calibration_history
            threshold = (
                _quantile(change_history, self.config.high_surprise_quantile)
                if enough_history
                else None
            )
            is_high_surprise = threshold is not None and change > threshold + 1e-12
            if (
                is_high_surprise
                and sample.timestamp_s - start_s >= self.config.min_window_s
                and sample.timestamp_s < duration_s
            ):
                windows.append(
                    AdaptiveWindow(
                        start_s=start_s,
                        end_s=sample.timestamp_s,
                        boundary_reason="representation_surprise",
                        sample_count=max(1, index - start_index),
                    )
                )
                start_s = sample.timestamp_s
                start_index = index

            change_history.append(change)
            if len(change_history) > self.config.calibration_history:
                del change_history[0]

        while duration_s - start_s > self.config.max_window_s:
            end_s = start_s + self.config.max_window_s
            windows.append(
                AdaptiveWindow(
                    start_s=start_s,
                    end_s=end_s,
                    boundary_reason="maximum_duration",
                    sample_count=max(1, len(ordered) - start_index),
                )
            )
            start_s = end_s
        if duration_s - start_s > 1e-9:
            windows.append(
                AdaptiveWindow(
                    start_s=start_s,
                    end_s=duration_s,
                    boundary_reason="stream_end",
                    sample_count=max(1, len(ordered) - start_index),
                )
            )
        return tuple(windows)


@dataclass
class SelectStreamWindowProvider:
    """Runnable adapter from a video feature provider to Qwen L1 windows."""

    feature_provider: VisualFeatureProvider
    config: SurpriseWindowConfig = SurpriseWindowConfig()

    @property
    def provider_name(self) -> str:
        return f"selectstream:{self.feature_provider.provider_name}"

    def windows(
        self,
        video_path: str | Path,
        *,
        duration_s: float,
    ) -> list[dict[str, object]]:
        samples = self.feature_provider.sample(
            video_path,
            duration_s=duration_s,
            sample_period_s=self.config.sample_period_s,
        )
        return [
            window.to_dict()
            for window in SurpriseWindowWriter(self.config).segment(
                samples,
                duration_s=duration_s,
            )
        ]


@dataclass(frozen=True)
class OpenCVPerceptualFeatureProvider:
    """Explicit fallback representation for a runnable local smoke.

    The feature is a compact normalized HSV histogram plus low-resolution
    luminance layout.  Research runs should inject the chosen learned visual
    representation provider while preserving the same writer contract.
    """

    provider_name: str = "opencv_hsv_luminance_fallback/v0.1"

    def sample(
        self,
        video_path: str | Path,
        *,
        duration_s: float,
        sample_period_s: float,
    ) -> Sequence[VisualFeatureSample]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("opencv-python and numpy are required") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"video is unreadable: {video_path}")
        samples: list[VisualFeatureSample] = []
        try:
            timestamp_s = 0.0
            while timestamp_s < duration_s + 1e-9:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    timestamp_s += sample_period_s
                    continue
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
                hist = cv2.normalize(hist, None).flatten()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                layout = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
                layout = layout.astype("float32").reshape(-1) / 255.0
                vector = np.concatenate((hist.astype("float32"), layout))
                samples.append(
                    VisualFeatureSample(
                        timestamp_s=min(timestamp_s, duration_s),
                        vector=tuple(float(value) for value in vector),
                        frame_index=int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
                    )
                )
                timestamp_s += sample_period_s
        finally:
            capture.release()
        return samples


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 1e-12 and right_norm <= 1e-12:
        return 0.0
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 1.0
    similarity = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return 1.0 - similarity


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
