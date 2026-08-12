"""Leakage-safe deterministic data splitting without neural dependencies."""

from __future__ import annotations

import hashlib
from typing import Sequence


def video_disjoint_indices(
    video_ids: Sequence[str],
    *,
    validation_fraction: float = 0.2,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")
    videos = sorted(
        set(video_ids),
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(videos) < 2:
        raise ValueError("video-disjoint split requires at least two videos")
    count = max(1, min(len(videos) - 1, round(len(videos) * validation_fraction)))
    validation_videos = set(videos[:count])
    train = tuple(index for index, value in enumerate(video_ids) if value not in validation_videos)
    validation = tuple(index for index, value in enumerate(video_ids) if value in validation_videos)
    return train, validation

