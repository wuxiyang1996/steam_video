"""Clip segmentation for short, long, and streaming video regimes."""

from __future__ import annotations

from typing import Literal

from ..schemas import ClipPolicyConfig, ClipSpan, VideoRegime

FineExpansion = Literal["all", "none", "selected"]


def _sliding_windows(
    *,
    duration_s: float,
    window_s: float,
    overlap_s: float,
    observation_end_s: float | None,
    granularity: str,
    parent_index: int | None = None,
    start_index: int = 0,
) -> list[ClipSpan]:
    if window_s <= 0 or overlap_s >= window_s:
        raise ValueError("window_s must be positive and overlap_s must be smaller than window_s")

    limit = max(duration_s, window_s)
    cursor = 0.0
    spans: list[ClipSpan] = []
    index = start_index
    while cursor < limit:
        end = min(cursor + window_s, limit)
        if observation_end_s is None or end <= observation_end_s + 1e-6:
            spans.append(
                ClipSpan(
                    start_s=cursor,
                    end_s=end,
                    granularity=granularity,  # type: ignore[arg-type]
                    parent_index=parent_index,
                    clip_index=index,
                )
            )
            index += 1
        cursor += window_s - overlap_s
    return spans


def _expand_fine_within_coarse(
    coarse: list[ClipSpan],
    policy: ClipPolicyConfig,
    *,
    observation_end_s: float | None,
    parent_indices: list[int] | None = None,
) -> list[ClipSpan]:
    fine: list[ClipSpan] = []
    next_index = len(coarse)
    targets = parent_indices if parent_indices is not None else list(range(len(coarse)))
    for parent_idx in targets:
        if parent_idx < 0 or parent_idx >= len(coarse):
            continue
        parent = coarse[parent_idx]
        local = _sliding_windows(
            duration_s=parent.end_s - parent.start_s,
            window_s=policy.fine_window_s,
            overlap_s=min(policy.overlap_s, policy.fine_window_s / 2),
            observation_end_s=observation_end_s,
            granularity="fine",
            parent_index=parent_idx,
            start_index=next_index,
        )
        for span in local:
            span.start_s += parent.start_s
            span.end_s = min(span.end_s + parent.start_s, parent.end_s)
            if span.end_s <= span.start_s:
                continue
            if observation_end_s is not None and span.end_s > observation_end_s + 1e-6:
                continue
            fine.append(span)
        next_index += len(local)
    return fine


def segment_video(
    duration_s: float,
    policy: ClipPolicyConfig,
    *,
    regime: VideoRegime | None = None,
    fine_expansion: FineExpansion = "all",
    selected_coarse_indices: list[int] | None = None,
) -> list[ClipSpan]:
    """Return ordered clip spans for a video under the given policy."""
    if duration_s <= 0:
        duration_s = float(policy.duration_s or 0.0)
    observation_end_s = policy.observation_end_s
    if regime == VideoRegime.STREAMING and observation_end_s is None:
        observation_end_s = duration_s

    strategy = policy.strategy
    if strategy == "whole_video":
        end_s = duration_s
        if observation_end_s is not None:
            end_s = min(end_s, observation_end_s)
        whole = [ClipSpan(start_s=0.0, end_s=max(end_s, 0.0), granularity="whole", clip_index=0)]
        if regime == VideoRegime.SHORT:
            fine = _sliding_windows(
                duration_s=duration_s,
                window_s=policy.window_s,
                overlap_s=policy.overlap_s,
                observation_end_s=observation_end_s,
                granularity="fine",
                start_index=1,
            )
            return whole + fine
        return whole

    if strategy == "fixed_window":
        return _sliding_windows(
            duration_s=duration_s,
            window_s=policy.window_s,
            overlap_s=policy.overlap_s,
            observation_end_s=observation_end_s,
            granularity="fine",
        )

    if strategy == "hierarchical":
        coarse = _sliding_windows(
            duration_s=duration_s,
            window_s=policy.coarse_window_s,
            overlap_s=policy.overlap_s,
            observation_end_s=observation_end_s,
            granularity="coarse",
        )
        if fine_expansion == "none":
            return coarse
        parent_indices = None if fine_expansion == "all" else selected_coarse_indices
        fine = _expand_fine_within_coarse(
            coarse,
            policy,
            observation_end_s=observation_end_s,
            parent_indices=parent_indices,
        )
        return coarse + fine

    if strategy == "coarse_only":
        return _sliding_windows(
            duration_s=duration_s,
            window_s=policy.coarse_window_s,
            overlap_s=policy.overlap_s,
            observation_end_s=observation_end_s,
            granularity="coarse",
        )

    raise ValueError(f"unsupported clip policy strategy: {strategy}")


def segment_coarse_index(
    duration_s: float,
    policy: ClipPolicyConfig,
    *,
    regime: VideoRegime | None = None,
) -> list[ClipSpan]:
    """Return coarse index spans only (M3-style 30–45s windows)."""
    return segment_video(duration_s, policy, regime=regime, fine_expansion="none")


def segment_perception_clips(
    duration_s: float,
    policy: ClipPolicyConfig,
    *,
    regime: VideoRegime | None = None,
    selected_coarse_indices: list[int],
) -> list[ClipSpan]:
    """Expand fine windows inside retrieved coarse candidates only."""
    return segment_video(
        duration_s,
        policy,
        regime=regime,
        fine_expansion="selected",
        selected_coarse_indices=selected_coarse_indices,
    )
