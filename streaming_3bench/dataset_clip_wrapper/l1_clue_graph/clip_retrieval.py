"""Lexical coarse-clip retrieval (M3-style top-k gating without embedding API)."""

from __future__ import annotations

import re
from typing import Any, Literal

from ..schemas import ClipSpan

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "which",
    "does",
    "do",
    "did",
    "in",
    "on",
    "at",
    "to",
    "of",
    "and",
    "or",
    "for",
    "with",
    "from",
    "that",
    "this",
    "it",
    "its",
    "their",
    "they",
    "he",
    "she",
    "his",
    "her",
}


def tokenize(text: str) -> set[str]:
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _segment_overlaps_clip(segment: dict[str, Any], clip: ClipSpan) -> bool:
    span = segment.get("time_span")
    if not span:
        return False
    return not (span["end_s"] < clip.start_s or span["start_s"] > clip.end_s)


def _clip_before_observation(clip: ClipSpan, observation_end_s: float | None) -> bool:
    if observation_end_s is None:
        return True
    return clip.end_s <= observation_end_s + 1e-6


def score_coarse_clip(
    clip: ClipSpan,
    *,
    query_tokens: set[str],
    segments: list[dict[str, Any]],
    clue_interval_boost: float = 2.0,
) -> float:
    """Score a coarse clip by lexical overlap with the query and visible segments."""
    if not query_tokens:
        return 0.0

    score = 0.0
    for segment in segments:
        if not _segment_overlaps_clip(segment, clip):
            continue
        text = segment.get("text") or ""
        if not text:
            continue
        seg_tokens = tokenize(text)
        overlap = len(query_tokens & seg_tokens)
        if overlap <= 0:
            continue
        weight = 1.0
        source_type = segment.get("source_type") or ""
        if source_type in {"clue_interval", "clue_clip", "reasoning_process_step"}:
            weight = clue_interval_boost
        score += overlap * weight

    return score


def retrieve_coarse_clips(
    *,
    coarse_spans: list[ClipSpan],
    query_text: str,
    segments: list[dict[str, Any]] | None = None,
    topk: int = 2,
    threshold: float = 0.0,
    observation_end_s: float | None = None,
    mode: Literal["lexical", "sequential"] = "lexical",
) -> dict[str, Any]:
    """Return top-k coarse clip indices and scores (M3-style retrieve gate)."""
    segments = segments or []
    query_tokens = tokenize(query_text)

    candidates: list[tuple[int, ClipSpan, float]] = []
    visible_coarse: list[tuple[int, ClipSpan]] = []
    for coarse_index, clip in enumerate(coarse_spans):
        if not _clip_before_observation(clip, observation_end_s):
            continue
        visible_coarse.append((coarse_index, clip))
        if mode == "sequential":
            score = float(len(coarse_spans) - coarse_index)
        else:
            score = score_coarse_clip(
                clip,
                query_tokens=query_tokens,
                segments=segments,
            )
        if score >= threshold:
            candidates.append((coarse_index, clip, score))

    fallback_reason = None
    positive_candidates = [item for item in candidates if item[2] > 0]
    if mode == "lexical" and query_tokens and not positive_candidates and visible_coarse:
        fallback_reason = "uniform_probe_no_lexical_match"
        if topk <= 1:
            chosen_positions = [len(visible_coarse) // 2]
        else:
            chosen_positions = [
                min(len(visible_coarse) - 1, max(0, round((rank + 1) * len(visible_coarse) / (topk + 1)) - 1))
                for rank in range(topk)
            ]
        seen_positions: set[int] = set()
        candidates = []
        for position in chosen_positions:
            if position in seen_positions:
                continue
            seen_positions.add(position)
            coarse_index, clip = visible_coarse[position]
            candidates.append((coarse_index, clip, 0.0))

    if not candidates and visible_coarse:
        fallback_reason = fallback_reason or "sequential_visible_prefix"
        # Fallback: first visible coarse clips (M3 always returns something searchable).
        for coarse_index, clip in visible_coarse:
            candidates.append((coarse_index, clip, 0.0))
            if len(candidates) >= topk:
                break

    candidates.sort(key=lambda item: item[2], reverse=True)
    selected = candidates[:topk]

    return {
        "mode": mode,
        "topk": topk,
        "threshold": threshold,
        "query_tokens": sorted(query_tokens),
        "selected_coarse_indices": [item[0] for item in selected],
        "fallback_reason": fallback_reason,
        "scores": [
            {
                "coarse_index": item[0],
                "clip_index": item[1].clip_index,
                "start_s": item[1].start_s,
                "end_s": item[1].end_s,
                "score": item[2],
            }
            for item in selected
        ],
    }
