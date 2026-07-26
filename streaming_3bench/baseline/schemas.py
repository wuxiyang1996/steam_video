"""JSONL-friendly standard records for baseline video QA retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


InputMode = Literal["frames", "video_clip", "text_memory"]


@dataclass(frozen=True)
class VideoQAExample:
    """One question over one video under a stated visibility policy."""

    example_id: str
    dataset: str
    split: str
    video_id: str
    video_path: str
    question_text: str
    options: list[dict[str, str]]
    answer_label: str | None
    answer_text: str | None
    visible_until_s: float | None
    streaming_mode: str
    input_mode: InputMode = "video_clip"
    question_embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoClipRecord:
    """One stable clip span from a canonical wrapper example."""

    row_id: int
    example_id: str
    dataset: str
    video_id: str
    clip_id: str
    video_path: str
    start_s: float
    end_s: float
    granularity: str
    visible_until_s: float | None
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedClip:
    """One FAISS retrieval result."""

    rank: int
    score: float
    row_id: int
    clip: VideoClipRecord

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["clip"] = self.clip.to_dict()
        return payload


def visible_until_from_canonical(example: dict[str, Any], default_videomme_cutoff_s: float | None = 60.0) -> float | None:
    """Infer the baseline streaming cutoff from a canonical wrapper example."""

    question = example.get("question") or {}
    video = example.get("video") or {}
    duration = video.get("duration_s")
    duration_s = float(duration) if isinstance(duration, (int, float)) else None
    anchor = question.get("time_anchor_s")
    if isinstance(anchor, (int, float)):
        visible = float(anchor)
    elif example.get("dataset") == "videomme":
        visible = default_videomme_cutoff_s if default_videomme_cutoff_s is not None else duration_s
    else:
        visible = duration_s
    if visible is not None and duration_s is not None:
        visible = max(0.0, min(visible, duration_s))
    return visible


def qa_example_from_canonical(
    example: dict[str, Any],
    *,
    input_mode: InputMode = "video_clip",
    default_videomme_cutoff_s: float | None = 60.0,
    question_embedding: list[float] | None = None,
) -> VideoQAExample:
    question = example.get("question") or {}
    answer = question.get("answer") or {}
    video = example.get("video") or {}
    return VideoQAExample(
        example_id=str(example.get("example_id") or ""),
        dataset=str(example.get("dataset") or ""),
        split=str(example.get("split") or ""),
        video_id=str(video.get("video_id") or ""),
        video_path=str(video.get("primary_path") or ""),
        question_text=str(question.get("question_text") or ""),
        options=list(question.get("options") or []),
        answer_label=answer.get("label"),
        answer_text=answer.get("text"),
        visible_until_s=visible_until_from_canonical(example, default_videomme_cutoff_s),
        streaming_mode=str((example.get("metadata") or {}).get("video_regime") or example.get("task_family") or ""),
        input_mode=input_mode,
        question_embedding=question_embedding,
        metadata={
            "task_family": example.get("task_family"),
            "question_id": question.get("question_id"),
            "answer_format": question.get("answer_format"),
        },
    )


def canonical_video_key(example: dict[str, Any]) -> str:
    """Stable per-video key. Prefer path so StreamingBench collisions stay unique."""

    dataset = str(example.get("dataset") or "")
    video = example.get("video") or {}
    path = str(video.get("primary_path") or "")
    video_id = str(video.get("video_id") or "")
    return f"{dataset}|{path or video_id}"


def clip_records_from_canonical(
    example: dict[str, Any],
    *,
    start_row_id: int = 0,
    default_videomme_cutoff_s: float | None = 60.0,
    embeddings: list[list[float]] | None = None,
    text_mode: str = "metadata_question",
    apply_visible_cutoff: bool = True,
    bind_example_id: bool = True,
) -> list[VideoClipRecord]:
    """Convert canonical `video.derived_clips` into baseline clip records.

    For per-video indexes, set ``apply_visible_cutoff=False`` and
    ``bind_example_id=False`` so one video ref is shared across QAs and
    visibility is enforced only at query time.
    """

    qa = qa_example_from_canonical(example, default_videomme_cutoff_s=default_videomme_cutoff_s)
    records: list[VideoClipRecord] = []
    clips = ((example.get("video") or {}).get("derived_clips") or [])
    for clip in clips:
        span = clip.get("source_span") or {}
        start_s = float(span.get("start_s", 0.0))
        end_s = float(span.get("end_s", start_s))
        if apply_visible_cutoff and qa.visible_until_s is not None:
            if start_s > qa.visible_until_s:
                continue
            end_s = min(end_s, qa.visible_until_s)
        if end_s <= start_s:
            continue
        clip_id = str(clip.get("clip_id") or f"{qa.video_id}:{len(records)}")
        granularity = str(clip.get("granularity") or "unknown")
        text = clip_text_from_canonical(
            clip,
            qa=qa,
            clip_id=clip_id,
            start_s=start_s,
            end_s=end_s,
            granularity=granularity,
            text_mode=text_mode,
        )
        embedding = embeddings[len(records)] if embeddings is not None and len(records) < len(embeddings) else None
        records.append(
            VideoClipRecord(
                row_id=start_row_id + len(records),
                example_id=qa.example_id if bind_example_id else "",
                dataset=qa.dataset,
                video_id=qa.video_id,
                clip_id=clip_id,
                video_path=str(clip.get("path") or qa.video_path),
                start_s=start_s,
                end_s=end_s,
                granularity=granularity,
                visible_until_s=qa.visible_until_s if apply_visible_cutoff else None,
                text=text,
                embedding=embedding,
                metadata={
                    "parent_index": clip.get("parent_index"),
                    "index_granularity": "example" if bind_example_id else "video",
                    "video_key": canonical_video_key(example),
                },
            )
        )
    return records


def select_canonical_videos(
    examples: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Deduplicate canonical QA examples down to one representative per video.

    Prefers the example with the most derived clips / latest clip end time so the
    per-video ref covers the longest available span.
    """

    best: dict[str, tuple[tuple[int, float], dict[str, Any]]] = {}
    for example in examples:
        key = canonical_video_key(example)
        clips = ((example.get("video") or {}).get("derived_clips") or [])
        max_end = 0.0
        for clip in clips:
            span = clip.get("source_span") or {}
            try:
                max_end = max(max_end, float(span.get("end_s", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        score = (len(clips), max_end)
        prev = best.get(key)
        if prev is None or score > prev[0]:
            best[key] = (score, example)
    return sorted((key, payload[1]) for key, payload in best.items())


def clip_text_from_canonical(
    clip: dict[str, Any],
    *,
    qa: VideoQAExample,
    clip_id: str,
    start_s: float,
    end_s: float,
    granularity: str,
    text_mode: str,
) -> str:
    captions = _clip_caption_texts(clip)
    metadata_parts = [
        f"dataset={qa.dataset}",
        f"video={qa.video_id}",
        f"clip={clip_id}",
        f"time={start_s:.2f}-{end_s:.2f}s",
        f"granularity={granularity}",
    ]
    if text_mode == "metadata_question":
        parts = [*metadata_parts, f"example={qa.example_id}", f"question={qa.question_text}"]
    elif text_mode == "metadata":
        parts = metadata_parts
    elif text_mode == "caption":
        parts = captions
    elif text_mode == "caption_metadata":
        parts = [*metadata_parts, *captions]
    else:
        raise ValueError(f"unsupported clip text_mode: {text_mode}")
    return " ".join(part for part in parts if part).strip()

def _clip_caption_texts(clip: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in (
        "caption",
        "caption_text",
        "summary",
        "video_summary",
        "transcript",
        "subtitle",
        "text",
        "description",
    ):
        _append_text_value(texts, clip.get(key))
    for key in ("captions", "caption_span", "subtitle_span", "transcript_span", "annotations"):
        _append_text_value(texts, clip.get(key))
    metadata = clip.get("metadata")
    if isinstance(metadata, dict):
        for key in ("caption", "caption_text", "summary", "transcript", "subtitle", "description"):
            _append_text_value(texts, metadata.get(key))
    return texts


def _append_text_value(texts: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            texts.append(stripped)
        return
    if isinstance(value, dict):
        for key in ("text", "caption", "summary", "transcript", "subtitle", "description"):
            _append_text_value(texts, value.get(key))
        return
    if isinstance(value, list):
        for item in value:
            _append_text_value(texts, item)
