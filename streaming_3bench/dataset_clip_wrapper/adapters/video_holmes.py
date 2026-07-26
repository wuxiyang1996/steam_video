"""Video-Holmes adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .base import DatasetAdapter, RawDatasetItem


def parse_time_range(value: str | None) -> dict[str, float] | None:
  if not value:
    return None
  value = value.strip().replace("：", ":").replace("；", ":").replace(";", ":")
  if "-" in value:
    start, end = value.split("-", 1)
  else:
    start = end = value

  def to_seconds(part: str) -> float | None:
    part = part.strip()
    if not part:
      return None
    try:
      pieces = [float(p) for p in part.split(":") if p.strip()]
    except ValueError:
      return None
    if not pieces:
      return None
    if len(pieces) == 3:
      return pieces[0] * 3600 + pieces[1] * 60 + pieces[2]
    if len(pieces) == 2:
      return pieces[0] * 60 + pieces[1]
    return pieces[0]

  start_s = to_seconds(start)
  end_s = to_seconds(end)
  if start_s is None and end_s is None:
    return None
  if start_s is None:
    start_s = end_s
  if end_s is None:
    end_s = start_s
  if start_s is None or end_s is None:
    return None
  if end_s < start_s:
    start_s, end_s = end_s, start_s
  if start_s == end_s:
    end_s += 1.0
  return {"start_s": start_s, "end_s": end_s}


class VideoHolmesAdapter(DatasetAdapter):
  name = "video_holmes"

  def _annotation_path(self, video_id: str) -> Path | None:
    benchmark = self.dataset_root / "Video-Holmes" / "Benchmark"
    for folder in ("annotations", "annotation_training"):
      path = benchmark / folder / f"{video_id}.json"
      if path.exists():
        return path
    return None

  def _load_annotation(self, video_id: str) -> dict[str, Any]:
    path = self._annotation_path(video_id)
    if not path:
      return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
      return payload[0]
    return payload if isinstance(payload, dict) else {}

  def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
    benchmark = self.dataset_root / "Video-Holmes" / "Benchmark"
    qa_path = benchmark / f"{self.split}_Video-Holmes.json"
    records = json.loads(qa_path.read_text(encoding="utf-8"))
    count = 0
    for row in records:
      video_id = row["video ID"]
      qid = str(row["Question ID"])
      example_id = f"video_holmes:{self.split}:{video_id}:q{qid}"
      video_path = benchmark / "videos_cropped" / f"{video_id}.mp4"
      if not video_path.exists():
        video_path = None
      annotation = self._load_annotation(video_id)
      segments = annotation.get("Segment Description") or annotation.get("SegmentDescription") or []
      inference = annotation.get("Inference Shots") or annotation.get("InferenceScenes") or []
      relationships = annotation.get("Key Relationships") or annotation.get("KeyRelationships") or []

      annotation_segments: list[dict[str, Any]] = []
      evidence_seeds: list[dict[str, Any]] = []
      for i, seg in enumerate(segments, start=1):
        span = parse_time_range(seg.get("TimeRange"))
        text = seg.get("Description", "")
        annotation_segments.append(
          {
            "segment_id": f"vh_seg_{i:03d}",
            "source_type": "segment_description",
            "time_span": span,
            "text": text,
            "provenance": {"field": "Segment Description"},
          }
        )
        evidence_seeds.append(
          {
            "evidence_id": f"ev:vh:seg:{i:03d}",
            "source_type": "segment_description",
            "time_span": span,
            "text": text,
            "trust_level": "gold",
            "provenance": {"source_field": "Segment Description"},
          }
        )
      for i, shot in enumerate(inference, start=1):
        span = parse_time_range(shot.get("Time"))
        text = shot.get("Clue", "")
        if shot.get("Conclusion"):
          text = f"{text} Conclusion: {shot['Conclusion']}"
        annotation_segments.append(
          {
            "segment_id": f"vh_inf_{i:03d}",
            "source_type": "inference_shot",
            "time_span": span,
            "text": text,
            "provenance": {"field": "Inference Shots"},
          }
        )
        evidence_seeds.append(
          {
            "evidence_id": f"ev:vh:inf:{i:03d}",
            "source_type": "inference_shot",
            "time_span": span,
            "text": text,
            "trust_level": "gold",
            "provenance": {"source_field": "Inference Shots"},
          }
        )
      for i, rel in enumerate(relationships, start=1):
        text = " ".join(str(rel.get(k, "")) for k in ("Combination", "Relation", "Reason") if rel.get(k))
        if not text or text.lower() == "none none":
          continue
        annotation_segments.append(
          {
            "segment_id": f"vh_rel_{i:03d}",
            "source_type": "key_relationship",
            "text": text,
            "provenance": {"field": "Key Relationships"},
          }
        )
        evidence_seeds.append(
          {
            "evidence_id": f"ev:vh:rel:{i:03d}",
            "source_type": "key_relationship",
            "text": text,
            "trust_level": "strong",
            "provenance": {"source_field": "Key Relationships"},
          }
        )

      options = row.get("Options") or {}
      answer_label = row.get("Answer")
      yield RawDatasetItem(
        dataset=self.name,
        example_id=example_id,
        split=self.split,
        task_family="short_video_social_causal_qa",
        video_id=video_id,
        video_path=video_path,
        duration_s=None,
        question={
          "question_id": qid,
          "question_text": row.get("Question", ""),
          "question_type": row.get("Question Type"),
          "options": [{"label": k, "text": v} for k, v in options.items()],
          "answer": {"label": answer_label, "text": options.get(answer_label)},
          "answer_format": "multiple_choice",
        },
        subtitle_paths=[],
        annotation_segments=annotation_segments,
        evidence_seeds=evidence_seeds,
        hidden_supervision_sources=[
          "official_answer",
          "segment_annotations",
          "inference_shots",
          "key_relationships",
        ],
        raw_source_refs=[
          {
            "source_name": f"{self.split}_Video-Holmes.json",
            "source_item_id": example_id,
            "fields_used": ["Question", "Options", "Answer", "Explanation"],
          }
        ],
        metadata={"explanation": row.get("Explanation")},
      )
      count += 1
      if limit is not None and count >= limit:
        break
