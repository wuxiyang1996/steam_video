"""Structured clip schema generation with a multimodal OpenRouter model."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .openrouter_client import OpenRouterClient
from ..schemas import ClipSchemaConfig, ClipSpan

CLIP_SCHEMA_PROMPT = """You convert one video clip span into a structured perception record.

Return JSON only with this shape:
{
  "clip_id": "string",
  "time_span": {"start_s": number, "end_s": number},
  "granularity": "whole|coarse|fine",
  "scene_description": "short grounded scene summary",
  "observable_facts": [
    {"text": "fact grounded in visible or spoken content", "modality": "visual|audio|subtitle|mixed"}
  ],
  "dialogue_spans": [
    {"speaker": "name or unknown", "text": "utterance", "time_span": {"start_s": number, "end_s": number}}
  ],
  "entity_mentions": [
    {"surface_form": "name or object", "entity_type": "person|object|place|other"}
  ],
  "salient_objects": [
    {"surface_form": "object name", "attributes": ["color/material/shape"], "searchable_phrases": ["phrase"]}
  ],
  "place": {"description": "place or setting", "searchable_phrases": ["phrase"]},
  "events": [
    {"description": "timestamped event", "time_span": {"start_s": number, "end_s": number}}
  ],
  "visual_social_cues": [
    {
      "description": "visible gesture/expression/posture/social interaction",
      "cue_type": "expression|gesture|posture|interaction|uncertain",
      "strength": "weak|medium|strong"
    }
  ],
  "cross_clip_cues": [
    {"cue_type": "same_object|same_place|reappears|before_after|unknown", "description": "reusable clue phrase"}
  ],
  "searchable_phrases": ["short phrases useful for later retrieval"],
  "uncertainty": "short note about ambiguous or missing evidence"
}

Rules:
1. Use only information supported by the clip frames or provided subtitle/context text.
2. Do not invent characters, objects, or events.
3. Prefer clue-oriented noun phrases over generic captions: objects, colors, place,
   repeated-looking props, screen text, spoken clues, temporal changes.
4. Include alternate names for visually salient items when grounded (for example
   "iron fence", "metal gate", "white vehicle", "same-looking doorway").
5. For social questions, record only visible social cues: facial expression,
   gaze direction, hesitation-like motion, distance, posture, gesture, or group
   interaction. Do not infer private motives unless the visual evidence is clear.
6. Keep lists short and precise.
7. If nothing is visible, return empty lists and a cautious scene_description.
"""

COMPACT_CLIP_SCHEMA_PROMPT = """Return compact JSON only for one video clip:
{
  "scene_description": "one short grounded sentence",
  "observable_facts": [{"text": "short visible/spoken fact", "modality": "visual|audio|subtitle|mixed"}],
  "dialogue_spans": [],
  "entity_mentions": [{"surface_form": "object/person/place", "entity_type": "person|object|place|other"}],
  "salient_objects": [{"surface_form": "object", "attributes": ["short"], "searchable_phrases": ["short phrase"]}],
  "place": {"description": "short setting", "searchable_phrases": ["short phrase"]},
  "events": [{"description": "short event", "time_span": {"start_s": number, "end_s": number}}],
  "visual_social_cues": [{"description": "visible social cue", "cue_type": "expression|gesture|posture|interaction|uncertain", "strength": "weak|medium|strong"}],
  "cross_clip_cues": [],
  "searchable_phrases": ["short phrase"],
  "uncertainty": "short note"
}
Use only grounded evidence. Keep every string under 80 characters.
"""


def _clip_schema_response_schema() -> dict[str, Any]:
    """OpenRouter response schema for clip perception records."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "video_clip_schema",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "clip_id": {"type": "string"},
                    "time_span": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "start_s": {"type": "number"},
                            "end_s": {"type": "number"},
                        },
                    },
                    "granularity": {"type": "string"},
                    "scene_description": {"type": "string"},
                    "observable_facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "text": {"type": "string"},
                                "modality": {"type": "string"},
                            },
                            "required": ["text"],
                        },
                    },
                    "dialogue_spans": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    "entity_mentions": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    "salient_objects": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    "place": {"type": "object", "additionalProperties": True},
                    "events": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    "visual_social_cues": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    "cross_clip_cues": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    "searchable_phrases": {"type": "array", "items": {"type": "string"}},
                    "uncertainty": {"type": "string"},
                },
                "required": [
                    "scene_description",
                    "observable_facts",
                    "dialogue_spans",
                    "entity_mentions",
                    "salient_objects",
                    "place",
                    "events",
                    "visual_social_cues",
                    "cross_clip_cues",
                    "searchable_phrases",
                    "uncertainty",
                ],
            },
        },
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _dict_items(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            rows.append(item)
        elif item is not None:
            rows.append({"text": _as_string(item)})
    return rows


def _string_items(value: Any) -> list[str]:
    strings = [_as_string(item) for item in _as_list(value)]
    return [item for item in strings if item]


def _normalize_clip_schema_payload(
    payload: dict[str, Any],
    *,
    clip_id: str,
    clip: ClipSpan,
    model: str,
    attempt: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("clip schema payload must be a JSON object")

    normalized: dict[str, Any] = dict(payload)
    normalized["clip_id"] = clip_id
    normalized["time_span"] = clip.to_dict()
    normalized["granularity"] = _as_string(normalized.get("granularity")) or clip.granularity
    normalized["scene_description"] = _as_string(normalized.get("scene_description"))
    normalized["observable_facts"] = _dict_items(normalized.get("observable_facts"))
    normalized["dialogue_spans"] = _dict_items(normalized.get("dialogue_spans"))
    normalized["entity_mentions"] = _dict_items(normalized.get("entity_mentions"))
    normalized["salient_objects"] = _dict_items(normalized.get("salient_objects"))
    normalized["place"] = normalized.get("place") if isinstance(normalized.get("place"), dict) else {}
    normalized["events"] = _dict_items(normalized.get("events"))
    normalized["cross_clip_cues"] = _dict_items(normalized.get("cross_clip_cues"))
    normalized["searchable_phrases"] = _string_items(normalized.get("searchable_phrases"))
    normalized["uncertainty"] = _as_string(normalized.get("uncertainty"))

    for fact in normalized["observable_facts"]:
        fact["text"] = _as_string(fact.get("text") or fact.get("description"))
        fact["modality"] = _as_string(fact.get("modality")) or "visual"
    normalized["observable_facts"] = [fact for fact in normalized["observable_facts"] if fact.get("text")]

    for mention in normalized["entity_mentions"]:
        mention["surface_form"] = _as_string(
            mention.get("surface_form") or mention.get("name") or mention.get("text")
        )
        mention["entity_type"] = _as_string(mention.get("entity_type")) or "other"
    normalized["entity_mentions"] = [
        mention for mention in normalized["entity_mentions"] if mention.get("surface_form")
    ]

    for obj in normalized["salient_objects"]:
        obj["surface_form"] = _as_string(obj.get("surface_form") or obj.get("name") or obj.get("text"))
        obj["attributes"] = _string_items(obj.get("attributes"))
        obj["searchable_phrases"] = _string_items(obj.get("searchable_phrases"))
    normalized["salient_objects"] = [obj for obj in normalized["salient_objects"] if obj.get("surface_form")]

    for event in normalized["events"]:
        event["description"] = _as_string(event.get("description") or event.get("text"))
        if not isinstance(event.get("time_span"), dict):
            event["time_span"] = clip.to_dict()
    normalized["events"] = [event for event in normalized["events"] if event.get("description")]

    place = normalized["place"]
    place["description"] = _as_string(place.get("description"))
    place["searchable_phrases"] = _string_items(place.get("searchable_phrases"))

    has_signal = any(
        [
            normalized["scene_description"],
            normalized["observable_facts"],
            normalized["entity_mentions"],
            normalized["salient_objects"],
            place.get("description"),
            normalized["events"],
            normalized["searchable_phrases"],
        ]
    )
    if not has_signal:
        raise ValueError("clip schema payload contains no usable perception signal")

    normalized["schema_attempt"] = attempt
    normalized["model"] = model
    normalized["producer"] = "qwen_clip_schema"
    return normalized


class QwenClipSchemaProducer:
    """Produce structured clip schemas from segmented video spans."""

    def __init__(self, config: ClipSchemaConfig, client: OpenRouterClient):
        self.config = config
        self.client = client

    def _sample_frame_jpegs(self, video_path: Path, clip: ClipSpan) -> list[str]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("clip schema generation requires opencv-python") from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        frames: list[str] = []
        count = self.config.request_frames
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

    def build_clip_schema(
        self,
        *,
        clip_id: str,
        clip: ClipSpan,
        video_path: Path | None,
        subtitle_context: str | None = None,
        question_context: str | None = None,
    ) -> dict[str, Any]:
        def _content(prompt: str) -> list[dict[str, Any]]:
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        f"{prompt}\n"
                        f"clip_id={clip_id}\n"
                        f"time_span={clip.to_dict()}\n"
                        f"granularity={clip.granularity}\n"
                        f"subtitle_context={subtitle_context or ''}"
                        + (f"\nquestion_context={question_context}" if question_context else "")
                    ),
                }
            ]
            if video_path and video_path.exists():
                for data in sampled_frames:
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
                    )
            return content

        sampled_frames: list[str] = []
        if video_path and video_path.exists():
            sampled_frames = self._sample_frame_jpegs(video_path, clip)

        user_content = _content(CLIP_SCHEMA_PROMPT)
        compact_content = _content(COMPACT_CLIP_SCHEMA_PROMPT)
        last_error: Exception | None = None
        attempts = (
            ("full", user_content, _clip_schema_response_schema()),
            ("compact_retry", compact_content, _clip_schema_response_schema()),
            ("compact_json_object_retry", compact_content, None),
        )
        for attempt_index, (attempt, content, response_format) in enumerate(attempts):
            try:
                payload = self.client.chat_json(
                    [
                        {"role": "system", "content": "You are a grounded video perception annotator."},
                        {"role": "user", "content": content},
                    ],
                    response_format=response_format,
                )
                payload = _normalize_clip_schema_payload(
                    payload,
                    clip_id=clip_id,
                    clip=clip,
                    model=self.config.model,
                    attempt=attempt,
                )
                payload["llm_usage"] = {
                    **(self.client.last_response_metadata or {}),
                    "attempt": attempt,
                    "attempt_index": attempt_index,
                    "compact_retry_count": attempt_index,
                    "sampled_frame_count": len(sampled_frames),
                }
                break
            except Exception as exc:
                last_error = exc
        else:
            payload = {
                "clip_id": clip_id,
                "time_span": clip.to_dict(),
                "granularity": clip.granularity,
                "scene_description": "clip schema generation failed",
                "observable_facts": [],
                "dialogue_spans": [],
                "entity_mentions": [],
                "salient_objects": [],
                "place": {},
                "events": [],
                "cross_clip_cues": [],
                "searchable_phrases": [],
                "uncertainty": "clip schema generation failed",
                "model_error": str(last_error),
                "llm_usage": {
                    **(self.client.last_response_metadata or {}),
                    "attempt": "failed",
                    "compact_retry_count": 2,
                    "sampled_frame_count": len(sampled_frames),
                },
            }
        return payload
