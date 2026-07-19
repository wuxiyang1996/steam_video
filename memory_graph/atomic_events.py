"""Structured extraction of atomic L1.5 events from grounded L1 observations."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Protocol

from .contracts import AtomicEvent, EntityMention, StateAssertion
from .types import MemoryNode, TimeSpan


FORBIDDEN_INFERENCE_PATTERNS = (
    r"\bbecause\b",
    r"\btherefore\b",
    r"\bso that\b",
    r"\bin order to\b",
    r"\bcauses?\b",
    r"\benables?\b",
    r"\bexplains?\b",
    r"\bwants? to\b",
    r"\bdecides? to\b",
)


class ChatJSONClient(Protocol):
    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]: ...


class StructuredAtomicEventExtractor:
    """Use a JSON-capable model to propose events; deterministic checks admit them."""

    def __init__(
        self,
        client: ChatJSONClient,
        *,
        model: str,
        max_attempts: int = 3,
    ) -> None:
        self.client = client
        self.model = model
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts

    def extract(self, nodes: list[MemoryNode]) -> list[AtomicEvent]:
        if not nodes:
            return []
        extracted: list[AtomicEvent] = []
        for node in sorted(
            nodes,
            key=lambda value: (
                value.time_span.start_s,
                value.time_span.end_s,
                value.node_id,
            ),
        ):
            local_events = self._extract_node(node)
            for index, event in enumerate(local_events, start=1):
                stable_id = f"atomic:{node.node_id}:{index:03d}"
                extracted.append(replace(event, event_id=stable_id))
        return sorted(
            extracted,
            key=lambda event: (
                event.time_span.start_s,
                event.time_span.end_s,
                event.event_id,
            ),
        )

    def _extract_node(self, node: MemoryNode) -> list[AtomicEvent]:
        """Bound one response to one L1 observation to avoid malformed long JSON."""
        prompt = _build_prompt([node])
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            if attempt == 1:
                retry_instruction = ""
            elif attempt < self.max_attempts:
                retry_instruction = (
                    "\n\nPrevious output was invalid. Return one complete JSON object only; "
                    "do not truncate, use markdown, or include trailing commentary."
                )
            else:
                retry_instruction = (
                    "\n\nCOMPACT FALLBACK: return at most 4 key events. Set participants "
                    "and states to empty arrays. Keep every predicate under 12 words. "
                    "Return one complete JSON object only."
                )
            try:
                payload = self.client.chat_json(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Extract only directly observed atomic video events. "
                                "Return strict JSON and make no causal or motivational inference."
                            ),
                        },
                        {"role": "user", "content": prompt + retry_instruction},
                    ]
                )
                return parse_atomic_events(payload, nodes=[node], model=self.model)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise ValueError(
            f"atomic-event extraction failed after {self.max_attempts} attempts: "
            f"{last_error}"
        ) from last_error


def parse_atomic_events(
    payload: dict[str, Any],
    *,
    nodes: list[MemoryNode],
    model: str,
) -> list[AtomicEvent]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("atomic-event response must contain an events list")

    evidence_by_id = {node.node_id: node for node in nodes}
    video_ids = {node.video_id for node in nodes}
    if len(video_ids) != 1:
        raise ValueError("atomic-event extraction requires nodes from exactly one video")
    expected_video_id = next(iter(video_ids))
    event_ids: set[str] = set()
    parsed: list[AtomicEvent] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise ValueError(f"events[{index}] must be an object")
        event_id = str(raw.get("event_id") or "")
        if event_id in event_ids:
            raise ValueError(f"duplicate atomic event id: {event_id}")
        event_ids.add(event_id)

        predicate = str(raw.get("predicate") or "").strip()
        issues = atomicity_issues(predicate)
        if issues:
            raise ValueError(f"event {event_id!r} failed observation checks: {issues}")

        evidence_refs = tuple(str(value) for value in raw.get("evidence_refs") or [])
        unknown_refs = set(evidence_refs) - set(evidence_by_id)
        if unknown_refs:
            raise ValueError(f"event {event_id!r} cites unknown L1 nodes: {sorted(unknown_refs)}")

        time_span = TimeSpan.from_dict(_require_dict(raw.get("time_span"), "time_span"))
        if evidence_refs and not _span_is_grounded(
            time_span,
            [evidence_by_id[ref].time_span for ref in evidence_refs],
        ):
            raise ValueError(f"event {event_id!r} time span falls outside its L1 evidence")

        participants = tuple(
            _parse_participant(value, event_id=event_id)
            for value in _require_list(raw.get("participants", []), "participants")
        )
        mention_ids = [participant.mention_id for participant in participants]
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError(f"event {event_id!r} contains duplicate mention ids")
        parsed_states = tuple(
            _parse_state(value)
            for value in _require_list(raw.get("states", []), "states")
        )
        # A malformed local link must not create a state edge. Preserve the
        # observable event while conservatively dropping unlinked states.
        states = tuple(
            state for state in parsed_states if state.mention_id in set(mention_ids)
        )
        for participant in participants:
            unknown_grounding = set(participant.grounding_refs) - set(evidence_refs)
            if unknown_grounding:
                raise ValueError(
                    f"event {event_id!r} participant grounding is outside event evidence: "
                    f"{sorted(unknown_grounding)}"
                )
        video_id = str(raw.get("video_id") or expected_video_id)
        if video_id != expected_video_id:
            raise ValueError(f"event {event_id!r} references the wrong video")

        parsed.append(
            AtomicEvent(
                event_id=event_id,
                video_id=video_id,
                time_span=time_span,
                predicate=predicate,
                evidence_refs=evidence_refs,
                confidence=float(raw.get("confidence")),
                participants=participants,
                states=states,
                provenance={
                    "producer": "memory_graph.atomic_events.StructuredAtomicEventExtractor",
                    "model": model,
                    "layer": "L1.5",
                },
            )
        )
    return sorted(
        parsed,
        key=lambda event: (event.time_span.start_s, event.time_span.end_s, event.event_id),
    )


def atomicity_issues(predicate: str) -> list[str]:
    """Reject explicit inference language and obviously compound event text."""
    issues: list[str] = []
    normalized = " ".join(predicate.strip().split())
    if not normalized:
        return ["empty predicate"]
    if len(normalized.split()) > 24:
        issues.append("predicate exceeds 24 words")
    if ";" in normalized or " and then " in normalized.lower():
        issues.append("predicate appears to contain multiple ordered events")
    for pattern in FORBIDDEN_INFERENCE_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            issues.append(f"contains inferred causal or intentional language matching {pattern}")
    return issues


def _build_prompt(nodes: list[MemoryNode]) -> str:
    observations = [
        {
            "node_id": node.node_id,
            "video_id": node.video_id,
            "time_span": {
                "start_s": node.time_span.start_s,
                "end_s": node.time_span.end_s,
            },
            "text": node.text,
            "visibility": node.metadata.get("visibility"),
        }
        for node in nodes
    ]
    return f"""
Split each grounded L1 observation into the smallest directly observed events.

L1 observations:
{json.dumps(observations, indent=2)}

Rules:
1. One event contains one visible action or one visible state assertion.
2. Split sequential actions into separate events.
3. Do not infer causes, purposes, motivations, hidden identities, or unseen outcomes.
4. Preserve uncertainty in confidence instead of inventing detail.
5. Every event must cite one or more supplied node_id values in evidence_refs.
6. Event time spans must stay inside the union of their cited L1 observations.
7. Entity mentions are local. Do not assert cross-event identity.
8. State assertions may describe only a state visible in the cited observation.
9. Return at most 8 salient atomic events for this observation. Prefer key
   actions and state changes over exhaustive micro-actions.
10. Use at most 3 participants and 2 states per event. Omit uncertain states
    instead of expanding the output.
11. Keep predicate, surface, attribute, and value strings concise.

Return JSON:
{{
  "events": [
    {{
      "event_id": "atomic:<stable local id>",
      "video_id": "video id",
      "time_span": {{"start_s": 0.0, "end_s": 1.0}},
      "predicate": "one concise observed action or state",
      "evidence_refs": ["L1 node id"],
      "confidence": 0.0,
      "participants": [
        {{
          "mention_id": "local mention id",
          "role": "agent|patient|instrument|location|other",
          "entity_type": "person|object|place|other",
          "surface": "grounded description",
          "confidence": 0.0,
          "grounding_refs": ["L1 node id"]
        }}
      ],
      "states": [
        {{
          "mention_id": "local mention id",
          "attribute": "visible attribute",
          "value": "visible value",
          "polarity": "positive|negative",
          "confidence": 0.0
        }}
      ]
    }}
  ]
}}
""".strip()


def _parse_participant(payload: Any, *, event_id: str) -> EntityMention:
    value = _require_dict(payload, "participant")
    return EntityMention(
        mention_id=str(value.get("mention_id") or ""),
        role=str(value.get("role") or ""),
        entity_type=str(value.get("entity_type") or ""),
        surface=str(value.get("surface") or ""),
        confidence=float(value.get("confidence")),
        grounding_refs=tuple(str(ref) for ref in value.get("grounding_refs") or []),
    )


def _parse_state(payload: Any) -> StateAssertion:
    value = _require_dict(payload, "state")
    return StateAssertion(
        mention_id=str(value.get("mention_id") or ""),
        attribute=str(value.get("attribute") or ""),
        value=str(value.get("value") or ""),
        polarity=str(value.get("polarity") or "positive"),
        confidence=float(value.get("confidence")),
    )


def _span_is_grounded(event_span: TimeSpan, evidence_spans: list[TimeSpan]) -> bool:
    if not evidence_spans:
        return False
    return any(
        event_span.start_s >= span.start_s - 1e-6
        and event_span.end_s <= span.end_s + 1e-6
        for span in evidence_spans
    )


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value
