"""Deterministically project Video_Skills L1 neighborhoods into event slots."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from .identity_tracks import IdentityTrackReport, build_identity_tracks
from .identity_verifier import IdentityVerificationReport, verify_identity_candidates
from .types import MemoryNode


IDENTITY_EDGE_TYPES = {"same_entity", "same_object", "reappears"}
ENTITY_NODE_TYPES = {"entity_mention", "entity"}
EVENT_NODE_TYPES = {"event", "dialogue_span"}


@dataclass(frozen=True)
class L1StructuralizationReport:
    event_count: int
    events_with_participants: int
    events_with_states: int
    participant_count: int
    state_count: int
    unresolved_event_ids: tuple[str, ...]
    identity_tracks: IdentityTrackReport
    identity_verification: IdentityVerificationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "events_with_participants": self.events_with_participants,
            "events_with_states": self.events_with_states,
            "participant_count": self.participant_count,
            "state_count": self.state_count,
            "participant_coverage": (
                self.events_with_participants / self.event_count
                if self.event_count
                else None
            ),
            "state_coverage": (
                self.events_with_states / self.event_count
                if self.event_count
                else None
            ),
            "unresolved_event_ids": list(self.unresolved_event_ids),
            "identity_tracks": self.identity_tracks.to_dict(),
            "identity_verification": self.identity_verification.to_dict(),
            "method": "deterministic_video_skills_l1_subgraph_projection",
        }


def structuralize_video_skills_l1(
    graph: dict[str, Any],
    nodes: list[MemoryNode],
) -> tuple[list[MemoryNode], L1StructuralizationReport]:
    """Attach only graph-grounded entities/states; never infer them with an LLM."""

    raw_nodes = [
        value for value in graph.get("nodes") or [] if isinstance(value, dict)
    ]
    raw_by_id = {
        str(value.get("node_id")): value
        for value in raw_nodes
        if value.get("node_id")
    }
    edges = [
        value for value in graph.get("edges") or [] if isinstance(value, dict)
    ]
    verified_edges, verification_report = verify_identity_candidates(raw_by_id, edges)
    components, identity_report = build_identity_tracks(raw_by_id, verified_edges)
    component_members: dict[str, list[dict[str, Any]]] = {}
    for node_id, component_id in components.items():
        component_members.setdefault(component_id, []).append(raw_by_id[node_id])

    nodes_by_clip: dict[str, list[dict[str, Any]]] = {}
    identity_endpoints_by_clip: dict[str, list[dict[str, Any]]] = {}
    identity_endpoint_ids = {
        str(edge.get(endpoint))
        for edge in edges
        if str(edge.get("edge_type") or "") in IDENTITY_EDGE_TYPES
        for endpoint in ("src", "dst")
        if edge.get(endpoint)
    }
    for raw in raw_nodes:
        clip_id = str(raw.get("clip_id") or "")
        if not clip_id:
            continue
        nodes_by_clip.setdefault(clip_id, []).append(raw)
        if str(raw.get("node_id") or "") in identity_endpoint_ids:
            identity_endpoints_by_clip.setdefault(clip_id, []).append(raw)

    enriched: list[MemoryNode] = []
    unresolved: list[str] = []
    participant_total = 0
    state_total = 0
    event_count = 0
    events_with_participants = 0
    events_with_states = 0
    for node in nodes:
        source_id = str(node.source_node_id or "")
        source = raw_by_id.get(source_id)
        if source is None or str(source.get("node_type") or "") not in EVENT_NODE_TYPES:
            enriched.append(node)
            continue
        event_count += 1
        clip_id = str(source.get("clip_id") or "")
        clip_nodes = nodes_by_clip.get(clip_id, [])
        entity_candidates = [
            value
            for value in clip_nodes
            if str(value.get("node_type") or "") in ENTITY_NODE_TYPES
        ]
        if not entity_candidates:
            entity_candidates = identity_endpoints_by_clip.get(clip_id, [])
        participants = _participants_from_candidates(
            entity_candidates,
            components=components,
            component_members=component_members,
            event_ref=node.node_id,
            event_text=_text(source),
        )
        states = _states_for_event(
            clip_nodes,
            participants=participants,
            event_ref=node.node_id,
            event_text=_text(source),
        )
        metadata = {
            **node.metadata,
            "participants": participants,
            "states": states,
            "l1_structuralization": {
                "status": "grounded" if participants else "unresolved",
                "source_event_id": source_id,
                "source_clip_id": clip_id,
                "participant_source_ids": [
                    source_id
                    for participant in participants
                    for source_id in participant.get("source_l1_node_ids") or []
                ],
                "state_source_ids": [
                    str(state.get("source_l1_node_id") or "") for state in states
                ],
            },
        }
        enriched.append(replace(node, metadata=metadata))
        participant_total += len(participants)
        state_total += len(states)
        events_with_participants += bool(participants)
        events_with_states += bool(states)
        if not participants:
            unresolved.append(node.node_id)

    return enriched, L1StructuralizationReport(
        event_count=event_count,
        events_with_participants=events_with_participants,
        events_with_states=events_with_states,
        participant_count=participant_total,
        state_count=state_total,
        unresolved_event_ids=tuple(unresolved),
        identity_tracks=identity_report,
        identity_verification=verification_report,
    )


def _participants_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    components: dict[str, str],
    component_members: dict[str, list[dict[str, Any]]],
    event_ref: str,
    event_text: str,
) -> list[dict[str, Any]]:
    by_component: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        source_id = str(candidate.get("node_id") or "")
        if not source_id:
            continue
        component_id = components.get(source_id, source_id)
        members = component_members.get(component_id, [candidate])
        explicit_entities = [
            value
            for value in members
            if str(value.get("node_type") or "") in ENTITY_NODE_TYPES
            and _text(value)
        ]
        surface_node = min(
            explicit_entities or [value for value in members if _text(value)] or [candidate],
            key=lambda value: len(_text(value)) or 10_000,
        )
        surface = _compact_surface(_text(surface_node))
        if not surface:
            continue
        if not _entity_relevant_to_event(surface, event_text):
            continue
        source_ids = sorted(
            {
                str(value.get("node_id"))
                for value in members
                if value.get("node_id")
            }
        )
        confidence_values = [
            float(value["confidence"])
            for value in members
            if isinstance(value.get("confidence"), (int, float))
        ]
        by_component.setdefault(
            component_id,
            {
                "mention_id": f"l1-{component_id}",
                "role": "other",
                "entity_type": _entity_type(surface),
                "surface": surface,
                "confidence": min(confidence_values, default=0.7),
                "grounding_refs": [event_ref],
                "source_l1_node_ids": source_ids,
                "identity_basis": (
                    "accepted_conflict_aware_identity_track"
                    if len(source_ids) > 1
                    else "grounded_singleton_observation"
                ),
            },
        )
    return list(by_component.values())[:4]


def _states_for_event(
    clip_nodes: list[dict[str, Any]],
    *,
    participants: list[dict[str, Any]],
    event_ref: str,
    event_text: str,
) -> list[dict[str, Any]]:
    if len(participants) != 1:
        return []
    mention_id = str(participants[0]["mention_id"])
    states: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in clip_nodes:
        if str(raw.get("node_type") or "") != "state":
            continue
        text = _text(raw)
        attribute = _state_attribute(text)
        value = _state_value(text)
        key = (attribute, value)
        if (
            not attribute
            or not value
            or key in seen
            or not _state_relevant_to_event(
                text,
                event_text=event_text,
                participant_surface=str(participants[0].get("surface") or ""),
                attribute=attribute,
            )
        ):
            continue
        seen.add(key)
        confidence = raw.get("confidence")
        states.append(
            {
                "mention_id": mention_id,
                "attribute": attribute,
                "value": value,
                "polarity": "positive",
                "confidence": (
                    float(confidence)
                    if isinstance(confidence, (int, float))
                    else 0.7
                ),
                "grounding_refs": [event_ref],
                "source_l1_node_id": str(raw.get("node_id") or ""),
            }
        )
    return states[:3]


def _state_attribute(text: str) -> str:
    normalized = text.lower()
    patterns = (
        ("gaze_direction", ("gaze", "looking", "facing", "eye contact")),
        ("motion_state", ("moving", "stationary", "speed", "stopped", "walking")),
        ("possession", ("holding", "gripping", "carrying", "held")),
        ("openness", ("open", "closed", "shut")),
        ("visibility", ("visible", "hidden", "covered", "reveals", "out of frame")),
        ("expression", ("expression", "grimace", "distress", "smile", "fear")),
        ("illumination", ("light", "dark", "illuminat", "headlight", "flashlight")),
        ("location", ("inside", "outside", "interior", "road", "room")),
    )
    for attribute, terms in patterns:
        if any(term in normalized for term in terms):
            return attribute
    return ""


def _state_value(text: str) -> str:
    return " ".join(text.strip().split())[:160]


def _entity_relevant_to_event(surface: str, event_text: str) -> bool:
    normalized_event = event_text.strip().lower()
    if normalized_event.startswith(
        ("camera ", "view ", "transition ", "scene ", "shot ")
    ):
        return False
    surface_primary = _primary_entity_category(surface)
    event_primary = _primary_entity_category(event_text)
    if surface_primary and event_primary:
        return surface_primary == event_primary
    surface_categories = _entity_categories(surface)
    event_categories = _entity_categories(event_text)
    if surface_categories and event_categories:
        return bool(surface_categories & event_categories)
    surface_tokens = _content_tokens(surface)
    event_tokens = _content_tokens(event_text)
    return bool(surface_tokens & event_tokens)


def _same_local_entity(left: str, right: str) -> bool:
    left_primary = _primary_entity_category(left)
    right_primary = _primary_entity_category(right)
    if left_primary and right_primary:
        return left_primary == right_primary
    return bool(_content_tokens(left) & _content_tokens(right))


def _state_relevant_to_event(
    text: str,
    *,
    event_text: str,
    participant_surface: str,
    attribute: str,
) -> bool:
    state_primary = _primary_entity_category(text)
    participant_primary = _primary_entity_category(participant_surface)
    if state_primary and participant_primary:
        return state_primary == participant_primary
    state_categories = _entity_categories(text)
    participant_categories = _entity_categories(participant_surface)
    if state_categories:
        return bool(state_categories & participant_categories)
    attribute_terms = {
        "gaze_direction": {"gaze", "look", "face", "head", "eye"},
        "motion_state": {"move", "moving", "stationary", "stop", "walk", "speed"},
        "possession": {"hold", "held", "grip", "carry"},
        "openness": {"open", "close", "closed", "shut"},
        "visibility": {"visible", "hidden", "cover", "reveal", "frame"},
        "expression": {"expression", "grimace", "distress", "smile", "fear"},
        "illumination": {"light", "dark", "illuminat", "headlight", "flashlight"},
        "location": {"inside", "outside", "interior", "road", "room"},
    }
    normalized_event = event_text.lower()
    return any(term in normalized_event for term in attribute_terms.get(attribute, set()))


def _entity_categories(text: str) -> set[str]:
    normalized = text.lower()
    categories: set[str] = set()
    groups = {
        "human": (
            "man",
            "woman",
            "person",
            "driver",
            "boy",
            "girl",
            "face",
            "head",
            "hand",
            "passenger",
        ),
        "vehicle": ("car", "vehicle", "truck", "steering wheel", "dashboard"),
        "fabric": ("fabric", "blanket", "cloth", "bundle"),
        "road": ("road", "path", "terrain"),
        "structure": ("room", "interior", "wall", "door", "window"),
    }
    for category, terms in groups.items():
        if any(_contains_term(normalized, term) for term in terms):
            categories.add(category)
    return categories


def _primary_entity_category(text: str) -> str:
    normalized = text.lower()
    terms = {
        "human": (
            "man",
            "woman",
            "person",
            "driver",
            "boy",
            "girl",
            "face",
            "head",
            "hand",
            "passenger",
            " he ",
            " she ",
        ),
        "vehicle": ("car", "vehicle", "truck", "steering wheel", "dashboard"),
        "fabric": ("fabric", "blanket", "cloth", "bundle"),
        "road": ("road", "path", "terrain"),
        "structure": ("room", "interior", "wall", "door", "window"),
    }
    padded = f" {normalized} "
    matches = [
        (match.start(), category)
        for category, values in terms.items()
        for term in values
        for match in [re.search(rf"\b{re.escape(term.strip())}\b", padded)]
        if match is not None
    ]
    return min(matches)[1] if matches else ""


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _content_tokens(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "in",
        "on",
        "at",
        "to",
        "of",
        "and",
        "with",
        "from",
        "as",
        "it",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop
    }


def _entity_type(surface: str) -> str:
    normalized = surface.lower()
    if any(
        term in normalized
        for term in ("man", "woman", "person", "driver", "boy", "girl", "hand")
    ):
        return "person"
    if any(term in normalized for term in ("road", "room", "interior", "outside")):
        return "place"
    return "object"


def _compact_surface(text: str) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= 120:
        return normalized
    return normalized[:117].rstrip() + "..."


def _text(node: dict[str, Any]) -> str:
    return str(node.get("text") or node.get("description") or "").strip()
