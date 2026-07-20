"""Derive state transitions only on accepted conflict-aware identity tracks."""

from __future__ import annotations

import re
from typing import Any

from .types import MemoryNode, RelationBelief, RelationStatus


def derive_state_transition_candidates(
    events: list[MemoryNode],
) -> list[RelationBelief]:
    """Return adjacent graph-grounded deltas on one accepted identity track."""

    ordered = sorted(events, key=lambda node: (node.time_span.start_s, node.node_id))
    candidates: list[RelationBelief] = []
    seen: set[tuple[str, str, str, str]] = set()
    previous_by_track_attribute: dict[
        tuple[str, str], tuple[MemoryNode, dict[str, Any], str]
    ] = {}
    for dst in ordered:
        for track_id, states in _states_by_mention(dst).items():
            if not track_id.startswith("l1-track:"):
                continue
            for after in states:
                if not _formal_state_contract(after):
                    continue
                attribute, after_value = _canonical_state(after)
                if not attribute or not after_value:
                    continue
                state_key = (track_id, attribute)
                previous = previous_by_track_attribute.get(state_key)
                previous_by_track_attribute[state_key] = (dst, after, after_value)
                if previous is None:
                    continue
                src, before, before_value = previous
                if src.time_span.end_s > dst.time_span.start_s:
                    continue
                if before_value == after_value:
                    continue
                if str(before.get("polarity") or "positive") != "positive":
                    continue
                if str(after.get("polarity") or "positive") != "positive":
                    continue
                key = (src.node_id, dst.node_id, track_id, attribute.casefold())
                if key in seen:
                    continue
                seen.add(key)
                probability = min(
                    _confidence(before),
                    _confidence(after),
                    0.99,
                )
                candidates.append(
                    RelationBelief(
                        edge_id=(
                            f"derived-state:{src.node_id}->{dst.node_id}:"
                            f"{_slug(attribute)}"
                        ),
                        src=src.node_id,
                        dst=dst.node_id,
                        relation_probabilities={"state_transition": probability},
                        status=RelationStatus.UNCALIBRATED_PRIOR,
                        direction_confidence=1.0,
                        evidence_refs=list(
                            dict.fromkeys(src.source_segments + dst.source_segments)
                        ),
                        warrant=(
                            f"Accepted track {track_id} changes {attribute} from "
                            f"{before['value']} to {after['value']}."
                        ),
                        provenance={
                            "producer": "memory_graph.state_relations",
                            "accepted_identity_track": track_id,
                            "participant_alignment": [
                                {
                                    "src_mention_id": track_id,
                                    "dst_mention_id": track_id,
                                }
                            ],
                            "before_state": dict(before),
                            "after_state": dict(after),
                            "normalized_state_delta": {
                                "attribute": attribute,
                                "before": before_value,
                                "after": after_value,
                            },
                            "derivation": "accepted_track_grounded_state_delta/v1",
                        },
                    )
                )
    return candidates


def _states_by_mention(node: MemoryNode) -> dict[str, list[dict[str, Any]]]:
    participants = {
        str(value.get("mention_id") or "")
        for value in node.metadata.get("participants") or []
        if isinstance(value, dict)
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for value in node.metadata.get("states") or []:
        if not isinstance(value, dict):
            continue
        mention_id = str(value.get("mention_id") or "")
        if mention_id not in participants:
            continue
        if not value.get("attribute") or not value.get("value"):
            continue
        grouped.setdefault(mention_id, []).append(value)
    return grouped


def _canonical_state(value: dict[str, Any]) -> tuple[str, str]:
    attribute = _norm(value.get("attribute")).replace(" ", "_")
    state_value = _norm(value.get("value"))
    if attribute == "visibility" and state_value in {"open", "closed"}:
        attribute = "openness"
    if attribute == "gaze_direction":
        state_value = _canonical_gaze(state_value)
    return attribute, state_value


def _canonical_gaze(value: str) -> str:
    token_set = set(re.findall(r"[a-z]+", value))
    if "down" in token_set or "downward" in token_set:
        return "downward"
    if "up" in token_set or "upward" in token_set:
        return "upward"
    if "forward" in token_set or "ahead" in token_set:
        return "forward"
    if "outward" in token_set or (
        "out" in token_set and "window" in token_set
    ):
        return "outward"
    if "left" in token_set:
        return "left"
    if "right" in token_set:
        return "right"
    return value


def _confidence(value: dict[str, Any]) -> float:
    raw = value.get("confidence", 0.7)
    return max(0.0, min(1.0, float(raw)))


def _formal_state_contract(value: dict[str, Any]) -> bool:
    contract = value.get("contract_version")
    if "contract_version" not in value:
        # Unit fixtures and pre-Video_Skills callers remain supported. Persisted
        # Video_Skills states always carry this field and must use the strict contract.
        return True
    return (
        contract == "grounded-state-assertion/v1"
        and value.get("subject_binding") == "explicit_state_of"
        and bool(value.get("source_l1_node_id"))
        and bool(value.get("subject_source_l1_node_id"))
    )


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "state"
