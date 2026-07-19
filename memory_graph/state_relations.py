"""Derive state transitions only on accepted conflict-aware identity tracks."""

from __future__ import annotations

import re
from typing import Any

from .types import MemoryNode, RelationBelief, RelationStatus


def derive_state_transition_candidates(
    events: list[MemoryNode],
) -> list[RelationBelief]:
    """Return graph-grounded before/after deltas on one accepted track."""

    ordered = sorted(events, key=lambda node: (node.time_span.start_s, node.node_id))
    candidates: list[RelationBelief] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, src in enumerate(ordered):
        for dst in ordered[index + 1 :]:
            if src.time_span.end_s > dst.time_span.start_s:
                continue
            for before, after, track_id in _track_state_deltas(src, dst):
                attribute = str(before["attribute"])
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
                            "derivation": "accepted_track_grounded_state_delta/v1",
                        },
                    )
                )
    return candidates


def _track_state_deltas(
    src: MemoryNode,
    dst: MemoryNode,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    src_states = _states_by_mention(src)
    dst_states = _states_by_mention(dst)
    deltas: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for track_id in sorted(set(src_states) & set(dst_states)):
        if not track_id.startswith("l1-track:"):
            continue
        for before in src_states[track_id]:
            for after in dst_states[track_id]:
                if _norm(before.get("attribute")) != _norm(after.get("attribute")):
                    continue
                if _norm(before.get("value")) == _norm(after.get("value")):
                    continue
                if str(before.get("polarity") or "positive") != "positive":
                    continue
                if str(after.get("polarity") or "positive") != "positive":
                    continue
                deltas.append((before, after, track_id))
    return deltas


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


def _confidence(value: dict[str, Any]) -> float:
    raw = value.get("confidence", 0.7)
    return max(0.0, min(1.0, float(raw)))


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "state"
