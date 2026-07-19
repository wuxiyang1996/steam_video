"""Mechanism-first candidate generation for the L1.5 causal overlay."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import MechanismKind, MemoryNode


@dataclass(frozen=True)
class MechanismCandidate:
    """A recall proposal, not an accepted causal claim."""

    src: str
    dst: str
    mechanism_hints: tuple[str, ...]
    shared_entity: dict[str, str] | None = None
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    requires_temporal_refine: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "mechanism_hints": list(self.mechanism_hints),
            "shared_entity": self.shared_entity,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "requires_temporal_refine": self.requires_temporal_refine,
        }


def generate_mechanism_candidates(
    nodes: list[MemoryNode],
    *,
    max_gap_s: float = 30.0,
    max_forward_neighbors: int = 8,
) -> list[MechanismCandidate]:
    """Propose pairs from visible entity/state structure before embeddings.

    The output deliberately carries mechanism *hints*. A teacher, targeted
    visual reread, and hard verifier must still establish a causal witness.
    """

    ordered = sorted(
        nodes,
        key=lambda node: (node.time_span.start_s, node.time_span.end_s, node.node_id),
    )
    candidates: list[MechanismCandidate] = []
    seen: set[tuple[str, str]] = set()
    for index, src in enumerate(ordered):
        for dst in ordered[index + 1 : index + 1 + max_forward_neighbors]:
            gap = dst.time_span.start_s - src.time_span.end_s
            if gap > max_gap_s:
                break
            shared = _shared_mentions(src, dst)
            state_delta = _state_delta(src, dst, shared)
            hints: list[str] = []
            if state_delta is not None:
                hints.append(MechanismKind.STATE_BRIDGE.value)
            if shared and _has_contact_transfer_language(src, dst):
                hints.append(MechanismKind.CONTACT_TRANSFER.value)
            if shared and _has_observable_precondition(src, dst):
                hints.append(MechanismKind.OBSERVABLE_PRECONDITION.value)
            if shared and _has_response_language(src, dst):
                hints.append(MechanismKind.RULE_RESPONSE.value)
            if not hints:
                continue

            key = (src.node_id, dst.node_id)
            if key in seen:
                continue
            seen.add(key)
            shared_entity = shared[0] if shared else None
            candidates.append(
                MechanismCandidate(
                    src=src.node_id,
                    dst=dst.node_id,
                    mechanism_hints=tuple(dict.fromkeys(hints)),
                    shared_entity=shared_entity,
                    before_state=state_delta[0] if state_delta else None,
                    after_state=state_delta[1] if state_delta else None,
                    requires_temporal_refine=bool(
                        set(src.source_segments) & set(dst.source_segments)
                    ),
                )
            )
    return candidates


def generate_video_skills_l1_edge_candidates(
    event_nodes: list[MemoryNode],
    *,
    l1_nodes: list[MemoryNode],
    event_to_l1: dict[str, tuple[str, ...]],
) -> list[MechanismCandidate]:
    """Use accepted L1 edges as recall hints, never as accepted causal claims."""

    event_by_l1: dict[str, str] = {}
    for event_id, refs in event_to_l1.items():
        for ref in refs:
            event_by_l1[ref] = event_id
    memory_ref_by_source = {
        str(node.source_node_id): node.node_id
        for node in l1_nodes
        if node.source_node_id
    }
    event_ids = {node.node_id for node in event_nodes}
    event_by_id = {node.node_id: node for node in event_nodes}
    hints_by_pair: dict[tuple[str, str], set[str]] = {}
    seen_edges: set[str] = set()
    hint_map = {
        "same_entity": "l1_entity_continuity",
        "same_object": "l1_entity_continuity",
        "same_place": "l1_place_continuity",
        "reappears": "l1_entity_continuity",
        "before_after": "visible_state_delta",
        "state_change": "visible_state_delta",
        "causal_hint": "l1_causal_hint_requires_verification",
        "social_cue": "visible_response",
    }
    for node in l1_nodes:
        for edge in node.metadata.get("source_l1_edges") or []:
            if not isinstance(edge, dict):
                continue
            edge_id = str(edge.get("edge_id") or "")
            if edge_id and edge_id in seen_edges:
                continue
            if edge_id:
                seen_edges.add(edge_id)
            hint = hint_map.get(str(edge.get("edge_type") or ""))
            src_ref = memory_ref_by_source.get(str(edge.get("src") or ""))
            dst_ref = memory_ref_by_source.get(str(edge.get("dst") or ""))
            src = event_by_l1.get(src_ref or "")
            dst = event_by_l1.get(dst_ref or "")
            if not hint or src not in event_ids or dst not in event_ids or src == dst:
                continue
            if event_by_id[src].time_span.start_s > event_by_id[dst].time_span.start_s:
                src, dst = dst, src
            hints_by_pair.setdefault((src, dst), set()).add(hint)
    return [
        MechanismCandidate(
            src=src,
            dst=dst,
            mechanism_hints=tuple(sorted(hints)),
            requires_temporal_refine=(
                "l1_causal_hint_requires_verification" in hints
            ),
        )
        for (src, dst), hints in sorted(hints_by_pair.items())
    ]


def _shared_mentions(
    src: MemoryNode,
    dst: MemoryNode,
) -> list[dict[str, str]]:
    src_mentions = _mentions(src)
    dst_mentions = _mentions(dst)
    shared: list[dict[str, str]] = []
    for left in src_mentions:
        for right in dst_mentions:
            same_surface = _norm(left["surface"]) == _norm(right["surface"])
            same_type = _norm(left["entity_type"]) == _norm(right["entity_type"])
            if same_surface and same_type and left["surface"]:
                shared.append(
                    {
                        "src_mention_id": left["mention_id"],
                        "dst_mention_id": right["mention_id"],
                        "entity_type": left["entity_type"],
                        "surface": left["surface"],
                    }
                )
    return shared


def _state_delta(
    src: MemoryNode,
    dst: MemoryNode,
    shared: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    src_states = _states(src)
    dst_states = _states(dst)
    for entity in shared:
        for before in src_states:
            if before["mention_id"] != entity["src_mention_id"]:
                continue
            for after in dst_states:
                if after["mention_id"] != entity["dst_mention_id"]:
                    continue
                if _norm(before["attribute"]) != _norm(after["attribute"]):
                    continue
                if _norm(before["value"]) == _norm(after["value"]):
                    continue
                return before, after
    return None


def _mentions(node: MemoryNode) -> list[dict[str, str]]:
    values = node.metadata.get("participants") or []
    return [
        {
            "mention_id": str(value.get("mention_id") or ""),
            "entity_type": str(value.get("entity_type") or ""),
            "surface": str(value.get("surface") or ""),
        }
        for value in values
        if isinstance(value, dict)
    ]


def _states(node: MemoryNode) -> list[dict[str, Any]]:
    values = node.metadata.get("states") or []
    return [
        {
            "mention_id": str(value.get("mention_id") or ""),
            "attribute": str(value.get("attribute") or ""),
            "value": str(value.get("value") or ""),
            "confidence": float(value.get("confidence", 0.0)),
        }
        for value in values
        if isinstance(value, dict)
    ]


def _has_contact_transfer_language(src: MemoryNode, dst: MemoryNode) -> bool:
    source = _norm(src.text or "")
    effect = _norm(dst.text or "")
    contact = ("push", "pull", "hit", "strike", "kick", "drop", "throw", "press")
    response = ("move", "fall", "open", "close", "break", "tip", "roll", "slide")
    return any(word in source for word in contact) and any(
        word in effect for word in response
    )


def _has_observable_precondition(src: MemoryNode, dst: MemoryNode) -> bool:
    source = _norm(src.text or "")
    effect = _norm(dst.text or "")
    established = ("open", "unlock", "available", "accessible", "visible", "clear")
    dependent = ("enter", "exit", "pass", "take", "pick", "use", "retrieve")
    return any(word in source for word in established) and any(
        word in effect for word in dependent
    )


def _has_response_language(src: MemoryNode, dst: MemoryNode) -> bool:
    source = _norm(src.text or "")
    effect = _norm(dst.text or "")
    trigger = ("knock", "ring", "call", "signal", "ask", "shout", "wave")
    response = ("answer", "respond", "open", "turn", "look", "approach")
    return any(word in source for word in trigger) and any(
        word in effect for word in response
    )


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
