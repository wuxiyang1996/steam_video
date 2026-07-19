"""Adapt grounded L1.5 atomic events into relation-graph nodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .contracts import AtomicEvent
from .types import MemoryNode


@dataclass(frozen=True)
class OverlayIndex:
    l1_to_events: dict[str, tuple[str, ...]]
    event_to_l1: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, dict[str, list[str]]]:
        return {
            "l1_to_events": {
                node_id: list(event_ids)
                for node_id, event_ids in self.l1_to_events.items()
            },
            "event_to_l1": {
                event_id: list(node_ids)
                for event_id, node_ids in self.event_to_l1.items()
            },
        }


def atomic_events_to_graph_nodes(
    events: list[AtomicEvent],
    *,
    l1_nodes: list[MemoryNode],
) -> tuple[list[MemoryNode], OverlayIndex]:
    """Create event endpoints while retaining immutable L1 grounding."""
    l1_by_id = {node.node_id: node for node in l1_nodes}
    event_nodes: list[MemoryNode] = []
    event_to_l1: dict[str, tuple[str, ...]] = {}
    l1_to_events_lists: dict[str, list[str]] = {node_id: [] for node_id in l1_by_id}

    for event in events:
        unknown = set(event.evidence_refs) - set(l1_by_id)
        if unknown:
            raise ValueError(
                f"atomic event {event.event_id} references unknown L1 nodes: "
                f"{sorted(unknown)}"
            )
        node_id = _event_node_id(event.event_id)
        evidence_refs = tuple(event.evidence_refs)
        event_to_l1[node_id] = evidence_refs
        for evidence_ref in evidence_refs:
            l1_to_events_lists[evidence_ref].append(node_id)

        event_nodes.append(
            MemoryNode(
                node_id=node_id,
                video_id=event.video_id,
                time_span=event.time_span,
                provenance={
                    **event.provenance,
                    "layer": "L1.5",
                    "adapter": "memory_graph.event_adapter.atomic_events_to_graph_nodes",
                },
                node_type="atomic_event",
                text=atomic_event_text(event),
                source_node_id=event.event_id,
                source_segments=list(evidence_refs),
                metadata={
                    "predicate": event.predicate,
                    "confidence": event.confidence,
                    "participants": [
                        {
                            **asdict(participant),
                            "grounding_refs": list(participant.grounding_refs),
                        }
                        for participant in event.participants
                    ],
                    "states": [asdict(state) for state in event.states],
                    "evidence_refs": list(evidence_refs),
                    "layer": "L1.5",
                },
            )
        )

    event_nodes.sort(
        key=lambda node: (
            node.time_span.start_s,
            node.time_span.end_s,
            node.node_id,
        )
    )
    index = OverlayIndex(
        l1_to_events={
            node_id: tuple(event_ids)
            for node_id, event_ids in l1_to_events_lists.items()
        },
        event_to_l1=event_to_l1,
    )
    return event_nodes, index


def atomic_event_text(event: AtomicEvent) -> str:
    """Text embedding input at event granularity, not source-segment granularity."""
    participant_text = ", ".join(
        f"{mention.role}={mention.surface}" for mention in event.participants
    )
    state_text = ", ".join(
        f"{state.mention_id}.{state.attribute}={state.value}"
        for state in event.states
    )
    parts = [event.predicate]
    if participant_text:
        parts.append(f"participants: {participant_text}")
    if state_text:
        parts.append(f"visible states: {state_text}")
    return " | ".join(parts)


def _event_node_id(event_id: str) -> str:
    return event_id if event_id.startswith("event:") else f"event:{event_id}"
