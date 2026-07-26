"""Loss-audited adapter from frozen v1 L1/L1.5 graph artifacts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
    RetainedEvidenceGraph,
)

from .contracts import (
    EntityMention,
    EvidenceAddress,
    EvidenceMemory,
    EvidenceRecord,
    EvidenceValue,
    GroundedState,
    GroundedStateDelta,
    TemporalLink,
)


def from_retained_graph(graph: RetainedEvidenceGraph) -> EvidenceMemory:
    records = tuple(_record(node) for node in graph.nodes)
    links = tuple(
        TemporalLink(edge.edge_id, edge.src, edge.dst, edge.relation)
        for edge in graph.temporal_edges
    )
    return EvidenceMemory(
        memory_id=f"evidence-v2:{graph.graph_id}",
        records=records,
        temporal_links=links,
        metadata={
            "schema_version": "steam-reasoning-v2-evidence-memory/v0.1",
            "source_graph_id": graph.graph_id,
            "legacy_adapter": True,
            "question_independent": bool(graph.metadata.get("question_independent")),
        },
    )


def _record(node: Any) -> EvidenceRecord:
    metadata = node.metadata
    entities = tuple(
        EntityMention(
            mention_id=str(row.get("mention_id") or f"{node.node_id}:entity:{index}"),
            role=str(row.get("role") or "other"),
            entity_type=str(row.get("entity_type") or "other"),
            surface=str(row.get("surface") or ""),
            visual_signature=str(row.get("visual_signature") or ""),
            track_status=str(row.get("track_status") or "event_local_only"),
        )
        for index, row in enumerate(metadata.get("participants") or ())
        if isinstance(row, dict)
    )
    mention_ids = {row.mention_id for row in entities}
    states = tuple(
        GroundedState(
            mention_id=str(row.get("mention_id") or ""),
            attribute=str(row.get("attribute") or ""),
            value=str(row.get("value") or ""),
            polarity=str(row.get("polarity") or "positive"),
        )
        for row in metadata.get("states") or ()
        if isinstance(row, dict)
        and row.get("mention_id") in mention_ids
        and row.get("attribute")
        and row.get("value")
    )
    state_delta = _state_delta(metadata.get("state_change"), mention_ids)
    action_kind = str(metadata.get("action_kind") or node.node_type or "event")
    address = EvidenceAddress(
        node_id=node.node_id,
        video_id=node.video_id,
        start_s=node.time_span.start_s,
        end_s=node.time_span.end_s,
        # V2 intentionally exposes only a coarse family before a read.  The
        # predicate and participant attributes remain in EvidenceValue.
        event_family=action_kind,
        structural_tags=tuple(
            sorted(
                {
                    action_kind,
                    str(metadata.get("source_type") or "visual_observation"),
                }
            )
        ),
        source_segments=tuple(node.source_segments or ()),
        embedding_ref=(asdict(node.embedding_ref) if node.embedding_ref else None),
    )
    predicate = str(metadata.get("predicate") or node.text or action_kind)
    value = EvidenceValue(
        descriptor=str(node.text or predicate),
        predicate=predicate,
        entities=entities,
        states=states,
        state_delta=state_delta,
        provenance=node.provenance,
    )
    return EvidenceRecord(address=address, value=value)


def _state_delta(value: Any, mention_ids: set[str]) -> GroundedStateDelta | None:
    if not isinstance(value, dict):
        return None
    mention_id = str(value.get("mention_id") or "")
    attribute = str(value.get("attribute") or "")
    before = str(value.get("before") or "")
    after = str(value.get("after") or "")
    if (
        mention_id not in mention_ids
        or not all((attribute, before, after))
        or before == after
    ):
        return None
    return GroundedStateDelta(mention_id, attribute, before, after)
