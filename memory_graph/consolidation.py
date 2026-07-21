"""Materialize a bounded-memory plan into a valid retained evidence graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Iterable

from .selectstream_policy import MemoryPolicyDecision
from .types import MemoryNode, RelationBelief, RelationStatus, TimeSpan


@dataclass(frozen=True)
class ConsolidationResult:
    nodes: tuple[MemoryNode, ...]
    relations: tuple[RelationBelief, ...]
    id_map: dict[str, str | None]
    merged_lineage: dict[str, tuple[str, ...]]
    evicted: tuple[str, ...]
    capacity: int

    def __post_init__(self) -> None:
        if len(self.nodes) > self.capacity:
            raise ValueError("materialized memory exceeds its fixed capacity")
        known = {node.node_id for node in self.nodes}
        if any(
            edge.src not in known or edge.dst not in known for edge in self.relations
        ):
            raise ValueError("materialized memory contains an orphan relation")

    def audit_dict(self) -> dict[str, Any]:
        return {
            "policy": "selectstream_materialized_consolidation/v0.1",
            "capacity": self.capacity,
            "retained_count": len(self.nodes),
            "relation_count": len(self.relations),
            "merged_lineage": {
                node_id: list(lineage)
                for node_id, lineage in self.merged_lineage.items()
            },
            "evicted": list(self.evicted),
            "id_map": self.id_map,
            "materialized": True,
        }


def materialize_bounded_memory(
    nodes: Iterable[MemoryNode],
    relations: Iterable[RelationBelief],
    decision: MemoryPolicyDecision,
) -> ConsolidationResult:
    """Execute keep/merge/evict and rewire every surviving relation.

    Merging never carries a stale embedding forward.  Non-deterministic edges
    touching a merged endpoint are retained only as uncalibrated candidates and
    explicitly marked for re-verification.  Temporal adjacency is rebuilt from
    the retained timeline, so eviction cannot leave a broken temporal chain.
    """

    original_nodes = list(nodes)
    original_relations = list(relations)
    by_id = {node.node_id: node for node in original_nodes}
    if len(by_id) != len(original_nodes):
        raise ValueError("cannot consolidate duplicate node IDs")
    unknown = (set(decision.keep) | set(decision.evict)) - set(by_id)
    if unknown:
        raise ValueError(f"memory decision references unknown nodes: {sorted(unknown)}")
    if set(decision.keep) & set(decision.evict):
        raise ValueError("memory decision cannot both keep and evict a node")
    if set(decision.keep) | set(decision.evict) != set(by_id):
        raise ValueError("memory decision must cover every input node")
    for witness in decision.protected_witness_sets:
        missing = set(witness) - set(decision.keep)
        if missing:
            raise ValueError(
                "materialization would evict protected witness members: "
                + ", ".join(sorted(missing))
            )

    merge_groups = _validated_merge_groups(decision, by_id)
    merged_members = {node_id for group in merge_groups for node_id in group}
    id_map: dict[str, str | None] = {
        node.node_id: (node.node_id if node.node_id in decision.keep else None)
        for node in original_nodes
    }
    retained: list[MemoryNode] = [
        node
        for node in original_nodes
        if node.node_id in decision.keep and node.node_id not in merged_members
    ]
    merged_lineage: dict[str, tuple[str, ...]] = {}
    for group in merge_groups:
        source_nodes = sorted(
            (by_id[node_id] for node_id in group),
            key=lambda node: (
                node.time_span.start_s,
                node.time_span.end_s,
                node.node_id,
            ),
        )
        merged = _merge_nodes(source_nodes)
        retained.append(merged)
        merged_lineage[merged.node_id] = tuple(node.node_id for node in source_nodes)
        for node in source_nodes:
            id_map[node.node_id] = merged.node_id

    retained.sort(
        key=lambda node: (node.time_span.start_s, node.time_span.end_s, node.node_id)
    )
    rewired = _rewire_relations(original_relations, id_map, merged_lineage)
    rewired.extend(_rebuilt_temporal_relations(retained))
    rewired = _deduplicate_relations(rewired)
    result = ConsolidationResult(
        nodes=tuple(retained),
        relations=tuple(rewired),
        id_map=id_map,
        merged_lineage=merged_lineage,
        evicted=tuple(decision.evict),
        capacity=decision.capacity,
    )
    return result


def _validated_merge_groups(
    decision: MemoryPolicyDecision,
    by_id: dict[str, MemoryNode],
) -> tuple[tuple[str, ...], ...]:
    seen: set[str] = set()
    groups: list[tuple[str, ...]] = []
    protected = {value for group in decision.protected_witness_sets for value in group}
    for raw_group in decision.merge_groups:
        group = tuple(dict.fromkeys(raw_group))
        if len(group) < 2:
            continue
        if any(node_id not in decision.keep for node_id in group):
            raise ValueError("merge group must contain only kept nodes")
        if set(group) & protected:
            raise ValueError("protected witness nodes cannot be merged")
        overlap = seen & set(group)
        if overlap:
            raise ValueError(f"merge groups overlap: {sorted(overlap)}")
        video_ids = {by_id[node_id].video_id for node_id in group}
        if len(video_ids) != 1:
            raise ValueError("cannot merge evidence from different videos")
        seen.update(group)
        groups.append(group)
    return tuple(groups)


def _merge_nodes(nodes: list[MemoryNode]) -> MemoryNode:
    lineage = tuple(node.node_id for node in nodes)
    digest = hashlib.sha256("\x1f".join(lineage).encode("utf-8")).hexdigest()[:16]
    texts = tuple(
        dict.fromkeys((node.text or "").strip() for node in nodes if node.text)
    )
    node_types = {node.node_type for node in nodes}
    metadata = _merge_metadata(nodes)
    metadata["consolidation"] = {
        "lineage": list(lineage),
        "operation": "priority_preserving_merge",
        "embedding_status": "refresh_required",
    }
    return MemoryNode(
        node_id=f"memory:consolidated:{digest}",
        video_id=nodes[0].video_id,
        time_span=TimeSpan(
            min(node.time_span.start_s for node in nodes),
            max(node.time_span.end_s for node in nodes),
        ),
        provenance={
            "producer": "memory_graph.consolidation.materialize_bounded_memory",
            "operation": "priority_preserving_merge",
            "source_node_ids": list(lineage),
            "source_provenance": [node.provenance for node in nodes],
        },
        node_type=(
            next(iter(node_types)) if len(node_types) == 1 else "consolidated_evidence"
        ),
        text=" | ".join(texts) if texts else None,
        source_segments=list(
            dict.fromkeys(
                segment
                for node in nodes
                for segment in (node.source_segments or [node.node_id])
            )
        ),
        embedding_ref=None,
        metadata=metadata,
    )


def _merge_metadata(nodes: list[MemoryNode]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    keys = {key for node in nodes for key in node.metadata}
    for key in sorted(keys):
        values = [node.metadata[key] for node in nodes if key in node.metadata]
        if all(value == values[0] for value in values):
            merged[key] = values[0]
        elif all(isinstance(value, list) for value in values):
            merged[key] = _unique_json_values(
                item for value in values for item in value
            )
        else:
            merged[f"source_{key}"] = values
    return merged


def _unique_json_values(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    fingerprints: set[str] = set()
    for value in values:
        fingerprint = repr(value)
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            result.append(value)
    return result


def _rewire_relations(
    relations: list[RelationBelief],
    id_map: dict[str, str | None],
    merged_lineage: dict[str, tuple[str, ...]],
) -> list[RelationBelief]:
    rewired: list[RelationBelief] = []
    temporal_names = {"temporal_next", "before", "overlaps", "during"}
    for edge in relations:
        src = id_map.get(edge.src)
        dst = id_map.get(edge.dst)
        if src is None or dst is None or src == dst:
            continue
        non_temporal_probabilities = {
            name: value
            for name, value in edge.relation_probabilities.items()
            if name not in temporal_names
        }
        if not non_temporal_probabilities:
            continue
        touched_merge = src in merged_lineage or dst in merged_lineage
        digest = hashlib.sha256(
            f"{edge.edge_id}\x1f{src}\x1f{dst}".encode("utf-8")
        ).hexdigest()[:16]
        provenance = {
            **edge.provenance,
            "consolidation": {
                "source_edge_id": edge.edge_id,
                "src_lineage": list(merged_lineage.get(src, (src,))),
                "dst_lineage": list(merged_lineage.get(dst, (dst,))),
                "requires_reverification": touched_merge
                and edge.status is not RelationStatus.DETERMINISTIC,
            },
        }
        rewired.append(
            replace(
                edge,
                edge_id=f"relation:consolidated:{digest}",
                src=src,
                dst=dst,
                relation_probabilities=non_temporal_probabilities,
                status=(
                    RelationStatus.UNCALIBRATED_PRIOR
                    if touched_merge and edge.status is not RelationStatus.DETERMINISTIC
                    else edge.status
                ),
                evidence_refs=list(
                    dict.fromkeys(
                        id_map.get(ref, ref)
                        for ref in edge.evidence_refs
                        if id_map.get(ref, ref) is not None
                    )
                ),
                provenance=provenance,
            )
        )
    return rewired


def _rebuilt_temporal_relations(nodes: list[MemoryNode]) -> list[RelationBelief]:
    relations: list[RelationBelief] = []
    for left, right in zip(nodes, nodes[1:]):
        overlaps = left.time_span.overlaps(right.time_span)
        names = {"overlaps": 1.0} if overlaps else {"temporal_next": 1.0, "before": 1.0}
        digest = hashlib.sha256(
            f"{left.node_id}\x1f{right.node_id}\x1f{','.join(names)}".encode("utf-8")
        ).hexdigest()[:16]
        relations.append(
            RelationBelief(
                edge_id=f"relation:retained-temporal:{digest}",
                src=left.node_id,
                dst=right.node_id,
                relation_probabilities=names,
                status=RelationStatus.DETERMINISTIC,
                direction_confidence=1.0,
                evidence_refs=[left.node_id, right.node_id],
                warrant="retained timeline adjacency after consolidation",
                provenance={
                    "producer": "memory_graph.consolidation._rebuilt_temporal_relations",
                    "question_independent": True,
                },
            )
        )
    return relations


def _deduplicate_relations(relations: list[RelationBelief]) -> list[RelationBelief]:
    selected: dict[tuple[str, str, tuple[str, ...]], RelationBelief] = {}
    for edge in relations:
        key = (edge.src, edge.dst, tuple(sorted(edge.relation_probabilities)))
        existing = selected.get(key)
        if existing is None or (
            existing.status is not RelationStatus.DETERMINISTIC
            and edge.status is RelationStatus.DETERMINISTIC
        ):
            selected[key] = edge
    return sorted(
        selected.values(),
        key=lambda edge: (edge.src, edge.dst, edge.edge_id),
    )
