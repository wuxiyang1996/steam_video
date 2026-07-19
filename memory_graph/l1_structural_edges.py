"""Materialize accepted Video_Skills relations for L1-only navigation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .types import MemoryNode, RelationBelief, RelationStatus


# Native Video_Skills edge types that may become L1 navigation relations.
# Broad renames are forbidden: preserve candidate vs verified semantics.
NATIVE_IDENTITY_SOURCES = frozenset({"same_entity", "same_object", "reappears"})
NATIVE_SUPPORT_SOURCES = frozenset({"supports_observation"})
NATIVE_STATE_SOURCES = frozenset({"state_change"})
FORBIDDEN_L1_RELATIONS = frozenset({"explains", "enables", "causal_hint"})
SYMMETRIC_CANDIDATE_RELATIONS = frozenset(
    {
        "same_entity",
        "same_object",
        "same_instance_candidate",
        "reappears_candidate",
    }
)


@dataclass(frozen=True)
class L1StructuralEdgeReport:
    source_edges: int
    materialized_edges: int
    relation_counts: dict[str, int]
    unresolved_edge_ids: tuple[str, ...]
    skipped_state_change_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_edges": self.source_edges,
            "materialized_edges": self.materialized_edges,
            "relation_counts": self.relation_counts,
            "unresolved_edge_ids": list(self.unresolved_edge_ids),
            "skipped_state_change_ids": list(self.skipped_state_change_ids),
            "scope": "L1 navigation only; excludes candidate causality",
        }


def materialize_l1_structural_relations(
    source_graph: dict[str, Any],
    l1_nodes: list[MemoryNode],
) -> tuple[list[RelationBelief], L1StructuralEdgeReport]:
    """Project native L1 edges without collapsing distinct relation contracts."""

    raw_nodes = [
        value
        for value in source_graph.get("nodes") or []
        if isinstance(value, dict) and value.get("node_id")
    ]
    raw_by_id = {str(value["node_id"]): value for value in raw_nodes}
    memory_by_source = {
        str(node.source_node_id): node for node in l1_nodes if node.source_node_id
    }

    reference_neighbors: dict[str, list[str]] = {}
    raw_edges = [
        value for value in source_graph.get("edges") or [] if isinstance(value, dict)
    ]
    for edge in raw_edges:
        if str(edge.get("edge_type") or "") not in {
            "entity_mention",
            "state_of",
            "derived_from",
        }:
            continue
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        if src and dst:
            reference_neighbors.setdefault(src, []).append(dst)
            reference_neighbors.setdefault(dst, []).append(src)

    relations: list[RelationBelief] = []
    unresolved: list[str] = []
    skipped_state_change: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for index, edge in enumerate(raw_edges):
        source_type = str(edge.get("edge_type") or "")
        if source_type in FORBIDDEN_L1_RELATIONS:
            continue
        edge_id = str(edge.get("edge_id") or f"native-l1-edge:{index}")
        if source_type in NATIVE_STATE_SOURCES:
            # Phase 1: never admit unconditional state_change → state_transition.
            skipped_state_change.append(edge_id)
            unresolved.append(edge_id)
            continue
        relation = _admitted_relation(source_type, edge=edge)
        if relation is None:
            continue
        src, src_method = _resolve_endpoint(
            str(edge.get("src") or ""),
            raw_by_id=raw_by_id,
            memory_by_source=memory_by_source,
            reference_neighbors=reference_neighbors,
        )
        dst, dst_method = _resolve_endpoint(
            str(edge.get("dst") or ""),
            raw_by_id=raw_by_id,
            memory_by_source=memory_by_source,
            reference_neighbors=reference_neighbors,
        )
        if src is None or dst is None or src.node_id == dst.node_id:
            unresolved.append(edge_id)
            continue
        key = (src.node_id, dst.node_id, relation)
        reverse = (dst.node_id, src.node_id, relation)
        if key in seen or (relation in SYMMETRIC_CANDIDATE_RELATIONS and reverse in seen):
            continue
        seen.add(key)
        confidence = edge.get("confidence")
        probability = (
            max(0.0, min(1.0, float(confidence)))
            if isinstance(confidence, (int, float))
            else 0.8
        )
        deterministic_source = _is_deterministic_source(edge)
        relations.append(
            RelationBelief(
                edge_id=f"l1-structural:{edge_id}",
                src=src.node_id,
                dst=dst.node_id,
                relation_probabilities={relation: probability},
                status=(
                    RelationStatus.DETERMINISTIC
                    if deterministic_source and relation in {"same_object", "same_entity"}
                    else RelationStatus.UNCALIBRATED_PRIOR
                ),
                direction_confidence=(
                    0.5 if relation in SYMMETRIC_CANDIDATE_RELATIONS else probability
                ),
                evidence_refs=[src.node_id, dst.node_id],
                warrant=str(edge.get("text") or "") or None,
                provenance={
                    "producer": "memory_graph.l1_structural_edges",
                    "source_l1_edge_id": edge_id,
                    "source_edge_type": source_type,
                    "source_label": source_type,
                    "source_confidence": (
                        float(confidence)
                        if isinstance(confidence, (int, float))
                        else None
                    ),
                    "source_edge_producer": edge.get("producer"),
                    "endpoint_resolution": {
                        "src": src_method,
                        "dst": dst_method,
                    },
                    "relation_endpoint_layer": "L1_observation",
                    "navigation_only": True,
                    "candidate_causality": False,
                    "admission_tier": _admission_tier(relation),
                },
            )
        )
    counts = Counter(
        next(iter(relation.relation_probabilities)) for relation in relations
    )
    return relations, L1StructuralEdgeReport(
        source_edges=len(raw_edges),
        materialized_edges=len(relations),
        relation_counts=dict(sorted(counts.items())),
        unresolved_edge_ids=tuple(dict.fromkeys(unresolved)),
        skipped_state_change_ids=tuple(dict.fromkeys(skipped_state_change)),
    )


def _admitted_relation(source_type: str, *, edge: dict[str, Any]) -> str | None:
    if source_type in NATIVE_SUPPORT_SOURCES:
        return "observation_support"
    if source_type == "reappears":
        return "reappears_candidate"
    if source_type in {"same_object", "same_entity"}:
        if _is_identity_verified(edge):
            return source_type
        return "same_instance_candidate"
    return None


def _is_deterministic_source(edge: dict[str, Any]) -> bool:
    producer = str(edge.get("producer") or "")
    return producer.endswith("reference_structuralizer")


def _is_identity_verified(edge: dict[str, Any]) -> bool:
    """Verified identity requires an explicit verifier flag, not a source label."""

    if edge.get("identity_verified") is True:
        return True
    payload = edge.get("payload")
    return isinstance(payload, dict) and payload.get("identity_verified") is True


def _admission_tier(relation: str) -> str:
    if relation in {"same_object", "same_entity"}:
        return "verified_identity"
    if relation in {"same_instance_candidate", "reappears_candidate"}:
        return "identity_candidate"
    if relation == "observation_support":
        return "observation_support"
    return "other"


def _resolve_endpoint(
    raw_id: str,
    *,
    raw_by_id: dict[str, dict[str, Any]],
    memory_by_source: dict[str, MemoryNode],
    reference_neighbors: dict[str, list[str]],
) -> tuple[MemoryNode | None, str]:
    """Resolve an edge endpoint without clip-primary substitution."""

    if not raw_id:
        return None, "missing_endpoint_id"
    direct = memory_by_source.get(raw_id)
    if direct is not None:
        return direct, "direct_source_node"
    referenced = {
        memory_by_source[neighbor].node_id: memory_by_source[neighbor]
        for neighbor in reference_neighbors.get(raw_id, [])
        if neighbor in memory_by_source
    }
    preferred = _unique_preferred(list(referenced.values()))
    if preferred is not None:
        return preferred, "reference_neighbor"
    if raw_id in raw_by_id:
        # Explicit local/graph node exists but is not an admitted memory endpoint.
        return None, "unresolved_explicit_endpoint"
    return None, "unknown_endpoint"


def _unique_preferred(nodes: list[MemoryNode]) -> MemoryNode | None:
    for source_type in ("event", "observation", "state", "dialogue_span"):
        matches = [
            node
            for node in nodes
            if str(node.metadata.get("source_node_type") or "") == source_type
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return nodes[0] if len(nodes) == 1 else None
