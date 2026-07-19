"""Temporal skeleton and sparse embedding-based relation candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from .types import (
    MemoryGraph,
    MemoryNode,
    RelationBelief,
    RelationStatus,
    RelationType,
)


PROBABILISTIC_RELATIONS = (
    RelationType.SAME_ENTITY,
    RelationType.STATE_TRANSITION,
    RelationType.TRANSITION_SUPPORT,
    RelationType.RESPONSE_CANDIDATE,
    RelationType.EXPLAINS,
    RelationType.ENABLES,
    RelationType.CONTRADICTS,
)


class RelationScorer(Protocol):
    """A trained or teacher-backed scorer for non-deterministic relations."""

    status: RelationStatus
    producer: str

    def score(
        self,
        src: MemoryNode,
        dst: MemoryNode,
        features: dict[str, float],
    ) -> tuple[dict[str, float], str | None]:
        """Return relation probabilities and an optional evidence warrant."""


@dataclass
class LinearRelationScorer:
    """Small relation head loaded from trained weights.

    This class intentionally has no built-in weights. Cosine similarity alone
    is not a defensible causal posterior. Callers must provide learned or
    teacher-fitted coefficients and state whether they were calibrated.
    """

    weights: dict[str, dict[str, float]]
    biases: dict[str, float]
    calibrated: bool = False
    producer: str = "memory_graph.linear_relation_scorer"

    @property
    def status(self) -> RelationStatus:
        if self.calibrated:
            return RelationStatus.CALIBRATED_POSTERIOR
        return RelationStatus.UNCALIBRATED_PRIOR

    def score(
        self,
        src: MemoryNode,
        dst: MemoryNode,
        features: dict[str, float],
    ) -> tuple[dict[str, float], str | None]:
        probabilities: dict[str, float] = {}
        for relation in PROBABILISTIC_RELATIONS:
            name = relation.value
            coefficients = self.weights.get(name)
            if coefficients is None:
                continue
            logit = float(self.biases.get(name, 0.0))
            logit += sum(float(weight) * float(features.get(feature, 0.0)) for feature, weight in coefficients.items())
            probabilities[name] = _sigmoid(logit)
        return probabilities, None


def build_memory_graph(
    *,
    graph_id: str,
    example_id: str,
    video_id: str,
    nodes: list[MemoryNode],
    embeddings: Sequence[Sequence[float]] | None = None,
    relation_scorer: RelationScorer | None = None,
    top_k_candidates: int = 4,
    max_before_neighbors: int = 4,
    temporal_tolerance_s: float = 1e-3,
) -> MemoryGraph:
    """Build deterministic temporal edges and optional learned relation priors."""
    ordered = sorted(nodes, key=lambda node: (node.time_span.start_s, node.time_span.end_s, node.node_id))
    relations = build_temporal_relations(
        ordered,
        max_before_neighbors=max_before_neighbors,
        tolerance_s=temporal_tolerance_s,
    )

    if relation_scorer is not None:
        if embeddings is None:
            raise ValueError("embeddings are required when relation_scorer is provided")
        if len(embeddings) != len(nodes):
            raise ValueError("embeddings must align with the input nodes")
        embedding_by_id = {node.node_id: embeddings[index] for index, node in enumerate(nodes)}
        relations.extend(
            build_probabilistic_relations(
                ordered,
                embedding_by_id=embedding_by_id,
                scorer=relation_scorer,
                top_k=top_k_candidates,
            )
        )

    endpoint_layer = (
        "L1.5_atomic_event"
        if ordered and all(node.node_type == "atomic_event" for node in ordered)
        else "legacy_unverified_node"
    )
    return MemoryGraph(
        graph_id=graph_id,
        example_id=example_id,
        video_id=video_id,
        nodes=ordered,
        relations=relations,
        metadata={
            "node_event_assumption": (
                "l1_observation_container_with_l1_5_atomic_overlay"
                if endpoint_layer == "L1.5_atomic_event"
                else "legacy_nodes_not_valid_for_candidate_causality"
            ),
            "relation_endpoint_layer": endpoint_layer,
            "embedding_model": nodes[0].embedding_ref.model if nodes and nodes[0].embedding_ref else None,
            "top_k_relation_candidates": top_k_candidates,
            "temporal_relations": "deterministic_interval_constraints",
            "probabilistic_relation_status": relation_scorer.status.value if relation_scorer else "not_scored",
        },
    )


def build_temporal_relations(
    nodes: list[MemoryNode],
    *,
    max_before_neighbors: int = 4,
    tolerance_s: float = 1e-3,
) -> list[RelationBelief]:
    """Build a sparse deterministic interval graph.

    ``temporal_next`` links adjacent nodes in sorted order. ``before`` is
    limited to the next few nodes to avoid a dense transitive closure.
    """
    relations: list[RelationBelief] = []
    seen: set[tuple[str, str, str]] = set()

    for index, src in enumerate(nodes):
        for offset, dst in enumerate(nodes[index + 1 : index + 1 + max_before_neighbors], start=1):
            if offset == 1:
                _append_deterministic(
                    relations,
                    seen,
                    src,
                    dst,
                    RelationType.TEMPORAL_NEXT,
                    direction_confidence=1.0,
                    warrant="Adjacent memory events after sorting by time span.",
                )

            if src.time_span.end_s <= dst.time_span.start_s + tolerance_s:
                _append_deterministic(
                    relations,
                    seen,
                    src,
                    dst,
                    RelationType.BEFORE,
                    direction_confidence=1.0,
                    warrant="The source interval ends no later than the destination interval starts.",
                )
                continue

            if src.time_span.contains(dst.time_span, tolerance_s=tolerance_s):
                _append_deterministic(
                    relations,
                    seen,
                    dst,
                    src,
                    RelationType.DURING,
                    direction_confidence=1.0,
                    warrant="The source interval is contained by the destination interval.",
                )
                continue

            if dst.time_span.contains(src.time_span, tolerance_s=tolerance_s):
                _append_deterministic(
                    relations,
                    seen,
                    src,
                    dst,
                    RelationType.DURING,
                    direction_confidence=1.0,
                    warrant="The source interval is contained by the destination interval.",
                )
                continue

            if src.time_span.overlaps(dst.time_span, tolerance_s=tolerance_s):
                _append_deterministic(
                    relations,
                    seen,
                    src,
                    dst,
                    RelationType.OVERLAPS,
                    direction_confidence=0.5,
                    warrant="The two observed intervals overlap; this relation is symmetric.",
                )
    return relations


def build_probabilistic_relations(
    nodes: list[MemoryNode],
    *,
    embedding_by_id: dict[str, Sequence[float]],
    scorer: RelationScorer,
    top_k: int = 4,
) -> list[RelationBelief]:
    """Score sparse semantic neighbors while retaining temporal direction."""
    pairs = select_top_k_pairs(nodes, embedding_by_id=embedding_by_id, top_k=top_k)
    relations: list[RelationBelief] = []
    for src, dst, cosine_similarity in pairs:
        features = relation_features(src, dst, cosine_similarity=cosine_similarity)
        probabilities, warrant = scorer.score(src, dst, features)
        if not probabilities:
            continue
        relations.append(
            RelationBelief(
                edge_id=f"relation:{src.node_id}->{dst.node_id}:candidate",
                src=src.node_id,
                dst=dst.node_id,
                relation_probabilities=probabilities,
                status=scorer.status,
                direction_confidence=features["forward_order"],
                warrant=warrant,
                evidence_refs=_evidence_refs(src, dst),
                provenance={
                    "producer": scorer.producer,
                    "candidate_generation": "top_k_embedding_neighbors",
                },
                features=features,
            )
        )
    return relations


def select_top_k_pairs(
    nodes: list[MemoryNode],
    *,
    embedding_by_id: dict[str, Sequence[float]],
    top_k: int,
) -> list[tuple[MemoryNode, MemoryNode, float]]:
    """Return unique time-directed pairs proposed by embedding similarity."""
    if top_k <= 0:
        return []
    candidates: dict[tuple[str, str], tuple[MemoryNode, MemoryNode, float]] = {}
    for node in nodes:
        source_embedding = embedding_by_id[node.node_id]
        scored: list[tuple[float, MemoryNode]] = []
        for other in nodes:
            if other.node_id == node.node_id:
                continue
            similarity = cosine(source_embedding, embedding_by_id[other.node_id])
            scored.append((similarity, other))
        scored.sort(key=lambda item: (-item[0], item[1].node_id))
        for similarity, other in scored[:top_k]:
            src, dst = _time_direct(node, other)
            candidates[(src.node_id, dst.node_id)] = (src, dst, similarity)
    return sorted(
        candidates.values(),
        key=lambda item: (item[0].time_span.start_s, item[1].time_span.start_s, item[0].node_id, item[1].node_id),
    )


def relation_features(
    src: MemoryNode,
    dst: MemoryNode,
    *,
    cosine_similarity: float,
    proximity_scale_s: float = 30.0,
) -> dict[str, float]:
    gap_s = max(0.0, dst.time_span.start_s - src.time_span.end_s)
    overlap_s = max(
        0.0,
        min(src.time_span.end_s, dst.time_span.end_s)
        - max(src.time_span.start_s, dst.time_span.start_s),
    )
    src_duration = max(src.time_span.end_s - src.time_span.start_s, 1e-6)
    dst_duration = max(dst.time_span.end_s - dst.time_span.start_s, 1e-6)
    return {
        "cosine_similarity": cosine_similarity,
        "temporal_gap_s": gap_s,
        "temporal_proximity": math.exp(-gap_s / max(proximity_scale_s, 1e-6)),
        "overlap_s": overlap_s,
        "duration_ratio": min(src_duration, dst_duration) / max(src_duration, dst_duration),
        "forward_order": 1.0 if src.time_span.start_s <= dst.time_span.start_s else 0.0,
    }


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("cosine vectors must have the same dimension")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _append_deterministic(
    relations: list[RelationBelief],
    seen: set[tuple[str, str, str]],
    src: MemoryNode,
    dst: MemoryNode,
    relation: RelationType,
    *,
    direction_confidence: float,
    warrant: str,
) -> None:
    key = (src.node_id, dst.node_id, relation.value)
    if key in seen or src.node_id == dst.node_id:
        return
    seen.add(key)
    relations.append(
        RelationBelief(
            edge_id=f"relation:{src.node_id}->{dst.node_id}:{relation.value}",
            src=src.node_id,
            dst=dst.node_id,
            relation_probabilities={relation.value: 1.0},
            status=RelationStatus.DETERMINISTIC,
            direction_confidence=direction_confidence,
            warrant=warrant,
            evidence_refs=_evidence_refs(src, dst),
            provenance={"producer": "memory_graph.graph_builder.build_temporal_relations"},
        )
    )


def _time_direct(left: MemoryNode, right: MemoryNode) -> tuple[MemoryNode, MemoryNode]:
    left_key = (left.time_span.start_s, left.time_span.end_s, left.node_id)
    right_key = (right.time_span.start_s, right.time_span.end_s, right.node_id)
    return (left, right) if left_key <= right_key else (right, left)


def _evidence_refs(src: MemoryNode, dst: MemoryNode) -> list[str]:
    return list(dict.fromkeys(src.source_segments + dst.source_segments))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
