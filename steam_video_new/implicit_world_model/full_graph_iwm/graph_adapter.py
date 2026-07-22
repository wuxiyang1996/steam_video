"""Build semantic-temporal L1 and soft-correlation L1.5 navigation memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from memory_graph.consolidation import materialize_bounded_memory
from memory_graph.correlation_overlay import (
    CategoricalCorrelationEvaluator,
    CorrelationStatus,
    project_legacy_overlay_correlations,
)
from memory_graph.multichannel_correlation import build_multichannel_pair_audit
from memory_graph.selectstream_policy import plan_bounded_memory
from memory_graph.soft_correlation import (
    SEMANTIC_EQUIVALENCE_COSINE,
    SoftCorrelationAdmissionPolicy,
    build_soft_semantic_correlations,
)
from memory_graph.types import CausalTemporalOverlay, RelationBelief

from .contracts import RetainedEvidenceGraph, TemporalNavigationEdge


@dataclass(frozen=True)
class L1L15NavigationBuildResult:
    """Compiled graph plus a separate, question-independent pair audit."""

    graph: RetainedEvidenceGraph
    correlation_pair_audit: dict[str, object]
    multichannel_pair_audit: dict[str, object]
    source_l1_fingerprint: str
    retained_l1_fingerprint: str | None


def build_l1_l15_navigation_graph(
    overlay: CausalTemporalOverlay,
    *,
    capacity: int,
    redundancy: dict[str, float] | None = None,
    embedding_vectors: dict[str, Sequence[float]] | None = None,
    admission_policy: SoftCorrelationAdmissionPolicy | None = None,
) -> RetainedEvidenceGraph:
    """Compatibility API returning only the compiled navigation graph."""

    return compile_l1_l15_navigation_graph(
        overlay,
        capacity=capacity,
        redundancy=redundancy,
        embedding_vectors=embedding_vectors,
        admission_policy=admission_policy,
    ).graph


def compile_l1_l15_navigation_graph(
    overlay: CausalTemporalOverlay,
    *,
    capacity: int,
    redundancy: dict[str, float] | None = None,
    embedding_vectors: dict[str, Sequence[float]] | None = None,
    admission_policy: SoftCorrelationAdmissionPolicy | None = None,
) -> L1L15NavigationBuildResult:
    """Build fixed-capacity semantic-temporal L1 plus soft L1.5 correlation.

    Embedding similarity is a soft navigation feature, never a probability or
    a verified relation.  Only explicitly verified legacy relations enter the
    separate strict-relation layer.
    """

    if not overlay.l1_observations:
        raise ValueError("legacy overlay contains no L1 evidence")
    source_l1_fingerprint = _nodes_fingerprint(overlay.l1_observations)
    source_vectors = (
        {
            str(key): tuple(float(value) for value in vector)
            for key, vector in embedding_vectors.items()
        }
        if embedding_vectors is not None
        else _load_node_embeddings(overlay.l1_observations)
    )
    embedding_redundancy, pairwise_redundancy = _embedding_redundancy(
        overlay.l1_observations,
        source_vectors,
    )
    policy_relations = [
        *overlay.l1_structural_relations,
        *_project_event_relations_for_policy(overlay),
    ]
    decision = plan_bounded_memory(
        overlay.l1_observations,
        policy_relations,
        capacity=capacity,
        redundancy=redundancy or embedding_redundancy,
        pairwise_redundancy=pairwise_redundancy,
        merge_redundancy_threshold=SEMANTIC_EQUIVALENCE_COSINE,
    )
    consolidated = materialize_bounded_memory(
        overlay.l1_observations,
        policy_relations,
        decision,
    )
    retained_ids = tuple(node.node_id for node in consolidated.nodes)
    legacy_edges = project_legacy_overlay_correlations(
        overlay,
        retained_node_ids=retained_ids,
        id_map=consolidated.id_map,
    )
    verified_before_merge = tuple(
        edge for edge in legacy_edges if edge.status is CorrelationStatus.VERIFIED
    )
    verified_relations = tuple(
        edge
        for edge in verified_before_merge
        if edge.src not in consolidated.merged_lineage
        and edge.dst not in consolidated.merged_lineage
    )
    temporal = _temporal_navigation_edges(consolidated.relations)
    retained_vectors = _retained_embedding_vectors(
        consolidated.nodes,
        source_vectors,
        consolidated.merged_lineage,
    )
    embedding_models = sorted(
        {
            node.embedding_ref.model
            for node in overlay.l1_observations
            if node.embedding_ref is not None and node.embedding_ref.model
        }
    )
    score_source = (
        "provided_embedding_vectors"
        if embedding_vectors is not None
        else (
            embedding_models[0]
            if len(embedding_models) == 1
            else "mixed_or_unspecified_embedding_models"
        )
    )
    missing_embeddings = sorted(set(retained_ids) - set(retained_vectors))
    soft_build = (
        build_soft_semantic_correlations(
            consolidated.nodes,
            retained_vectors,
            score_source=score_source,
            excluded_pairs=[(edge.src, edge.dst) for edge in temporal],
            admission_policy=admission_policy,
        )
        if consolidated.nodes and not missing_embeddings
        else None
    )
    graph = RetainedEvidenceGraph(
        graph_id=f"{overlay.overlay_id}:semantic-temporal-l1:soft-l1.5",
        nodes=consolidated.nodes,
        temporal_edges=temporal,
        correlation_edges=soft_build.edges if soft_build is not None else (),
        capacity=capacity,
        verified_relations=verified_relations,
        metadata={
            "source_overlay_id": overlay.overlay_id,
            "observation_end_s": (
                overlay.metadata.get("observation_end_s")
                if overlay.metadata.get("observation_end_s") is not None
                else overlay.metadata.get("observation_horizon_s")
            ),
            "l1_writer": "surprise_driven_when_configured",
            "l1_contract": "grounded_semantic_nodes_plus_deterministic_temporal_backbone",
            "l1_5_contract": (
                "soft_semantic_navigation_plus_non_admitting_grounded_pair_descriptors"
            ),
            "source_l1_fingerprint": source_l1_fingerprint,
            "retained_l1_fingerprint": (
                soft_build.l1_fingerprint if soft_build is not None else None
            ),
            "correlation_builder_mutates_source_l1": False,
            "semantic_duplicate_coalescing": {
                "enabled": bool(pairwise_redundancy),
                "criterion": (
                    "adjacent_embedding_cosine_at_least_"
                    f"{SEMANTIC_EQUIVALENCE_COSINE}"
                ),
                "question_independent": True,
            },
            "consolidation": consolidated.audit_dict(),
            "correlation_build": (
                soft_build.audit_dict()
                if soft_build is not None
                else {
                    "status": "unavailable_missing_embeddings",
                    "missing_node_ids": missing_embeddings,
                    "top_k_applied": False,
                    "question_independent": True,
                }
            ),
            "strict_relation_build": {
                "verified_relation_count": len(verified_relations),
                "merged_endpoint_relations_requiring_reverification": (
                    len(verified_before_merge) - len(verified_relations)
                ),
                "candidate_relations_are_navigation_edges": False,
            },
            "navigation_contract": "single_cursor_temporal_plus_soft_correlation_no_top_k",
            "relation_verifier_available": bool(verified_relations),
            "question_independent": True,
        },
    )
    if _nodes_fingerprint(overlay.l1_observations) != source_l1_fingerprint:
        raise RuntimeError("L1.5 compilation mutated the frozen source L1 nodes")
    pair_audit = (
        soft_build.pair_audit_dict()
        if soft_build is not None
        else {
            "schema_version": "steam-soft-l1.5-pair-audit/v0.1",
            "status": "unavailable_missing_embeddings",
            "source_l1_fingerprint": source_l1_fingerprint,
            "missing_node_ids": missing_embeddings,
            "question_independent": True,
            "contains_question_or_answer": False,
        }
    )
    pair_audit["source_l1_fingerprint"] = source_l1_fingerprint
    pair_audit["graph_id"] = graph.graph_id
    multichannel_pair_audit = (
        build_multichannel_pair_audit(
            consolidated.nodes,
            soft_build,
            temporal_pairs=[(edge.src, edge.dst) for edge in temporal],
        )
        if soft_build is not None
        else {
            "schema_version": "steam-l1.5-multichannel-pair-audit/v0.1",
            "status": "unavailable_missing_embeddings",
            "source_l1_fingerprint": source_l1_fingerprint,
            "question_independent": True,
            "contains_question_or_answer": False,
            "grounded_descriptors_admit_edges": False,
            "learned_edge_selector_present": False,
            "iwm_training_performed": False,
        }
    )
    multichannel_pair_audit["source_l1_fingerprint"] = source_l1_fingerprint
    multichannel_pair_audit["graph_id"] = graph.graph_id
    return L1L15NavigationBuildResult(
        graph=graph,
        correlation_pair_audit=pair_audit,
        multichannel_pair_audit=multichannel_pair_audit,
        source_l1_fingerprint=source_l1_fingerprint,
        retained_l1_fingerprint=(
            soft_build.l1_fingerprint if soft_build is not None else None
        ),
    )


def build_retained_graph_from_legacy_overlay(
    overlay: CausalTemporalOverlay,
    *,
    capacity: int,
    redundancy: dict[str, float] | None = None,
    correlation_evaluator: CategoricalCorrelationEvaluator | None = None,
    embedding_vectors: dict[str, Sequence[float]] | None = None,
) -> RetainedEvidenceGraph:
    """Compatibility alias for the semantic L1 plus soft L1.5 builder."""

    if correlation_evaluator is not None:
        raise ValueError(
            "categorical relation proposals are no longer navigation correlations; "
            "run them as an optional strict-relation diagnostic"
        )
    return build_l1_l15_navigation_graph(
        overlay,
        capacity=capacity,
        redundancy=redundancy,
        embedding_vectors=embedding_vectors,
    )


def _project_event_relations_for_policy(
    overlay: CausalTemporalOverlay,
) -> list[RelationBelief]:
    sources = {
        node.node_id: tuple(node.source_segments) for node in overlay.atomic_events
    }
    projected: list[RelationBelief] = []
    for relation in overlay.relations:
        for src in sources.get(relation.src, ()):
            for dst in sources.get(relation.dst, ()):
                if src == dst:
                    continue
                digest = hashlib.sha256(
                    f"{relation.edge_id}\x1f{src}\x1f{dst}".encode("utf-8")
                ).hexdigest()[:16]
                provenance = dict(relation.provenance)
                witness = provenance.get("causal_witness")
                if isinstance(witness, dict):
                    provenance["causal_witness"] = {
                        **witness,
                        "witness_id": f"{witness.get('witness_id', 'witness')}:l1:{digest}",
                        "cause_event_id": src,
                        "effect_event_id": dst,
                        "evidence_refs": [src, dst],
                        "minimal_support_set": [src, dst],
                        "mechanism_event_id": None,
                    }
                projected.append(
                    replace(
                        relation,
                        edge_id=f"relation:l1-policy:{digest}",
                        src=src,
                        dst=dst,
                        evidence_refs=[src, dst],
                        provenance=provenance,
                    )
                )
    return projected


def _temporal_navigation_edges(
    relations: Sequence[RelationBelief],
) -> tuple[TemporalNavigationEdge, ...]:
    temporal_names = ("temporal_next", "before", "overlaps", "during")
    edges: dict[tuple[str, str, str], TemporalNavigationEdge] = {}
    for relation in relations:
        present = [
            name for name in temporal_names if name in relation.relation_probabilities
        ]
        # ``temporal_next`` already entails local ``before`` for navigation;
        # retaining both would create two functionally identical cursor moves.
        names = ["temporal_next"] if "temporal_next" in present else present
        for name in names:
            key = (relation.src, relation.dst, name)
            edges[key] = TemporalNavigationEdge(
                edge_id=f"{relation.edge_id}:{name}",
                src=relation.src,
                dst=relation.dst,
                relation=name,
            )
    return tuple(
        sorted(edges.values(), key=lambda edge: (edge.src, edge.dst, edge.relation))
    )


def _load_node_embeddings(
    nodes: Sequence[object],
) -> dict[str, tuple[float, ...]]:
    try:
        import numpy as np
    except ImportError:
        return {}
    matrices: dict[Path, object] = {}
    checksums: dict[Path, str] = {}
    vectors: dict[str, tuple[float, ...]] = {}
    for node in nodes:
        ref = getattr(node, "embedding_ref", None)
        if ref is None or ref.row_index is None:
            continue
        path = Path(ref.path).expanduser()
        if not path.is_file():
            continue
        checksum = checksums.get(path)
        if checksum is None:
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            checksums[path] = checksum
        if ref.checksum and checksum != ref.checksum:
            raise ValueError(f"embedding checksum mismatch for {node.node_id}")
        matrix = matrices.get(path)
        if matrix is None:
            matrix = np.load(path, mmap_mode="r")
            matrices[path] = matrix
        if (
            len(matrix.shape) != 2
            or ref.row_index < 0
            or ref.row_index >= int(matrix.shape[0])
        ):
            raise ValueError(f"invalid embedding matrix alignment for {node.node_id}")
        row = tuple(float(value) for value in matrix[ref.row_index])
        if len(row) != int(ref.dimension):
            raise ValueError(f"embedding dimension mismatch for {node.node_id}")
        vectors[str(node.node_id)] = row
    return vectors


def _retained_embedding_vectors(
    nodes: Sequence[object],
    source_vectors: dict[str, Sequence[float]],
    merged_lineage: dict[str, tuple[str, ...]],
) -> dict[str, tuple[float, ...]]:
    vectors: dict[str, tuple[float, ...]] = {}
    for node in nodes:
        node_id = str(getattr(node, "node_id"))
        if node_id in source_vectors:
            vectors[node_id] = tuple(float(value) for value in source_vectors[node_id])
            continue
        lineage = merged_lineage.get(node_id, ())
        members = [
            source_vectors[source] for source in lineage if source in source_vectors
        ]
        if len(members) != len(lineage) or not members:
            continue
        dimensions = {len(vector) for vector in members}
        if len(dimensions) != 1:
            raise ValueError(f"merged embedding dimensions differ for {node_id}")
        mean = tuple(
            sum(float(vector[index]) for vector in members) / len(members)
            for index in range(len(members[0]))
        )
        norm = math.sqrt(sum(value * value for value in mean))
        if norm > 1e-12:
            vectors[node_id] = tuple(value / norm for value in mean)
    return vectors


def _embedding_redundancy(
    nodes: Sequence[object],
    vectors: dict[str, Sequence[float]],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Measure local semantic repetition for L1 coalescing and retention.

    Only adjacent nodes can be coalesced.  Near-identity is treated as duplicate
    suppression, not as an inferred fact relation.  Lower similarities still
    influence retention utility but never trigger a merge.
    """

    ordered = sorted(
        nodes,
        key=lambda node: (
            getattr(node, "time_span").start_s,
            getattr(node, "time_span").end_s,
            str(getattr(node, "node_id")),
        ),
    )
    normalized: dict[str, tuple[float, ...]] = {}
    for node in ordered:
        node_id = str(getattr(node, "node_id"))
        vector = tuple(float(value) for value in vectors.get(node_id, ()))
        norm = math.sqrt(sum(value * value for value in vector))
        if vector and norm > 1e-12:
            normalized[node_id] = tuple(value / norm for value in vector)
    node_scores: dict[str, float] = {}
    pair_scores: dict[tuple[str, str], float] = {}
    for left, right in zip(ordered, ordered[1:]):
        left_id = str(getattr(left, "node_id"))
        right_id = str(getattr(right, "node_id"))
        left_vector = normalized.get(left_id)
        right_vector = normalized.get(right_id)
        if left_vector is None or right_vector is None:
            continue
        if len(left_vector) != len(right_vector):
            raise ValueError("L1 embedding dimensions must match")
        similarity = max(
            0.0,
            min(1.0, sum(a * b for a, b in zip(left_vector, right_vector))),
        )
        pair_scores[(left_id, right_id)] = similarity
        node_scores[left_id] = max(node_scores.get(left_id, 0.0), similarity)
        node_scores[right_id] = max(node_scores.get(right_id, 0.0), similarity)
    return node_scores, pair_scores


def _nodes_fingerprint(nodes: Sequence[object]) -> str:
    payload = [
        getattr(node, "to_dict")()
        for node in sorted(
            nodes,
            key=lambda value: (
                getattr(value, "time_span").start_s,
                getattr(value, "time_span").end_s,
                str(getattr(value, "node_id")),
            ),
        )
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
