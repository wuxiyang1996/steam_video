"""Build the retained L1 plus categorical L1.5 graph used by the main IWM arm."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Sequence

from memory_graph.consolidation import materialize_bounded_memory
from memory_graph.correlation_overlay import (
    CategoricalCorrelationEvaluator,
    build_categorical_correlation_overlay,
    project_legacy_overlay_correlations,
)
from memory_graph.selectstream_policy import plan_bounded_memory
from memory_graph.types import CausalTemporalOverlay, RelationBelief

from .contracts import RetainedEvidenceGraph, TemporalNavigationEdge


def build_retained_graph_from_legacy_overlay(
    overlay: CausalTemporalOverlay,
    *,
    capacity: int,
    redundancy: dict[str, float] | None = None,
    correlation_evaluator: CategoricalCorrelationEvaluator | None = None,
) -> RetainedEvidenceGraph:
    """Adapt, consolidate, and correlate without copying L1 evidence values.

    Legacy numeric relation values are retained only inside the compatibility
    source.  Admission into the new navigation overlay is categorical.
    """

    if not overlay.l1_observations:
        raise ValueError("legacy overlay contains no L1 evidence")
    policy_relations = [
        *overlay.l1_structural_relations,
        *_project_event_relations_for_policy(overlay),
    ]
    decision = plan_bounded_memory(
        overlay.l1_observations,
        policy_relations,
        capacity=capacity,
        redundancy=redundancy,
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
    correlations = build_categorical_correlation_overlay(
        consolidated.nodes,
        evaluator=correlation_evaluator,
        seed_edges=legacy_edges,
        overlay_id=f"{overlay.overlay_id}:categorical-correlations",
    )
    temporal = _temporal_navigation_edges(consolidated.relations)
    return RetainedEvidenceGraph(
        graph_id=f"{overlay.overlay_id}:retained-full-iwm",
        nodes=consolidated.nodes,
        temporal_edges=temporal,
        correlation_edges=correlations.edges,
        capacity=capacity,
        metadata={
            "source_overlay_id": overlay.overlay_id,
            "observation_end_s": overlay.metadata.get("observation_end_s"),
            "l1_writer": "surprise_driven_when_configured",
            "consolidation": consolidated.audit_dict(),
            "correlation_build": correlations.to_dict()["build_audit"],
            "navigation_contract": "single_cursor_all_legal_actions_no_top_k",
            "relation_verifier_available": False,
            "question_independent": True,
        },
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
