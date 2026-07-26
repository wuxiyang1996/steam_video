"""Build a typed proposal graph from frozen v1 navigation artifacts."""

from __future__ import annotations

from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
    RetainedEvidenceGraph,
)

from .contracts import (
    NavigationGraph,
    NavigationProposal,
    ProposalCalibration,
    ProposalKind,
)


def from_retained_graph(
    graph: RetainedEvidenceGraph,
    *,
    entry_node_ids: tuple[str, ...] = (),
) -> NavigationGraph:
    temporal = tuple(
        NavigationProposal(
            proposal_id=edge.edge_id,
            src=edge.src,
            dst=edge.dst,
            kind=(
                ProposalKind.EVENT_CONTINUATION_CANDIDATE
                if edge.relation in {"temporal_next", "before"}
                else ProposalKind.TEMPORAL
            ),
            bidirectional=True,
            evidence_refs=(edge.src, edge.dst),
            descriptor=(edge.relation,),
            calibration=ProposalCalibration.STRUCTURAL,
        )
        for edge in graph.temporal_edges
    )
    correlations = tuple(
        NavigationProposal(
            proposal_id=edge.edge_id,
            src=edge.src,
            dst=edge.dst,
            kind=(
                ProposalKind.SEMANTIC_RECURRENCE
                if edge.channel == "semantic_recurrence"
                else ProposalKind.SEMANTIC_NEIGHBOR
            ),
            bidirectional=True,
            evidence_refs=tuple(edge.evidence_refs),
            descriptor=(edge.channel,),
            calibration=ProposalCalibration.UNCALIBRATED,
            audit_metadata={
                "legacy_soft_affinity_present": True,
                "affinity_exposed_to_model": False,
                "affinity_is_probability": False,
            },
        )
        for edge in graph.correlation_edges
    )
    return NavigationGraph(
        graph_id=f"navigation-v2:{graph.graph_id}",
        node_ids=tuple(node.node_id for node in graph.nodes),
        proposals=(*temporal, *correlations),
        entry_node_ids=entry_node_ids,
        metadata={
            "schema_version": "steam-reasoning-v2-navigation/v0.1",
            "source_graph_id": graph.graph_id,
            "top_k_applied": False,
            "correlation_is_navigation_proposal_not_fact": True,
        },
    )
