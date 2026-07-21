"""Leakage-safe full retained-graph input construction for the IWM."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from memory_graph.types import MemoryNode

from .contracts import (
    CursorBeliefState,
    IWMGraphInput,
    LegalGraphAction,
    NodeKey,
    NodeModelView,
    RetainedEvidenceGraph,
)


def build_iwm_graph_input(
    belief: CursorBeliefState,
    graph: RetainedEvidenceGraph,
    legal_actions: tuple[LegalGraphAction, ...],
) -> IWMGraphInput:
    """Expose all node keys but full values only for executed reads."""

    acquired = set(belief.acquired_evidence)
    imagined = set(belief.imagined_evidence)
    views = tuple(
        NodeModelView(
            key=_node_key(node),
            acquired=node.node_id in acquired,
            evidence_value=(
                _evidence_value(node)
                if node.node_id in acquired and node.node_id not in imagined
                else None
            ),
        )
        for node in graph.nodes
    )
    return IWMGraphInput(
        question=belief.question,
        current_node_id=belief.current_node_id,
        nodes=views,
        temporal_edges=graph.temporal_edges,
        correlation_edges=graph.correlation_edges,
        legal_actions=legal_actions,
        acquired_evidence=belief.acquired_evidence,
        missing_roles=belief.missing_roles,
        contradictions=belief.contradictions,
        answerability=belief.answerability,
    )


def graph_input_to_categorical_payload(graph_input: IWMGraphInput) -> dict[str, Any]:
    """JSON-safe payload containing references, never raw embedding vectors."""

    return {
        "question": graph_input.question,
        "current_node_id": graph_input.current_node_id,
        "nodes": [
            {
                "key": asdict(view.key),
                "acquired": view.acquired,
                "evidence_value": view.evidence_value,
            }
            for view in graph_input.nodes
        ],
        "temporal_edges": [asdict(edge) for edge in graph_input.temporal_edges],
        "correlation_edges": [edge.to_dict() for edge in graph_input.correlation_edges],
        "legal_actions": [
            {
                **asdict(action),
                "kind": action.kind.value,
            }
            for action in graph_input.legal_actions
        ],
        "belief": {
            "acquired_evidence": list(graph_input.acquired_evidence),
            "missing_roles": list(graph_input.missing_roles),
            "contradictions": list(graph_input.contradictions),
            "answerability": graph_input.answerability.value,
        },
        "contract": {
            "all_retained_nodes_present": True,
            "top_k_applied": False,
            "unread_evidence_values_hidden": True,
            "model_output": "categorical_transition_and_pairwise_preference_only",
        },
    }


def _node_key(node: MemoryNode) -> NodeKey:
    embedding = asdict(node.embedding_ref) if node.embedding_ref is not None else None
    tags = tuple(
        sorted(
            {
                str(value)
                for value in (
                    node.metadata.get("modality"),
                    node.metadata.get("action_kind"),
                    node.metadata.get("source_type"),
                )
                if value
            }
        )
    )
    return NodeKey(
        node_id=node.node_id,
        node_type=node.node_type,
        start_s=node.time_span.start_s,
        end_s=node.time_span.end_s,
        embedding_ref=embedding,
        structural_tags=tags,
    )


def _evidence_value(node: MemoryNode) -> str | None:
    if node.text:
        return node.text
    for name in ("grounded_descriptor", "predicate", "caption", "transcript"):
        if node.metadata.get(name):
            return str(node.metadata[name])
    return None
