"""Leakage-safe full retained-graph input construction for the IWM."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from memory_graph.types import MemoryNode

from .action_compiler import visible_graph_nodes
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
        for node in visible_graph_nodes(graph)
    )
    return IWMGraphInput(
        question=belief.question,
        current_node_id=belief.current_node_id,
        nodes=views,
        temporal_edges=graph.temporal_edges,
        correlation_edges=graph.correlation_edges,
        candidate_edges=graph.candidate_edges,
        verified_relations=graph.verified_relations,
        legal_actions=legal_actions,
        acquired_evidence=belief.acquired_evidence,
        missing_roles=belief.missing_roles,
        contradictions=belief.contradictions,
        answerability=belief.answerability,
        required_roles=belief.required_roles,
        grounded_role_evidence=belief.grounded_role_evidence,
    )


def graph_input_to_categorical_payload(graph_input: IWMGraphInput) -> dict[str, Any]:
    """JSON-safe payload containing references, never raw embedding vectors."""

    return {
        "question": graph_input.question,
        "current_node_id": graph_input.current_node_id,
        "nodes": [
            {
                "key": {
                    "node_id": view.key.node_id,
                    "node_type": view.key.node_type,
                    "start_s": view.key.start_s,
                    "end_s": view.key.end_s,
                    "semantic_key": view.key.semantic_key,
                    "structural_tags": list(view.key.structural_tags),
                    "embedding_available": view.key.embedding_ref is not None,
                },
                "acquired": view.acquired,
                "evidence_value": view.evidence_value,
            }
            for view in graph_input.nodes
        ],
        "temporal_edges": [asdict(edge) for edge in graph_input.temporal_edges],
        "correlation_edges": [
            {
                "edge_id": edge.edge_id,
                "src": edge.src,
                "dst": edge.dst,
                "channel": edge.channel,
                "direction": "bidirectional",
            }
            for edge in graph_input.correlation_edges
        ],
        "candidate_edges": [edge.to_dict() for edge in graph_input.candidate_edges],
        "verified_relations": [
            {
                "edge_id": edge.edge_id,
                "src": edge.src,
                "dst": edge.dst,
                "relation": edge.relation.value,
                "status": edge.status.value,
            }
            for edge in graph_input.verified_relations
        ],
        "legal_actions": [
            {
                **asdict(action),
                "kind": action.kind.value,
            }
            for action in graph_input.legal_actions
        ],
        "belief": {
            "acquired_evidence": list(graph_input.acquired_evidence),
            "required_roles": list(graph_input.required_roles),
            "missing_roles": list(graph_input.missing_roles),
            "grounded_role_evidence": [
                {"role": role, "node_id": node_id}
                for role, node_id in graph_input.grounded_role_evidence
            ],
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
    embedding = (
        {
            "model": node.embedding_ref.model,
            "dimension": node.embedding_ref.dimension,
            "normalized": node.embedding_ref.normalized,
            "row_index": node.embedding_ref.row_index,
            "checksum": node.embedding_ref.checksum,
        }
        if node.embedding_ref is not None
        else None
    )
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
        semantic_key=_semantic_key(node),
        embedding_ref=embedding,
        structural_tags=tags,
    )


def _semantic_key(node: MemoryNode) -> str:
    """Expose a discriminative grounded address, not the full evidence value."""

    metadata = node.metadata
    parts: list[str] = []
    for name in ("semantic_key", "predicate", "action_kind", "grounded_descriptor"):
        value = str(metadata.get(name) or "").strip()
        if value and value not in parts:
            parts.append(value)
    for participant in metadata.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        values = [
            str(participant.get(name) or "").strip()
            for name in ("role", "entity_type", "surface", "visual_signature")
        ]
        description = " ".join(value for value in values if value)
        if description:
            parts.append(description)
    for state in metadata.get("states") or []:
        if not isinstance(state, dict):
            continue
        attribute = str(state.get("attribute") or "").strip()
        value = str(state.get("value") or "").strip()
        if attribute or value:
            parts.append(" ".join(item for item in (attribute, value) if item))
    state_change = metadata.get("state_change")
    if isinstance(state_change, dict) and state_change:
        compact = " ".join(
            str(state_change.get(name) or "").strip()
            for name in ("attribute", "before", "after")
            if state_change.get(name) is not None
        )
        if compact:
            parts.append(compact)
    if not parts:
        parts.append(str(node.text or node.node_type).strip())
    return " | ".join(dict.fromkeys(part for part in parts if part))[:480]


def _evidence_value(node: MemoryNode) -> str | None:
    if node.text:
        return node.text
    for name in ("grounded_descriptor", "predicate", "caption", "transcript"):
        if node.metadata.get(name):
            return str(node.metadata[name])
    return None
