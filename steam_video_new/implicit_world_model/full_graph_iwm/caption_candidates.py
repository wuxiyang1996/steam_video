"""Question-independent categorical caption candidates for L1.5 navigation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

from memory_graph.multichannel_correlation import CandidateNavigationEdge
from memory_graph.types import MemoryNode

from .contracts import RetainedEvidenceGraph


CAPTION_CANDIDATE_SCHEMA = "steam-l1.5-caption-candidate-overlay/v0.1"
ALLOWED_CATEGORIES = frozenset(
    {
        "shared_context_recurrence",
        "event_continuation",
        "possible_state_change",
        "explicit_contrast",
    }
)
CATEGORY_CHANNEL = {
    "shared_context_recurrence": "entity_candidate",
    "event_continuation": "caption_bridge",
    "possible_state_change": "change_candidate",
    "explicit_contrast": "contrast_candidate",
}


class CategoricalJSONClient(Protocol):
    model: str

    def complete_json(self, *, task: str, payload: dict[str, Any]) -> dict[str, Any]: ...


def propose_caption_candidate_overlay(
    graph: RetainedEvidenceGraph,
    client: CategoricalJSONClient,
) -> dict[str, Any]:
    """Ask one categorical model call for sparse, grounded pair proposals.

    The model sees no question, answer, clue interval, or numeric utility and
    cannot mutate L1.  Returned candidates are navigation hypotheses, not facts.
    """

    nodes = tuple(sorted(graph.nodes, key=lambda node: (node.time_span.start_s, node.node_id)))
    node_payload = {node.node_id: _node_descriptor(node) for node in nodes}
    payload = {
        "nodes": node_payload,
        "allowed_categories": sorted(ALLOWED_CATEGORIES),
        "allowed_directions": ["bidirectional", "src_to_dst", "dst_to_src"],
        "category_contract": {
            "shared_context_recurrence": (
                "a distinctive entity, object, setting, or event pattern visibly recurs"
            ),
            "event_continuation": (
                "the destination explicitly continues or completes information in the source"
            ),
            "possible_state_change": (
                "the same described entity and attribute have explicit before/after values"
            ),
            "explicit_contrast": (
                "the descriptions contain directly incompatible or negated observations"
            ),
        },
        "required_output": {
            "only_key": "pairs",
            "pair_fields": [
                "src",
                "dst",
                "category",
                "direction",
                "src_evidence",
                "dst_evidence",
            ],
            "evidence_contract": (
                "src_evidence and dst_evidence must each copy a non-empty exact "
                "substring from that endpoint's grounding_text"
            ),
            "omit_unrelated_or_inconclusive_pairs": True,
        },
        "forbidden_claims": [
            "verified identity",
            "verified state transition",
            "causality",
            "question relevance",
            "planner preference",
            "numeric score or probability",
        ],
        "question_independent": True,
    }
    result = client.complete_json(
        task=(
            "Build a sparse question-independent L1.5 candidate overlay from the "
            "grounded node descriptors. Emit a pair only when both endpoint texts "
            "directly support one allowed category. Copy exact endpoint phrases from "
            "grounding_text into src_evidence and dst_evidence. Prefer omission over speculation. "
            "Do not choose a reasoning action and do not infer causality."
        ),
        payload=payload,
    )
    if set(result) != {"pairs"} or not isinstance(result.get("pairs"), list):
        raise ValueError("caption candidate output requires the sole key pairs")
    node_by_id = {node.node_id: node for node in nodes}
    existing_pairs = {
        frozenset((edge.src, edge.dst))
        for edge in (*graph.temporal_edges, *graph.correlation_edges)
    }
    edges: list[CandidateNavigationEdge] = []
    seen: set[tuple[str, str, str]] = set()
    proposed_count = 0
    excluded_existing_topology_count = 0
    for row in result["pairs"]:
        if not isinstance(row, Mapping):
            raise ValueError("caption candidate pair must be an object")
        if set(row) != {
            "src",
            "dst",
            "category",
            "direction",
            "src_evidence",
            "dst_evidence",
        }:
            raise ValueError("caption candidate pair fields do not match schema")
        src = str(row["src"])
        dst = str(row["dst"])
        category = str(row["category"])
        direction = str(row["direction"])
        if src not in node_by_id or dst not in node_by_id or src == dst:
            raise ValueError("caption candidate references an unknown/identical endpoint")
        proposed_count += 1
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"unsupported caption candidate category: {category}")
        if direction not in {"bidirectional", "src_to_dst", "dst_to_src"}:
            raise ValueError("caption candidate direction is invalid")
        src_evidence = str(row["src_evidence"]).strip()
        dst_evidence = str(row["dst_evidence"]).strip()
        if not src_evidence or not dst_evidence:
            raise ValueError("caption candidate requires evidence at both endpoints")
        if not _grounded_substring(src_evidence, node_payload[src]["grounding_text"]):
            raise ValueError("caption candidate src evidence is not grounded text")
        if not _grounded_substring(dst_evidence, node_payload[dst]["grounding_text"]):
            raise ValueError("caption candidate dst evidence is not grounded text")
        if category == "possible_state_change":
            if direction != "src_to_dst":
                raise ValueError("possible state change must follow src_to_dst time")
            if node_by_id[src].time_span.start_s > node_by_id[dst].time_span.start_s:
                raise ValueError("possible state change endpoints are not time ordered")
        if frozenset((src, dst)) in existing_pairs:
            excluded_existing_topology_count += 1
            continue
        key = (src, dst, category)
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256("\x1f".join(key).encode("utf-8")).hexdigest()[:20]
        edges.append(
            CandidateNavigationEdge(
                edge_id=f"caption-edge:{digest}",
                src=src,
                dst=dst,
                channel=CATEGORY_CHANNEL[category],
                category=category,
                direction=direction,
                evidence_refs=(src, dst),
                source=str(client.model),
                provenance={
                    "src_evidence": src_evidence,
                    "dst_evidence": dst_evidence,
                    "question_independent": True,
                    "model_output_is_categorical": True,
                    "verified_relation": False,
                },
            )
        )
    possible = len(nodes) * (len(nodes) - 1) // 2
    return {
        "schema_version": CAPTION_CANDIDATE_SCHEMA,
        "graph_id": graph.graph_id,
        "base_graph_fingerprint": _base_graph_fingerprint(graph),
        "source_l1_fingerprint": graph.metadata.get("source_l1_fingerprint"),
        "model": str(client.model),
        "question_independent": True,
        "contains_question_answer_or_clue": False,
        "numeric_score_present": False,
        "verified_relation_count": 0,
        "candidate_edge_count": len(edges),
        "model_proposed_pair_count": proposed_count,
        "excluded_existing_topology_pair_count": excluded_existing_topology_count,
        "possible_pair_count": possible,
        "candidate_edge_density": len(edges) / possible if possible else 0.0,
        "edges": [edge.to_dict() for edge in edges],
        "training_performed": False,
    }


def augment_graph_with_caption_candidates(
    graph: RetainedEvidenceGraph,
    artifact: Mapping[str, Any],
) -> RetainedEvidenceGraph:
    if artifact.get("schema_version") != CAPTION_CANDIDATE_SCHEMA:
        raise ValueError("caption candidate artifact schema mismatch")
    if artifact.get("question_independent") is not True:
        raise ValueError("caption candidate artifact must be question independent")
    if artifact.get("contains_question_answer_or_clue") is not False:
        raise ValueError("caption candidate artifact contains forbidden supervision")
    if artifact.get("numeric_score_present") is not False:
        raise ValueError("caption candidate artifact may not contain numeric scores")
    if artifact.get("base_graph_fingerprint") != _base_graph_fingerprint(graph):
        raise ValueError("caption candidate artifact does not match frozen base graph")
    edges = tuple(
        CandidateNavigationEdge.from_dict(row) for row in artifact.get("edges") or []
    )
    metadata = dict(graph.metadata)
    metadata["caption_candidate_overlay"] = {
        "schema_version": artifact.get("schema_version"),
        "model": artifact.get("model"),
        "candidate_edge_count": len(edges),
        "candidate_edge_density": artifact.get("candidate_edge_density"),
        "question_independent": True,
        "numeric_score_present": False,
        "verified_relation_count": 0,
    }
    return replace(graph, candidate_edges=edges, metadata=metadata)


def _node_descriptor(node: MemoryNode) -> dict[str, Any]:
    metadata = node.metadata
    participants = []
    for row in metadata.get("participants") or []:
        if not isinstance(row, Mapping):
            continue
        participants.append(
            {
                "role": row.get("role"),
                "entity_type": row.get("entity_type"),
                "surface": row.get("surface"),
                "visual_signature": row.get("visual_signature"),
                "track_status": row.get("track_status"),
            }
        )
    descriptor = {
        "text": node.text,
        "predicate": metadata.get("predicate"),
        "action_kind": metadata.get("action_kind"),
        "participants": participants,
        "states": metadata.get("states") or [],
        "state_change": metadata.get("state_change"),
        "source_type": metadata.get("source_type"),
    }
    descriptor["grounding_text"] = " | ".join(
        _flatten_grounding_text(descriptor)
    )[:2000]
    return descriptor


def _flatten_grounding_text(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, child in value.items():
            if key == "grounding_text":
                continue
            result.extend(_flatten_grounding_text(child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for child in value:
            result.extend(_flatten_grounding_text(child))
        return result
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _grounded_substring(evidence: str, grounding_text: str) -> bool:
    normalized = " ".join(evidence.casefold().split())
    source = " ".join(grounding_text.casefold().split())
    return len(normalized) >= 3 and normalized in source


def _base_graph_fingerprint(graph: RetainedEvidenceGraph) -> str:
    payload = {
        "graph_id": graph.graph_id,
        "nodes": [node.to_dict() for node in graph.nodes],
        "temporal_edges": [edge.__dict__ for edge in graph.temporal_edges],
        "correlation_edges": [edge.to_dict() for edge in graph.correlation_edges],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
