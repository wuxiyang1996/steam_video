"""Bridge canonical wrapped examples into atomic-skill evidence graphs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from atomic_skills.common import ensure_graph, stable_id

from ..schemas import RuntimeMode


HIDDEN_SUPERVISION_SOURCE_TYPES = {
    "segment_description",
    "inference_shot",
    "key_relationship",
    "clue_interval",
    "clue_clip",
    "reasoning_process_step",
    "video_summary",
    "qa_answer",
}

DEFAULT_TRUST_POLICY = {
    "gold_sources": [
        "segment_description",
        "inference_shot",
        "key_relationship",
        "clue_interval",
        "clue_clip",
        "reasoning_process_step",
    ],
    "strong_sources": ["video_summary", "subtitle_span"],
    "weak_sources": [],
    "model_labeled_sources": ["caption_span", "model_labeled_span"],
}


def canonical_example_to_skill_graph(
    example: dict[str, Any],
    *,
    mode: RuntimeMode | str | None = None,
    include_question_context: bool = True,
) -> dict[str, Any]:
    """Convert a CanonicalVideoExample row into an atomic-skill evidence graph.

    The wrapper schema is the durable dataset format. Atomic skills operate on a
    smaller graph runtime shape: ``nodes``, ``edges``, grounding ids, trust, and
    visibility metadata. This bridge keeps that transformation explicit.
    """

    runtime_mode = _resolve_mode(example, mode)
    graph = ensure_graph(
        {
            "schema_version": example.get("schema_version"),
            "graph_id": f"skill_graph:{example.get('example_id')}",
            "example_id": example.get("example_id"),
            "dataset": example.get("dataset"),
            "task_family": example.get("task_family"),
            "mode": runtime_mode,
            "nodes": [],
            "edges": [],
            "trust_policy": _resolved_trust_policy(example),
            "metadata": {
                "created_by": "dataset_clip_wrapper.l1_clue_graph.skill_graph_bridge",
                "source_schema_version": example.get("schema_version"),
            },
        }
    )

    index = example.get("evidence_index") or {}
    for node in index.get("nodes") or []:
        graph["nodes"].append(_normalize_index_node(node, example))
    for edge in index.get("edges") or []:
        graph["edges"].append(deepcopy(edge))

    candidate_node_ids: list[str] = []
    for evidence in example.get("evidence_candidates") or []:
        if _is_hidden_in_mode(evidence, runtime_mode):
            continue
        node = _evidence_candidate_to_node(evidence, example)
        if node is None:
            _attach_candidate_to_clip(graph, evidence, runtime_mode)
            continue
        graph["nodes"].append(node)
        candidate_node_ids.append(node["node_id"])
        edge = _candidate_source_edge(node, graph)
        if edge:
            graph["edges"].append(edge)

    if include_question_context:
        graph["question"] = deepcopy(example.get("question") or {})
        graph["question_context"] = {
            "question_text": (example.get("question") or {}).get("question_text", ""),
            "options": (example.get("question") or {}).get("options", []),
            "answer_format": (example.get("question") or {}).get("answer_format", "unknown"),
        }

    graph["metadata"]["candidate_node_count"] = len(candidate_node_ids)
    graph["metadata"]["clip_node_count"] = sum(1 for node in graph["nodes"] if node.get("node_type") == "clip")
    graph["metadata"]["hidden_supervision_sources"] = list((example.get("hidden_supervision") or {}).get("sources") or [])
    _dedupe_graph_in_place(graph)
    return graph


def _resolve_mode(example: dict[str, Any], mode: RuntimeMode | str | None) -> str:
    if isinstance(mode, RuntimeMode):
        return mode.value
    if mode:
        return str(mode)
    return str((example.get("available_inputs") or {}).get("mode") or RuntimeMode.EXPERT_DEMO.value)


def _resolved_trust_policy(example: dict[str, Any]) -> dict[str, list[str]]:
    policy = deepcopy(DEFAULT_TRUST_POLICY)
    existing = example.get("trust_policy") or {}
    for key in ("gold_sources", "strong_sources", "weak_sources", "model_labeled_sources"):
        merged = [*policy.get(key, []), *(existing.get(key) or [])]
        policy[key] = list(dict.fromkeys(merged))
    return policy


def _normalize_index_node(node: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(node)
    normalized.setdefault("source_ids", _source_ids_for_node(normalized, example))
    normalized.setdefault("provenance", {"created_by": "dataset_clip_wrapper.evidence_index"})
    normalized.setdefault("visibility", _visibility_payload(example, normalized.get("trust_level")))
    return _omit_none_values(normalized)


def _evidence_candidate_to_node(evidence: dict[str, Any], example: dict[str, Any]) -> dict[str, Any] | None:
    source_type = evidence.get("source_type")
    if source_type == "video_segment":
        return None
    node_type = _node_type_for_source(source_type)
    node = {
        "node_id": evidence.get("evidence_id"),
        "node_type": node_type,
        "evidence_id": evidence.get("evidence_id"),
        "source_type": source_type,
        "source_ids": _source_ids_for_evidence(evidence, example),
        "text": evidence.get("text"),
        "time_span": evidence.get("time_span"),
        "media_ref": evidence.get("media_ref"),
        "trust_level": evidence.get("trust_level", _trust_level_for_source(source_type, example)),
        "discovery_status": evidence.get("discovery_status") or _default_discovery_status(evidence),
        "confidence": evidence.get("confidence"),
        "entities": evidence.get("entities"),
        "claims": evidence.get("claims"),
        "evidence_role": evidence.get("evidence_role"),
        "provenance": {
            **(evidence.get("provenance") or {}),
            "created_by": "dataset_clip_wrapper.l1_clue_graph.skill_graph_bridge",
            "source_evidence_id": evidence.get("evidence_id"),
        },
    }
    if node_type == "event":
        node["event_description"] = evidence.get("text")
    if node_type == "dialogue_span":
        node["speaker"] = (evidence.get("provenance") or {}).get("speaker")
    node["visibility"] = _visibility_payload(example, node["trust_level"], node["discovery_status"])
    return _omit_none_values(node)


def _attach_candidate_to_clip(graph: dict[str, Any], evidence: dict[str, Any], runtime_mode: str) -> None:
    clip_id = (evidence.get("media_ref") or {}).get("clip_id")
    if not clip_id:
        return
    for node in graph.get("nodes", []):
        if node.get("node_id") != clip_id:
            continue
        refs = node.setdefault("evidence_candidate_refs", [])
        if evidence.get("evidence_id") and evidence["evidence_id"] not in refs:
            refs.append(evidence["evidence_id"])
        node.setdefault("trust_level", evidence.get("trust_level", "derived"))
        node.setdefault("discovery_status", evidence.get("discovery_status", "derived_runtime"))
        node.setdefault("visibility", {"mode": runtime_mode, "hidden_supervision": False})
        return


def _candidate_source_edge(node: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any] | None:
    node_ids = {item.get("node_id") for item in graph.get("nodes", [])}
    for source_id in node.get("source_ids") or []:
        if source_id in node_ids:
            return {
                "edge_id": stable_id("skill.edge", node["node_id"], source_id, "derived_from"),
                "src": node["node_id"],
                "dst": source_id,
                "edge_type": "derived_from",
            }
    return None


def _node_type_for_source(source_type: str | None) -> str:
    if source_type == "subtitle_span":
        return "dialogue_span"
    if source_type in {"caption_span", "model_labeled_span"}:
        return "observation"
    if source_type in {"reasoning_process_step", "inference_shot"}:
        return "event"
    return "observation"


def _source_ids_for_node(node: dict[str, Any], example: dict[str, Any]) -> list[str]:
    if node.get("source_ids"):
        return list(node["source_ids"])
    if node.get("node_type") == "clip":
        candidates = [node.get("video_id") or (example.get("video") or {}).get("video_id")]
    else:
        candidates = [(example.get("video") or {}).get("video_id")]
    return [item for item in candidates if item]


def _source_ids_for_evidence(evidence: dict[str, Any], example: dict[str, Any]) -> list[str]:
    media_ref = evidence.get("media_ref") or {}
    candidates = [
        media_ref.get("clip_id"),
        media_ref.get("video_id"),
        (example.get("video") or {}).get("video_id"),
    ]
    return [item for item in candidates if item]


def _trust_level_for_source(source_type: str | None, example: dict[str, Any]) -> str:
    policy = _resolved_trust_policy(example)
    for level in ("gold", "strong", "weak", "model_labeled"):
        if source_type in policy.get(f"{level}_sources", []):
            return level
    return "derived" if source_type == "video_segment" else "weak"


def _default_discovery_status(evidence: dict[str, Any]) -> str:
    trust = evidence.get("trust_level")
    if trust in {"gold", "strong"}:
        return "provided_supervision"
    if trust == "derived":
        return "derived_runtime"
    if trust == "model_labeled":
        return "discovered_runtime"
    return "provided_visible_context"


def _visibility_payload(
    example: dict[str, Any],
    trust_level: str | None,
    discovery_status: str | None = None,
) -> dict[str, Any]:
    mode = (example.get("available_inputs") or {}).get("mode", RuntimeMode.EXPERT_DEMO.value)
    return {
        "mode": mode,
        "visible_to_agent": True,
        "hidden_supervision": discovery_status == "provided_supervision" and mode == RuntimeMode.VIDEO_ONLY.value,
        "trust_level": trust_level,
        "discovery_status": discovery_status,
    }


def _is_hidden_in_mode(evidence: dict[str, Any], runtime_mode: str) -> bool:
    return runtime_mode == RuntimeMode.VIDEO_ONLY.value and evidence.get("source_type") in HIDDEN_SUPERVISION_SOURCE_TYPES


def _dedupe_graph_in_place(graph: dict[str, Any]) -> None:
    seen_nodes: set[str] = set()
    nodes = []
    for node in graph.get("nodes", []):
        node_id = node.get("node_id")
        if not node_id or node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        nodes.append(node)
    graph["nodes"] = nodes

    valid_node_ids = {node["node_id"] for node in nodes}
    seen_edges: set[str] = set()
    edges = []
    for edge in graph.get("edges", []):
        edge_id = edge.get("edge_id")
        if not edge_id or edge_id in seen_edges:
            continue
        if edge.get("src") not in valid_node_ids or edge.get("dst") not in valid_node_ids:
            continue
        seen_edges.add(edge_id)
        edges.append(edge)
    graph["edges"] = edges


def _omit_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
