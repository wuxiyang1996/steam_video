"""Adapters from Video_Skills canonical/L1 records to memory nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .types import MemoryNode, TimeSpan


PREFERRED_EVENT_NODE_TYPES = ("event", "observation", "state", "dialogue_span")
FALLBACK_NODE_TYPES = ("clip",)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def extract_l1_graph(canonical: dict[str, Any]) -> dict[str, Any]:
    """Read a materialized L1 graph or derive its minimal shell from evidence_index."""
    metadata = canonical.get("metadata") or {}
    materialized = metadata.get("clue_memory_graph")
    if isinstance(materialized, dict):
        return materialized

    evidence_index = canonical.get("evidence_index")
    if not isinstance(evidence_index, dict):
        raise ValueError("canonical example has neither metadata.clue_memory_graph nor evidence_index")

    video = canonical.get("video") or {}
    clip_policy = evidence_index.get("clip_policy") or {}
    return {
        "schema_version": canonical.get("schema_version"),
        "graph_id": f"clue_memory:{canonical.get('example_id')}",
        "example_id": canonical.get("example_id"),
        "dataset": canonical.get("dataset"),
        "video_id": video.get("video_id"),
        "video_regime": metadata.get("video_regime", "short"),
        "input_mode": (canonical.get("available_inputs") or {}).get("mode", "video_only"),
        "layer": "clue_memory",
        "clip_policy": clip_policy,
        "observation_end_s": clip_policy.get("observation_end_s"),
        "nodes": evidence_index.get("nodes") or [],
        "edges": evidence_index.get("edges") or [],
        "metadata": {
            "derived_by": "memory_graph.adapter.extract_l1_graph",
            "index_id": evidence_index.get("index_id"),
        },
    }


def canonical_to_memory_nodes(
    canonical: dict[str, Any],
    *,
    include_node_types: Iterable[str] | None = None,
    require_materialized_l1: bool = False,
) -> tuple[dict[str, Any], list[MemoryNode]]:
    """Convert one Video_Skills canonical example into memory nodes.

    By default, semantic event/observation nodes are preferred. Clip nodes are
    used only when the L1 graph has no semantic event nodes, preventing obvious
    duplicate memory entries for a clip and its derived observation.
    """
    materialized = (canonical.get("metadata") or {}).get("clue_memory_graph")
    if require_materialized_l1 and not isinstance(materialized, dict):
        raise ValueError(
            "trusted video-only memory requires metadata.clue_memory_graph"
        )
    graph = extract_l1_graph(canonical)
    video = canonical.get("video") or {}
    video_id = str(graph.get("video_id") or video.get("video_id") or "")
    if not video_id:
        raise ValueError("video_id is required")

    raw_nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    if include_node_types is None:
        preferred = [node for node in raw_nodes if node.get("node_type") in PREFERRED_EVENT_NODE_TYPES]
        selected = preferred or [node for node in raw_nodes if node.get("node_type") in FALLBACK_NODE_TYPES]
    else:
        allowed = set(include_node_types)
        selected = [node for node in raw_nodes if node.get("node_type") in allowed]

    text_by_clip = _clip_text_index(canonical)
    incident_edges = _incident_edge_index(graph)
    memory_nodes: list[MemoryNode] = []
    for source in selected:
        span_payload = source.get("time_span")
        if not isinstance(span_payload, dict):
            continue
        source_id = str(source.get("node_id") or "")
        if not source_id:
            continue

        provenance = dict(source.get("provenance") or {})
        provenance.update(
            {
                "source_graph_id": graph.get("graph_id"),
                "source_schema_version": graph.get("schema_version"),
                "source_node_type": source.get("node_type"),
                "adapter": "memory_graph.adapter.canonical_to_memory_nodes",
            }
        )
        source_segments = [str(value) for value in source.get("source_ids") or []]
        for key in ("clip_id", "parent_clip_id"):
            if source.get(key):
                source_segments.append(str(source[key]))
        if source_id not in source_segments:
            source_segments.append(source_id)
        source_segments = list(dict.fromkeys(source_segments))

        memory_nodes.append(
            MemoryNode(
                node_id=f"memory:{source_id}",
                video_id=video_id,
                node_type="observation",
                time_span=TimeSpan.from_dict(span_payload),
                text=_node_text(source, text_by_clip),
                provenance=provenance,
                source_node_id=source_id,
                source_segments=source_segments,
                metadata={
                    "source_node_type": source.get("node_type"),
                    "source_type": source.get("source_type"),
                    "trust_level": source.get("trust_level"),
                    "discovery_status": source.get("discovery_status"),
                    "granularity": source.get("granularity"),
                    "visibility": source.get("visibility"),
                    "confidence": source.get("confidence"),
                    "producer": source.get("producer"),
                    "modality": source.get("modality"),
                    "clip_id": source.get("clip_id"),
                    "parent_clip_id": source.get("parent_clip_id"),
                    "local_node_id": source.get("local_node_id"),
                    "local_id_clip_id": source.get("local_id_clip_id"),
                    "surface_form": source.get("surface_form"),
                    "observable_facts": source.get("observable_facts"),
                    "events": source.get("events"),
                    "entities": source.get("entities"),
                    "dialogue": source.get("dialogue"),
                    "ocr_text": source.get("ocr_text"),
                    "source_l1_edges": incident_edges.get(source_id, []),
                },
            )
        )

    memory_nodes.sort(key=lambda node: (node.time_span.start_s, node.time_span.end_s, node.node_id))
    return graph, memory_nodes


def _clip_text_index(canonical: dict[str, Any]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for evidence in canonical.get("evidence_candidates") or []:
        if not isinstance(evidence, dict) or not evidence.get("text"):
            continue
        media_ref = evidence.get("media_ref") or {}
        clip_id = media_ref.get("clip_id")
        if clip_id:
            texts[str(clip_id)] = str(evidence["text"])

    metadata = canonical.get("metadata") or {}
    for key in ("clip_schemas", "coarse_clip_schemas"):
        for clip in metadata.get(key) or []:
            if not isinstance(clip, dict):
                continue
            clip_id = clip.get("clip_id")
            if not clip_id:
                continue
            parts = [
                clip.get("scene_description"),
                clip.get("summary"),
                clip.get("text"),
            ]
            text = " ".join(str(part).strip() for part in parts if part)
            if text:
                texts[str(clip_id)] = text
    return texts


def _node_text(node: dict[str, Any], text_by_clip: dict[str, str]) -> str:
    if node.get("text"):
        return str(node["text"]).strip()
    node_id = str(node.get("node_id") or "")
    if node_id in text_by_clip:
        return text_by_clip[node_id]
    parent_clip_id = node.get("parent_clip_id")
    if parent_clip_id and str(parent_clip_id) in text_by_clip:
        return text_by_clip[str(parent_clip_id)]
    span = node.get("time_span") or {}
    return (
        f"{node.get('node_type', 'event')} in video interval "
        f"[{float(span.get('start_s', 0.0)):.2f}, {float(span.get('end_s', 0.0)):.2f}] seconds"
    )


def _incident_edge_index(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        compact = {
            key: edge.get(key)
            for key in (
                "edge_id",
                "src",
                "dst",
                "edge_type",
                "evidence_refs",
                "producer",
            )
            if edge.get(key) is not None
        }
        for endpoint in (edge.get("src"), edge.get("dst")):
            if endpoint:
                result.setdefault(str(endpoint), []).append(compact)
    return result
