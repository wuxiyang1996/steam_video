"""Layer-1 clue-memory graph extraction and validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..schemas import RuntimeMode, VideoRegime

HIDDEN_SUPERVISION_NODE_SOURCE_TYPES = {
    "segment_description",
    "inference_shot",
    "key_relationship",
    "clue_interval",
    "clue_clip",
    "reasoning_process_step",
    "video_summary",
    "qa_answer",
}


def _visible_under_observation_end(node: dict[str, Any], observation_end_s: float | None) -> bool:
    if observation_end_s is None:
        return True
    span = node.get("time_span")
    if not span:
        return True
    return float(span.get("end_s", 0.0)) <= observation_end_s + 1e-6


def _is_hidden_supervision_node(node: dict[str, Any], mode: str) -> bool:
    if mode != RuntimeMode.VIDEO_ONLY.value:
        return False
    if node.get("discovery_status") == "provided_supervision":
        return True
    return node.get("source_type") in HIDDEN_SUPERVISION_NODE_SOURCE_TYPES


def extract_clue_memory_graph(
    example: dict[str, Any],
    *,
    mode: RuntimeMode | str | None = None,
    include_hidden_supervision: bool | None = None,
) -> dict[str, Any]:
    """Build a question-agnostic layer-1 clue-memory graph from a canonical example."""
    runtime_mode = _resolve_mode(example, mode)
    if include_hidden_supervision is None:
        include_hidden_supervision = runtime_mode == RuntimeMode.EXPERT_DEMO.value

    video = example.get("video") or {}
    index = example.get("evidence_index") or {}
    clip_policy = deepcopy(index.get("clip_policy") or {})
    metadata = example.get("metadata") or {}
    video_regime = metadata.get("video_regime") or clip_policy.get("video_regime") or "short"
    observation_end_s = clip_policy.get("observation_end_s")

    nodes: list[dict[str, Any]] = []
    for node in index.get("nodes") or []:
        normalized = _normalize_clue_node(node, example, runtime_mode)
        if not include_hidden_supervision and _is_hidden_supervision_node(normalized, runtime_mode):
            continue
        if clip_policy.get("online") and not _visible_under_observation_end(normalized, observation_end_s):
            continue
        nodes.append(normalized)

    valid_ids = {n["node_id"] for n in nodes}
    edges: list[dict[str, Any]] = []
    for edge in index.get("edges") or []:
        if edge.get("src") not in valid_ids or edge.get("dst") not in valid_ids:
            continue
        edges.append(deepcopy(edge))

    # Attach visible subtitle / segment observations as layer-1 nodes when present.
    if include_hidden_supervision:
        for seg in video.get("segments") or []:
            node = _segment_to_clue_node(seg, example, runtime_mode)
            if node and node["node_id"] not in valid_ids:
                if clip_policy.get("online") and not _visible_under_observation_end(node, observation_end_s):
                    continue
                nodes.append(node)
                valid_ids.add(node["node_id"])

    stats = _index_stats(nodes, metadata)
    graph = {
        "schema_version": example.get("schema_version", "video-skills-relaunch/v0.1"),
        "graph_id": f"clue_memory:{example.get('example_id')}",
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "video_id": video.get("video_id"),
        "video_regime": video_regime,
        "input_mode": runtime_mode,
        "layer": "clue_memory",
        "clip_policy": clip_policy,
        "retrieval": deepcopy(metadata.get("retrieval") or index.get("retrieval") or {}),
        "observation_end_s": observation_end_s,
        "index_stats": stats,
        "perception": _perception_metadata(example, metadata),
        "nodes": nodes,
        "edges": edges,
        "trust_policy": deepcopy(example.get("trust_policy") or {}),
        "metadata": {
            "task_family": example.get("task_family"),
            "wrapper_version": metadata.get("wrapper_version"),
            "index_id": index.get("index_id"),
        },
    }
    return graph


def make_reasoning_rollout_shell(
    example: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    *,
    rollout_source: str = "pending",
) -> dict[str, Any]:
    """Layer-2 shell linked to a layer-1 clue-memory graph."""
    mode = clue_memory_graph.get("input_mode") or (example.get("available_inputs") or {}).get("mode", "expert_demo")
    return {
        "schema_version": example.get("schema_version", "video-skills-relaunch/v0.1"),
        "rollout_id": f"skill_rollout:{example.get('example_id')}:pending",
        "example_id": example.get("example_id"),
        "rollout_source": rollout_source,
        "input_mode": mode,
        "layer": "reasoning",
        "video_regime": clue_memory_graph.get("video_regime"),
        "clue_memory_ref": {
            "graph_id": clue_memory_graph.get("graph_id"),
            "index_id": clue_memory_graph.get("metadata", {}).get("index_id"),
            "observation_end_s": clue_memory_graph.get("observation_end_s"),
        },
        "question": deepcopy(example.get("question") or {}),
        "retrieval_budget": {
            "topk_coarse": (clue_memory_graph.get("retrieval") or {}).get("topk", 2),
            "max_retrieval_steps": 5,
        },
        "used_motifs": [],
        "nodes": [],
        "edges": [],
        "claims": [],
        "answer_support_chain": [],
        "final_answer": {"label": None, "text": None, "confidence": None},
        "verifier_summary": {
            "schema_valid": False,
            "all_commits_have_evidence": False,
            "answer_chain_valid": False,
            "timestamp_valid": False,
            "no_old_video_fact_leakage": True,
            "no_hidden_supervision_leakage": mode == "video_only",
        },
        "acceptance_status": "rejected",
        "failure_reasons": ["rollout_not_built"],
    }


def _resolve_mode(example: dict[str, Any], mode: RuntimeMode | str | None) -> str:
    if isinstance(mode, RuntimeMode):
        return mode.value
    if mode:
        return str(mode)
    return str((example.get("available_inputs") or {}).get("mode") or RuntimeMode.EXPERT_DEMO.value)


def _normalize_clue_node(node: dict[str, Any], example: dict[str, Any], runtime_mode: str) -> dict[str, Any]:
    normalized = deepcopy(node)
    normalized.setdefault("video_id", (example.get("video") or {}).get("video_id"))
    normalized.setdefault("provenance", {"created_by": "dataset_clip_wrapper.evidence_index"})
    normalized.setdefault(
        "visibility",
        {
            "mode": runtime_mode,
            "visible_to_agent": True,
            "hidden_supervision": _is_hidden_supervision_node(normalized, runtime_mode),
        },
    )
    return {k: v for k, v in normalized.items() if v is not None}


def _segment_to_clue_node(seg: dict[str, Any], example: dict[str, Any], runtime_mode: str) -> dict[str, Any] | None:
    if not seg.get("segment_id"):
        return None
    source_type = seg.get("source_type") or "segment_description"
    node_type = "event" if source_type in {"inference_shot", "reasoning_process_step"} else "observation"
    return {
        "node_id": seg["segment_id"],
        "node_type": node_type,
        "video_id": (example.get("video") or {}).get("video_id"),
        "time_span": seg.get("time_span"),
        "text": seg.get("text"),
        "source_type": source_type,
        "trust_level": "gold",
        "discovery_status": "provided_supervision",
        "source_ids": [(example.get("video") or {}).get("video_id")],
        "provenance": seg.get("provenance") or {},
        "visibility": {
            "mode": runtime_mode,
            "visible_to_agent": runtime_mode == RuntimeMode.EXPERT_DEMO.value,
            "hidden_supervision": True,
        },
    }


def _index_stats(nodes: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "index_clip_count": metadata.get("index_clip_count") or metadata.get("clip_count"),
        "coarse_clip_count": metadata.get("coarse_clip_count"),
        "fine_clip_count": metadata.get("fine_clip_count"),
        "perception_clip_count": (metadata.get("perception") or {}).get("perception_clip_count"),
        "observation_count": sum(1 for n in nodes if n.get("node_type") in {"observation", "caption_span", "subtitle_span", "dialogue_span"}),
        "event_count": sum(1 for n in nodes if n.get("node_type") == "event"),
        "clip_count": sum(1 for n in nodes if n.get("node_type") == "clip"),
    }


def _perception_metadata(example: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    index = example.get("evidence_index") or {}
    has_model_nodes = any(
        (n.get("provenance") or {}).get("created_by", "").endswith("graph_composer")
        for n in index.get("nodes") or []
    )
    build_phase = "index_only"
    if metadata.get("clip_schemas") or metadata.get("clip_schema_model"):
        build_phase = "perception_partial"
    if has_model_nodes:
        build_phase = "perception_partial" if metadata.get("clip_schema_max_clips") else "perception_full"
    return {
        "clip_schema_model": metadata.get("clip_schema_model"),
        "graph_composer_model": (metadata.get("graph_compose") or {}).get("composer_model"),
        "backbone": (index.get("backbone") or {}).get("name"),
        "build_phase": build_phase,
    }
