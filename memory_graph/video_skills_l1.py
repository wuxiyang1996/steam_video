"""Acceptance and passthrough contracts for Video_Skills clue-memory L1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import AtomicEvent, EntityMention, StateAssertion
from .types import MemoryNode


SEMANTIC_NODE_TYPES = {
    "observation",
    "event",
    "state",
    "dialogue_span",
    "clue",
    "entity_mention",
}
DIRECT_EVENT_NODE_TYPES = {
    "event",
    "dialogue_span",
}

# Aligned with Video_Skills VLM_L1_EDGE_TYPES + legacy clue-memory edges.
ALLOWED_L1_EDGE_TYPES = frozenset(
    {
        "temporal_next",
        "temporal_overlap",
        "temporal_inside",
        "entity_mention",
        "state_of",
        "derived_from",
        "same_entity",
        "same_object",
        "same_place",
        "reappears",
        "before_after",
        "state_change",
        "supports_observation",
        "contrasts_observation",
        "located_in",
        "causal_hint",
        "social_cue",
        "dialogue_speaker",
        "face_voice_equivalence",
        "supports",
        "retrieval_score",
        "requires_evidence",
        "limits_answerability",
        "weak_visual_context",
        "partially_supports_requirement",
        "repair_reminder_for",
    }
)


@dataclass(frozen=True)
class VideoSkillsL1QualityReport:
    status: str
    grade: str
    metrics: dict[str, Any]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "grade": self.grade,
            "metrics": self.metrics,
            "issues": list(self.issues),
            "scope": (
                "Video_Skills intrinsic structural/perception acceptance; "
                "not independent semantic correctness"
            ),
        }


def audit_video_skills_l1(canonical: dict[str, Any]) -> VideoSkillsL1QualityReport:
    """Reproduce and tighten the Video_Skills intrinsic high-grade L1 gate."""

    metadata = canonical.get("metadata")
    graph = metadata.get("clue_memory_graph") if isinstance(metadata, dict) else None
    if not isinstance(graph, dict):
        return VideoSkillsL1QualityReport(
            status="fail",
            grade="missing",
            metrics={},
            issues=("metadata.clue_memory_graph is not materialized",),
        )

    nodes = [value for value in graph.get("nodes") or [] if isinstance(value, dict)]
    edges = [value for value in graph.get("edges") or [] if isinstance(value, dict)]
    node_ids = [str(node.get("node_id")) for node in nodes if node.get("node_id")]
    edge_ids = [str(edge.get("edge_id")) for edge in edges if edge.get("edge_id")]
    missing_node_ids = [node for node in nodes if not node.get("node_id")]
    missing_edge_ids = [edge for edge in edges if not edge.get("edge_id")]
    node_id_set = set(node_ids)

    hidden = [node for node in nodes if _is_hidden(node)]
    semantic = [
        node
        for node in nodes
        if str(node.get("node_type") or "") in SEMANTIC_NODE_TYPES
        and _node_text(node)
        and not _is_hidden(node)
    ]
    semantic_ids = {str(node.get("node_id") or "") for node in semantic}
    invalid_edges = [
        edge
        for edge in edges
        if str(edge.get("src") or "") not in node_id_set
        or str(edge.get("dst") or "") not in node_id_set
    ]
    self_edges = [
        edge
        for edge in edges
        if str(edge.get("src") or "")
        and str(edge.get("src") or "") == str(edge.get("dst") or "")
    ]
    unknown_edge_types = sorted(
        {
            str(edge.get("edge_type") or "")
            for edge in edges
            if str(edge.get("edge_type") or "") not in ALLOWED_L1_EDGE_TYPES
        }
    )
    invalid_timestamps = [
        node
        for node in nodes
        if not _timestamp_valid(node.get("time_span"))
        and isinstance(node.get("time_span"), dict)
    ]
    invalid_confidences = [
        value
        for value in list(nodes) + list(edges)
        if not _confidence_valid(value.get("confidence"))
    ]
    duplicate_node_ids = _duplicate_values(node_ids)
    duplicate_edge_ids = _duplicate_values(edge_ids)
    duplicate_local_ids = _duplicate_local_node_ids(nodes)

    semantic_edges = [
        edge
        for edge in edges
        if str(edge.get("src") or "") in semantic_ids
        or str(edge.get("dst") or "") in semantic_ids
    ]
    clip_schemas = [
        value
        for value in metadata.get("clip_schemas") or []
        if isinstance(value, dict)
    ]
    successful_schemas = [
        value for value in clip_schemas if value.get("clip_id") and not value.get("model_error")
    ]
    schema_errors = [value for value in clip_schemas if value.get("model_error")]
    semantic_clip_ids = {
        str(node.get("clip_id"))
        for node in semantic
        if node.get("clip_id")
    }
    successful_clip_ids = {
        str(schema.get("clip_id"))
        for schema in successful_schemas
    }
    coverage = (
        len(semantic_clip_ids & successful_clip_ids) / len(successful_clip_ids)
        if successful_clip_ids
        else None
    )
    graph_compose = metadata.get("graph_compose") or {}
    trace = graph_compose.get("execution_trace") or []
    failed_steps = [
        step
        for step in trace
        if isinstance(step, dict) and step.get("ok") is False
    ]
    fallback = bool(graph_compose.get("used_deterministic_fallback"))
    enough_edges = len(semantic_edges) >= max(1, len(semantic) // 4)
    schema_errors_detail = _validate_clue_memory_schema(graph)
    if (
        semantic
        and not invalid_edges
        and not self_edges
        and not duplicate_node_ids
        and not duplicate_edge_ids
        and not missing_node_ids
        and not missing_edge_ids
        and not duplicate_local_ids
        and not unknown_edge_types
        and not invalid_timestamps
        and not invalid_confidences
        and not schema_errors_detail
        and not failed_steps
        and (coverage is None or coverage >= 0.5)
        and enough_edges
    ):
        grade = "high"
    elif semantic and not invalid_edges and not self_edges:
        grade = "medium"
    else:
        grade = "low"

    issues: list[str] = []
    mode = str(graph.get("input_mode") or "")
    if mode != "video_only":
        issues.append("clue-memory graph input_mode is not video_only")
    if grade != "high":
        issues.append(f"Video_Skills intrinsic L1 grade is {grade!r}, not 'high'")
    if hidden:
        issues.append("clue-memory graph contains hidden-supervision nodes")
    if invalid_edges:
        issues.append("clue-memory graph contains edges with unknown endpoints")
    if self_edges:
        issues.append(f"clue-memory graph contains {len(self_edges)} self-edges")
    if duplicate_node_ids:
        issues.append(
            "clue-memory graph contains duplicate node_id values: "
            + ", ".join(duplicate_node_ids[:5])
        )
    if duplicate_edge_ids:
        issues.append(
            "clue-memory graph contains duplicate edge_id values: "
            + ", ".join(duplicate_edge_ids[:5])
        )
    if missing_node_ids:
        issues.append(
            f"clue-memory graph contains {len(missing_node_ids)} nodes without node_id"
        )
    if missing_edge_ids:
        issues.append(
            f"clue-memory graph contains {len(missing_edge_ids)} edges without edge_id"
        )
    if duplicate_local_ids:
        issues.append(
            "clue-memory graph contains duplicate (clip_id, local_node_id) values: "
            + ", ".join(duplicate_local_ids[:5])
        )
    if unknown_edge_types:
        issues.append(
            "clue-memory graph contains unknown edge types: "
            + ", ".join(unknown_edge_types[:8])
        )
    if invalid_timestamps:
        issues.append(
            f"clue-memory graph contains {len(invalid_timestamps)} invalid timestamps"
        )
    if invalid_confidences:
        issues.append(
            f"clue-memory graph contains {len(invalid_confidences)} "
            "confidence values outside [0, 1]"
        )
    if schema_errors_detail:
        issues.append(
            "clue-memory graph failed schema validation: "
            + schema_errors_detail[0]
        )
    if failed_steps:
        issues.append(f"{len(failed_steps)} L1 compose steps failed")
    if fallback:
        issues.append("L1 graph compose used deterministic fallback")
    if not successful_schemas:
        issues.append("no successful Qwen clip schemas are present")
    if schema_errors:
        issues.append(f"{len(schema_errors)} clip schemas contain model errors")
    producer_count = sum(
        str(node.get("producer") or "")
        in {
            "neighbor_vlm_l1_graph_composer",
            "vlm_l1_graph_composer",
            "neighbor_vlm_l1_schema_anchor",
        }
        for node in semantic
    )
    if producer_count == 0:
        issues.append("no semantic nodes were produced by the Video_Skills L1 composer")

    return VideoSkillsL1QualityReport(
        status="pass" if not issues else "fail",
        grade=grade,
        metrics={
            "semantic_nodes": len(semantic),
            "semantic_edges": len(semantic_edges),
            "hidden_nodes": len(hidden),
            "invalid_edges": len(invalid_edges),
            "self_edges": len(self_edges),
            "duplicate_node_ids": len(duplicate_node_ids),
            "duplicate_edge_ids": len(duplicate_edge_ids),
            "missing_node_ids": len(missing_node_ids),
            "missing_edge_ids": len(missing_edge_ids),
            "duplicate_local_node_ids": len(duplicate_local_ids),
            "unknown_edge_types": unknown_edge_types,
            "invalid_timestamps": len(invalid_timestamps),
            "invalid_confidences": len(invalid_confidences),
            "schema_validation_errors": len(schema_errors_detail),
            "successful_clip_schemas": len(successful_schemas),
            "clip_schema_errors": len(schema_errors),
            "semantic_clip_coverage": coverage,
            "failed_compose_steps": len(failed_steps),
            "used_deterministic_fallback": fallback,
            "composer_semantic_nodes": producer_count,
        },
        issues=tuple(issues),
    )


@dataclass(frozen=True)
class VideoSkillsL1AtomicEventExtractor:
    """Promote accepted, already-semantic L1 nodes without regenerating L1."""

    model: str = "Video_Skills/ClueMemoryGraph"

    def extract(self, nodes: list[MemoryNode]) -> list[AtomicEvent]:
        events: list[AtomicEvent] = []
        for index, node in enumerate(
            sorted(nodes, key=lambda value: (value.time_span.start_s, value.node_id)),
            start=1,
        ):
            source_type = str(node.metadata.get("source_node_type") or "")
            if source_type not in DIRECT_EVENT_NODE_TYPES:
                continue
            predicate = str(node.text or "").strip()
            if not predicate:
                continue
            participants = tuple(
                EntityMention(
                    mention_id=str(value["mention_id"]),
                    role=str(value["role"]),
                    entity_type=str(value["entity_type"]),
                    surface=str(value["surface"]),
                    confidence=float(value["confidence"]),
                    grounding_refs=tuple(
                        str(ref) for ref in value.get("grounding_refs") or ()
                    ),
                )
                for value in node.metadata.get("participants") or ()
                if isinstance(value, dict)
                and all(
                    value.get(key)
                    for key in ("mention_id", "role", "entity_type", "surface")
                )
            )
            participant_ids = {value.mention_id for value in participants}
            states = tuple(
                StateAssertion(
                    mention_id=str(value["mention_id"]),
                    attribute=str(value["attribute"]),
                    value=str(value["value"]),
                    confidence=float(value["confidence"]),
                    polarity=str(value.get("polarity") or "positive"),
                )
                for value in node.metadata.get("states") or ()
                if isinstance(value, dict)
                and value.get("mention_id") in participant_ids
                and value.get("attribute")
                and value.get("value")
            )
            events.append(
                AtomicEvent(
                    event_id=f"atomic:video_skills_l1:{index:05d}",
                    video_id=node.video_id,
                    time_span=node.time_span,
                    predicate=predicate,
                    evidence_refs=(node.node_id,),
                    confidence=float(node.metadata.get("confidence") or 0.8),
                    participants=participants,
                    states=states,
                    provenance={
                        "producer": (
                            "memory_graph.video_skills_l1."
                            "VideoSkillsL1AtomicEventExtractor"
                        ),
                        "model": self.model,
                        "layer": "L1.5",
                        "video_skills_l1_passthrough": True,
                        "l1_structuralization": node.metadata.get(
                            "l1_structuralization"
                        ),
                    },
                )
            )
        return events


def _node_text(node: dict[str, Any]) -> str:
    for key in ("text", "description", "content", "scene_description"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_hidden(node: dict[str, Any]) -> bool:
    visibility = node.get("visibility")
    return bool(
        isinstance(visibility, dict)
        and (
            visibility.get("hidden_supervision") is True
            or visibility.get("visible_to_agent") is False
        )
    )


def _timestamp_valid(span: Any) -> bool:
    if not isinstance(span, dict):
        return True
    try:
        start = float(span["start_s"])
        end = float(span["end_s"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(start) and math.isfinite(end) and start >= 0.0 and end >= start


def _confidence_valid(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and 0.0 <= numeric <= 1.0


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _duplicate_local_node_ids(nodes: list[dict[str, Any]]) -> list[str]:
    seen: set[tuple[str, str]] = set()
    duplicates: list[str] = []
    for node in nodes:
        local_id = node.get("local_node_id")
        if not local_id:
            continue
        clip_id = str(
            node.get("local_id_clip_id") or node.get("clip_id") or ""
        )
        if not clip_id:
            continue
        key = (clip_id, str(local_id))
        if key in seen:
            label = f"{clip_id}:{local_id}"
            if label not in duplicates:
                duplicates.append(label)
        else:
            seen.add(key)
    return duplicates


def _validate_clue_memory_schema(graph: dict[str, Any]) -> list[str]:
    schema_path = _clue_memory_schema_path()
    if schema_path is None:
        return []
    try:
        import jsonschema
    except ImportError:
        return []
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(graph), key=lambda err: list(err.path))
    except Exception as exc:  # noqa: BLE001 - audit must stay resilient
        return [f"{type(exc).__name__}: {exc}"]
    messages: list[str] = []
    for error in errors[:5]:
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{path}: {error.message}")
    return messages


def _clue_memory_schema_path() -> Path | None:
    candidate = (
        Path(__file__).resolve().parents[2]
        / "Video_Skills"
        / "schemas"
        / "clue_memory_graph.schema.json"
    )
    return candidate if candidate.is_file() else None
