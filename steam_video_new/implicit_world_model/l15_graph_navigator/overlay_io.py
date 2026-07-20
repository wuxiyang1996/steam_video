"""Strict JSON loading for persisted L1/L1.5 overlay artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from memory_graph.schema_validation import require_valid_overlay_artifact
from memory_graph.types import (
    CausalTemporalOverlay,
    EmbeddingRef,
    MemoryNode,
    RelationBelief,
    RelationStatus,
    TimeSpan,
)


@dataclass(frozen=True)
class LoadedOverlayArtifact:
    overlay: CausalTemporalOverlay
    source_path: Path
    build_report: dict[str, Any]
    embedding_problems: tuple[str, ...] = ()


def load_overlay_artifact(
    path: str | Path,
    *,
    validate_schema: bool = True,
    require_embedding_files: bool = False,
    verify_embedding_checksums: bool = False,
) -> LoadedOverlayArtifact:
    """Load one serialized overlay without dropping embedding/provenance data."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("overlay artifact root must be a JSON object")
    if validate_schema:
        require_valid_overlay_artifact(payload)
    overlay = overlay_from_dict(payload)
    _resolve_relative_embedding_paths(overlay, source.parent)
    problems = _embedding_problems(
        overlay,
        verify_checksums=verify_embedding_checksums,
    )
    if require_embedding_files and problems:
        raise ValueError("invalid embedding references: " + "; ".join(problems[:5]))
    return LoadedOverlayArtifact(
        overlay=overlay,
        source_path=source,
        build_report=dict(payload.get("build_report") or {}),
        embedding_problems=tuple(problems),
    )


def _resolve_relative_embedding_paths(
    overlay: CausalTemporalOverlay,
    artifact_directory: Path,
) -> None:
    """Resolve portable serialized refs relative to their overlay artifact."""

    for node in overlay.l1_observations + overlay.atomic_events:
        ref = node.embedding_ref
        if ref is None or Path(ref.path).is_absolute():
            continue
        node.embedding_ref = replace(
            ref,
            path=str((artifact_directory / ref.path).resolve()),
        )


def overlay_from_dict(payload: dict[str, Any]) -> CausalTemporalOverlay:
    return CausalTemporalOverlay(
        overlay_id=str(payload.get("overlay_id") or ""),
        example_id=str(payload.get("example_id") or ""),
        video_id=str(payload.get("video_id") or ""),
        l1_observations=[
            _memory_node_from_dict(row)
            for row in payload.get("l1_observations") or []
        ],
        atomic_events=[
            _memory_node_from_dict(row)
            for row in payload.get("atomic_events") or []
        ],
        relations=[
            _relation_from_dict(row) for row in payload.get("relations") or []
        ],
        l1_structural_relations=[
            _relation_from_dict(row)
            for row in payload.get("l1_structural_relations") or []
        ],
        schema_version=str(payload.get("schema_version") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


def _memory_node_from_dict(payload: dict[str, Any]) -> MemoryNode:
    embedding = payload.get("embedding_ref")
    return MemoryNode(
        node_id=str(payload.get("node_id") or ""),
        video_id=str(payload.get("video_id") or ""),
        time_span=TimeSpan.from_dict(dict(payload.get("time_span") or {})),
        provenance=dict(payload.get("provenance") or {}),
        node_type=str(payload.get("node_type") or "event"),
        text=(str(payload["text"]) if payload.get("text") is not None else None),
        source_node_id=(
            str(payload["source_node_id"])
            if payload.get("source_node_id") is not None
            else None
        ),
        source_segments=[str(value) for value in payload.get("source_segments") or []],
        embedding_ref=(
            EmbeddingRef(
                path=str(embedding.get("path") or ""),
                model=str(embedding.get("model") or ""),
                dimension=int(embedding.get("dimension") or 0),
                dtype=str(embedding.get("dtype") or "float32"),
                normalized=bool(embedding.get("normalized", True)),
                row_index=(
                    int(embedding["row_index"])
                    if embedding.get("row_index") is not None
                    else None
                ),
                checksum=(
                    str(embedding["checksum"])
                    if embedding.get("checksum") is not None
                    else None
                ),
            )
            if isinstance(embedding, dict)
            else None
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def _relation_from_dict(payload: dict[str, Any]) -> RelationBelief:
    return RelationBelief(
        edge_id=str(payload.get("edge_id") or ""),
        src=str(payload.get("src") or ""),
        dst=str(payload.get("dst") or ""),
        relation_probabilities={
            str(name): float(value)
            for name, value in (payload.get("relation_probabilities") or {}).items()
        },
        status=RelationStatus(str(payload.get("status") or "")),
        direction_confidence=float(payload.get("direction_confidence") or 0.0),
        evidence_refs=[str(value) for value in payload.get("evidence_refs") or []],
        warrant=(
            str(payload["warrant"])
            if payload.get("warrant") is not None
            else None
        ),
        provenance=dict(payload.get("provenance") or {}),
        features={
            str(name): float(value)
            for name, value in (payload.get("features") or {}).items()
        },
    )


def _embedding_problems(
    overlay: CausalTemporalOverlay,
    *,
    verify_checksums: bool,
) -> list[str]:
    problems: list[str] = []
    checksums: dict[Path, str] = {}
    for node in overlay.l1_observations + overlay.atomic_events:
        ref = node.embedding_ref
        if ref is None:
            continue
        path = Path(ref.path).expanduser()
        if not path.is_file():
            problems.append(f"{node.node_id}: embedding file does not exist: {path}")
            continue
        if ref.row_index is not None and ref.row_index < 0:
            problems.append(f"{node.node_id}: embedding row_index must be non-negative")
        if verify_checksums and ref.checksum:
            checksum = checksums.get(path)
            if checksum is None:
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                checksums[path] = checksum
            if checksum != ref.checksum:
                problems.append(f"{node.node_id}: embedding checksum mismatch: {path}")
    return problems
