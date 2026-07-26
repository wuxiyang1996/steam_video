"""Load frozen L1/L1.5 navigation artifacts without rebuilding them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memory_graph.types import EmbeddingRef, MemoryNode, TimeSpan

from ..navigation.contracts import (
    NavigationGraph,
    NavigationProposal,
    ProposalCalibration,
    ProposalKind,
)
from .contracts import EvidenceMemory, TemporalLink
from .legacy_adapter import record_from_memory_node


def load_navigation_artifact(
    path: str | Path,
    *,
    entry_node_ids: tuple[str, ...] = (),
) -> tuple[EvidenceMemory, NavigationGraph]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("navigation artifact root must be an object")
    nodes = tuple(_node(row) for row in payload.get("nodes") or ())
    records = tuple(record_from_memory_node(node) for node in nodes)
    temporal = tuple(
        TemporalLink(
            str(row.get("edge_id") or ""),
            str(row.get("src") or ""),
            str(row.get("dst") or ""),
            str(row.get("relation") or ""),
        )
        for row in payload.get("temporal_edges") or ()
        if isinstance(row, dict)
    )
    temporal_proposals = tuple(
        NavigationProposal(
            proposal_id=link.edge_id,
            src=link.src,
            dst=link.dst,
            kind=(
                ProposalKind.EVENT_CONTINUATION_CANDIDATE
                if link.relation in {"temporal_next", "before"}
                else ProposalKind.TEMPORAL
            ),
            bidirectional=True,
            evidence_refs=(link.src, link.dst),
            descriptor=(link.relation,),
            calibration=ProposalCalibration.STRUCTURAL,
        )
        for link in temporal
    )
    correlations = tuple(
        NavigationProposal(
            proposal_id=str(row.get("edge_id") or ""),
            src=str(row.get("src") or ""),
            dst=str(row.get("dst") or ""),
            kind=(
                ProposalKind.SEMANTIC_RECURRENCE
                if row.get("channel") == "semantic_recurrence"
                else ProposalKind.SEMANTIC_NEIGHBOR
            ),
            bidirectional=True,
            evidence_refs=tuple(str(value) for value in row.get("evidence_refs") or ()),
            descriptor=(str(row.get("channel") or "semantic"),),
            calibration=ProposalCalibration.UNCALIBRATED,
            audit_metadata={
                "soft_affinity_present": True,
                "affinity_exposed_to_model": False,
                "source_artifact": str(source),
            },
        )
        for row in payload.get("correlation_edges") or ()
        if isinstance(row, dict)
    )
    graph_id = str(payload.get("graph_id") or "")
    memory = EvidenceMemory(
        memory_id=f"evidence-v2:{graph_id}",
        records=records,
        temporal_links=temporal,
        metadata={
            "source_path": str(source),
            "source_graph_id": graph_id,
            "question_independent": bool(
                (payload.get("metadata") or {}).get("question_independent")
            ),
        },
    )
    navigation = NavigationGraph(
        graph_id=f"navigation-v2:{graph_id}",
        node_ids=tuple(record.address.node_id for record in records),
        proposals=(*temporal_proposals, *correlations),
        entry_node_ids=entry_node_ids,
        metadata={
            "source_path": str(source),
            "top_k_applied": False,
            "soft_affinity_exposed_to_model": False,
        },
    )
    return memory, navigation


def _node(payload: Any) -> MemoryNode:
    if not isinstance(payload, dict):
        raise ValueError("navigation artifact node must be an object")
    return MemoryNode(
        node_id=str(payload.get("node_id") or ""),
        video_id=str(payload.get("video_id") or ""),
        time_span=TimeSpan.from_dict(dict(payload.get("time_span") or {})),
        provenance=dict(payload.get("provenance") or {}),
        node_type=str(payload.get("node_type") or "event"),
        text=str(payload["text"]) if payload.get("text") is not None else None,
        source_node_id=(
            str(payload["source_node_id"])
            if payload.get("source_node_id") is not None
            else None
        ),
        source_segments=[str(value) for value in payload.get("source_segments") or ()],
        embedding_ref=_embedding(payload.get("embedding_ref")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _embedding(payload: Any) -> EmbeddingRef | None:
    if not isinstance(payload, dict):
        return None
    return EmbeddingRef(
        path=str(payload.get("path") or ""),
        model=str(payload.get("model") or ""),
        dimension=int(payload.get("dimension") or 0),
        dtype=str(payload.get("dtype") or "float32"),
        normalized=bool(payload.get("normalized", True)),
        row_index=(
            int(payload["row_index"]) if payload.get("row_index") is not None else None
        ),
        checksum=(
            str(payload["checksum"]) if payload.get("checksum") is not None else None
        ),
    )
