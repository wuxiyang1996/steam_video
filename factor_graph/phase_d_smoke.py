"""Build a small grounded Phase D overlay from a real Video_Skills artifact."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from memory_graph.types import CausalTemporalOverlay, MemoryNode, RelationBelief, RelationStatus
from steam_video_new.implicit_world_model.l15_graph_navigator import load_overlay_artifact


SOURCE_EVENT_IDS = (
    "event:atomic:video_skills_l1:00140",
    "event:atomic:video_skills_l1:00258",
)
SOURCE_EDGE_ID = (
    "relation:event:atomic:video_skills_l1:00140"
    "->event:atomic:video_skills_l1:00258:gpt-oss-teacher"
)
ACCEPTED_TRACK_ID = "l1-track:phase-d:mKqiGQrHtW8:man"


def build_phase_d_overlay(source_path: Path) -> CausalTemporalOverlay:
    """Derive a reproducible coupled fixture while retaining source evidence refs."""

    source = load_overlay_artifact(source_path).overlay
    event_by_id = {node.node_id: node for node in source.atomic_events}
    edge_by_id = {edge.edge_id: edge for edge in source.relations}
    src = deepcopy(event_by_id[SOURCE_EVENT_IDS[0]])
    dst = deepcopy(event_by_id[SOURCE_EVENT_IDS[1]])
    source_edge = edge_by_id[SOURCE_EDGE_ID]

    src_old = _person_mention_id(src)
    dst_old = _person_mention_id(dst)
    _replace_mention_id(src, src_old, ACCEPTED_TRACK_ID)
    _replace_mention_id(dst, dst_old, ACCEPTED_TRACK_ID)
    _normalize_state_synonym(dst, ACCEPTED_TRACK_ID, "gaze_direction", "looking down")
    evidence_ids = tuple(dict.fromkeys(src.source_segments + dst.source_segments))
    l1_by_id = {node.node_id: node for node in source.l1_observations}
    l1 = [deepcopy(l1_by_id[node_id]) for node_id in evidence_ids]

    alignment = {"src": ACCEPTED_TRACK_ID, "dst": ACCEPTED_TRACK_ID}
    base_provenance = {
        "producer": "factor_graph.phase_d_smoke",
        "source_edge_id": source_edge.edge_id,
        "participant_alignment": alignment,
        "accepted_identity_track": ACCEPTED_TRACK_ID,
        "derivation": (
            "accepted-track remap of the source edge's independently structured "
            "person alignment; fixture is an integration smoke, not a gold label"
        ),
    }
    src_quote = str(src.metadata["predicate"])
    dst_quote = str(dst.metadata["predicate"])
    warrant = f"{src_quote} Then {dst_quote}"
    common = {
        "src": src.node_id,
        "dst": dst.node_id,
        "status": RelationStatus.UNCALIBRATED_PRIOR,
        "direction_confidence": source_edge.direction_confidence,
        "evidence_refs": list(evidence_ids),
    }
    relations = [
        RelationBelief(
            edge_id="phase-d:identity",
            relation_probabilities={"same_entity": source_edge.relation_probabilities["same_entity"]},
            provenance=dict(base_provenance),
            **common,
        ),
        RelationBelief(
            edge_id="phase-d:state",
            relation_probabilities={"state_transition": 0.2},
            provenance={
                **base_provenance,
                "state_delta": {
                    "mention_id": ACCEPTED_TRACK_ID,
                    "attribute": "expression",
                    "before": "neutral",
                    "after": "focused",
                },
                "prior_note": "conservative pilot prior; not model- or human-calibrated",
            },
            **common,
        ),
        RelationBelief(
            edge_id="phase-d:dependency",
            relation_probabilities={"transition_support": 0.55},
            warrant=warrant,
            provenance={
                **base_provenance,
                "dependency_basis": "state_continuity",
                "evidence_quotes": {"src": src_quote, "dst": dst_quote},
            },
            **common,
        ),
        RelationBelief(
            edge_id="phase-d:conflict",
            relation_probabilities={"contradicts": 0.2},
            provenance={
                **base_provenance,
                "conflict_basis": "same-track expression values are incompatible",
            },
            **common,
        ),
    ]
    return CausalTemporalOverlay(
        overlay_id="overlay:phase-d:grounded-coupled-smoke:v1",
        example_id=source.example_id,
        video_id=source.video_id,
        l1_observations=l1,
        atomic_events=[src, dst],
        relations=relations,
        metadata={
            "layer_contract": "l1_observations_plus_l1_5_atomic_overlay",
            "status": "derived_grounded_integration_smoke_not_gold_benchmark",
            "source_overlay_id": source.overlay_id,
            "source_artifact": str(source_path.resolve()),
            "source_artifact_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_edge_id": source_edge.edge_id,
            "input_mode": source.metadata.get("input_mode"),
            "accepted_identity_track": ACCEPTED_TRACK_ID,
            "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
            "llm_numeric_output": False,
        },
    )


def write_phase_d_overlay(source_path: Path, output_path: Path) -> CausalTemporalOverlay:
    overlay = build_phase_d_overlay(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_portable_embeddings(overlay, output_path)
    output_path.write_text(
        json.dumps(overlay.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return overlay


def _write_portable_embeddings(
    overlay: CausalTemporalOverlay,
    output_path: Path,
) -> None:
    import numpy as np

    nodes = [node for node in overlay.atomic_events if node.embedding_ref is not None]
    if len(nodes) != len(overlay.atomic_events):
        raise ValueError("Phase D smoke requires one embedding ref per event")
    rows = []
    for node in nodes:
        ref = node.embedding_ref
        assert ref is not None
        if ref.row_index is None:
            raise ValueError(f"embedding ref lacks row_index: {node.node_id}")
        matrix = np.load(ref.path, mmap_mode="r")
        rows.append(np.asarray(matrix[ref.row_index], dtype=np.float32))
    embedding_path = output_path.with_name(
        f"{output_path.stem}_embeddings.npy"
    )
    np.save(embedding_path, np.stack(rows))
    checksum = hashlib.sha256(embedding_path.read_bytes()).hexdigest()
    for index, node in enumerate(nodes):
        ref = node.embedding_ref
        assert ref is not None
        node.embedding_ref = replace(
            ref,
            path=embedding_path.name,
            row_index=index,
            checksum=checksum,
        )
    overlay.metadata["embedding_artifact"] = embedding_path.name
    overlay.metadata["embedding_artifact_sha256"] = checksum


def _person_mention_id(node: MemoryNode) -> str:
    for participant in node.metadata.get("participants") or []:
        if str(participant.get("entity_type") or "").casefold() == "person":
            return str(participant["mention_id"])
    raise ValueError(f"event {node.node_id} has no structured person participant")


def _replace_mention_id(node: MemoryNode, old: str, new: str) -> None:
    for key in ("participants", "states"):
        for row in node.metadata.get(key) or []:
            if row.get("mention_id") == old:
                row["mention_id"] = new
    node.provenance = {
        **node.provenance,
        "phase_d_track_remap": {"source_mention_id": old, "accepted_track_id": new},
    }


def _normalize_state_synonym(
    node: MemoryNode,
    mention_id: str,
    attribute: str,
    canonical_value: str,
) -> None:
    for row in node.metadata.get("states") or []:
        if row.get("mention_id") != mention_id or row.get("attribute") != attribute:
            continue
        source_value = str(row.get("value") or "")
        row["value"] = canonical_value
        row["phase_d_normalization"] = {
            "source_value": source_value,
            "reason": "directional synonym; prevents a false state delta in smoke",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    overlay = write_phase_d_overlay(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "overlay_id": overlay.overlay_id,
                "relation_count": len(overlay.relations),
                "embedding_ref_count": sum(
                    node.embedding_ref is not None for node in overlay.atomic_events
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
