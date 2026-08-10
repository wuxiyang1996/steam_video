"""Audit whether existing L1 artifacts can truthfully supply four feature slots."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import SLOT_NAMES


def audit_overlays(paths: Iterable[Path]) -> dict[str, Any]:
    unique_nodes: dict[tuple[str, str], dict[str, Any]] = {}
    overlay_count = 0
    for path in paths:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        overlay_count += 1
        for node in payload.get("l1_observations") or ():
            if not isinstance(node, dict):
                continue
            key = (str(node.get("video_id") or ""), str(node.get("node_id") or ""))
            unique_nodes.setdefault(key, node)
    counts = Counter()
    boundary_failures = Counter()
    for node in unique_nodes.values():
        metadata = node.get("metadata") or {}
        provenance = node.get("provenance") or {}
        visibility = metadata.get("visibility") or {}
        safe = (
            provenance.get("uses_hidden_supervision") is not True
            and visibility.get("hidden_supervision") is not True
        )
        if not safe:
            boundary_failures["hidden_supervision_present"] += 1
        caption_source = bool(str(node.get("text") or "").strip())
        entity_source = bool(metadata.get("participants") or metadata.get("states"))
        time_source = _valid_time(node.get("time_span") or {})
        embedding = node.get("embedding_ref") or {}
        combined_text_embedding = bool(embedding.get("path") and embedding.get("row_index") is not None)
        visual_embedding = _is_explicit_visual_embedding(embedding)
        for name, available in {
            "caption_source": caption_source,
            "entity_state_source": entity_source,
            "time_source": time_source,
            "combined_text_embedding": combined_text_embedding,
            "explicit_visual_embedding": visual_embedding,
            "safe_pre_read_source": safe,
        }.items():
            counts[name] += int(available)
        four_slot_ready = safe and caption_source and entity_source and visual_embedding and time_source
        counts["complete_four_slot_ready"] += int(four_slot_ready)

    total = len(unique_nodes)
    coverage = {
        name: (float(counts[name]) / total if total else 0.0)
        for name in (
            "caption_source",
            "entity_state_source",
            "time_source",
            "combined_text_embedding",
            "explicit_visual_embedding",
            "safe_pre_read_source",
            "complete_four_slot_ready",
        )
    }
    blockers: list[str] = []
    if coverage["explicit_visual_embedding"] < 1.0:
        blockers.append("independent_visual_embeddings_missing")
    if coverage["entity_state_source"] < 1.0:
        blockers.append("entity_state_source_incomplete")
    if boundary_failures:
        blockers.append("pre_read_boundary_failures_present")
    if coverage["complete_four_slot_ready"] < 1.0:
        blockers.append("complete_four_slot_training_not_ready")
    return {
        "schema_version": "steam-qformer-feature-readiness/v0.1",
        "overlay_count": overlay_count,
        "unique_video_count": len({key[0] for key in unique_nodes}),
        "unique_node_count": total,
        "counts": dict(counts),
        "coverage": coverage,
        "boundary_failures": dict(boundary_failures),
        "blockers": blockers,
        "qf1_training_ready": not blockers,
        "interpretation": {
            "combined_text_embedding": (
                "Existing event+participants+states text embedding; it is not an "
                "independent visual slot."
            ),
            "missing_slot_policy": "mask for engineering smoke; fail closed for formal four-slot training",
        },
    }


def _valid_time(payload: dict[str, Any]) -> bool:
    try:
        start = float(payload["start_s"])
        end = float(payload["end_s"])
    except (KeyError, TypeError, ValueError):
        return False
    return start >= 0.0 and end >= start


def _is_explicit_visual_embedding(payload: dict[str, Any]) -> bool:
    modality = str(payload.get("input_modality") or "").casefold()
    contract = str(payload.get("source_contract") or "").casefold()
    return bool(payload.get("path")) and (
        modality in {"visual", "video", "multimodal"}
        or "visual" in contract
        or "video" in contract
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = sorted(args.root.expanduser().resolve().glob("**/causal_temporal_overlay.json"))
    if not paths:
        raise ValueError("no causal_temporal_overlay.json files found")
    report = audit_overlays(paths)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["qf1_training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

