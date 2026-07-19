"""Re-run current hard verifiers over persisted overlay candidates without API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .types import MemoryNode, RelationBelief, RelationStatus, TimeSpan
from .verifiers import verify_relation


def _node(payload: dict[str, Any]) -> MemoryNode:
    return MemoryNode(
        node_id=str(payload["node_id"]),
        video_id=str(payload["video_id"]),
        time_span=TimeSpan.from_dict(payload["time_span"]),
        provenance=dict(payload.get("provenance") or {}),
        node_type=str(payload.get("node_type") or ""),
        text=payload.get("text"),
        source_node_id=payload.get("source_node_id"),
        source_segments=[str(value) for value in payload.get("source_segments") or []],
        metadata=dict(payload.get("metadata") or {}),
    )


def reverify(overlay: dict[str, Any]) -> dict[str, Any]:
    l1 = {
        node.node_id: node
        for node in (
            _node(value) for value in overlay.get("l1_observations") or []
        )
    }
    events = {
        node.node_id: node
        for node in (_node(value) for value in overlay.get("atomic_events") or [])
    }
    accepted_edges: list[dict[str, Any]] = []
    newly_rejected: list[dict[str, Any]] = []
    for edge in overlay.get("relations") or []:
        if not isinstance(edge, dict) or edge.get("status") == "deterministic":
            accepted_edges.append(edge)
            continue
        kept: dict[str, float] = {}
        decisions: dict[str, Any] = {}
        for relation, probability in (edge.get("relation_probabilities") or {}).items():
            belief = RelationBelief(
                edge_id=str(edge["edge_id"]),
                src=str(edge["src"]),
                dst=str(edge["dst"]),
                relation_probabilities={str(relation): float(probability)},
                status=RelationStatus(str(edge["status"])),
                direction_confidence=float(edge.get("direction_confidence") or 0.0),
                evidence_refs=[
                    str(value) for value in edge.get("evidence_refs") or []
                ],
                warrant=edge.get("warrant"),
                provenance=dict(edge.get("provenance") or {}),
                features={
                    str(key): float(value)
                    for key, value in (edge.get("features") or {}).items()
                },
            )
            result = verify_relation(
                belief,
                src=events[belief.src],
                dst=events[belief.dst],
                l1_by_id=l1,
            )
            decisions[str(relation)] = result.to_dict()
            if result.passed:
                kept[str(relation)] = float(probability)
            else:
                newly_rejected.append(
                    {
                        "edge_id": belief.edge_id,
                        "src": belief.src,
                        "dst": belief.dst,
                        "relation": str(relation),
                        "probability": float(probability),
                        "reasons": list(result.reasons),
                        "reverification": "l1_temporal_grounding/v1",
                    }
                )
        if kept:
            updated = dict(edge)
            updated["relation_probabilities"] = kept
            updated["provenance"] = {
                **dict(edge.get("provenance") or {}),
                "hard_verifier": decisions,
            }
            accepted_edges.append(updated)

    result = dict(overlay)
    result["relations"] = accepted_edges
    metadata = dict(result.get("metadata") or {})
    metadata["reverified_with"] = "l1_temporal_grounding/v1"
    result["metadata"] = metadata
    build_report = dict(result.get("build_report") or {})
    previous = list(build_report.get("rejected_relations") or [])
    build_report["rejected_relations"] = previous + newly_rejected
    summary = dict(build_report.get("verifier_summary") or {})
    summary["accepted_labels"] = max(
        0, int(summary.get("accepted_labels") or 0) - len(newly_rejected)
    )
    summary["rejected_labels"] = int(summary.get("rejected_labels") or 0) + len(
        newly_rejected
    )
    proposed = int(summary.get("proposed_labels_at_threshold") or 0)
    summary["acceptance_rate"] = (
        summary["accepted_labels"] / proposed if proposed else None
    )
    summary["newly_rejected_by_l1_temporal_grounding"] = len(newly_rejected)
    build_report["verifier_summary"] = summary
    result["build_report"] = build_report
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    totals = {"video_count": 0, "newly_rejected": 0}
    for path in sorted(args.run_dir.glob("*/causal_temporal_overlay.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = reverify(payload)
        newly_rejected = (
            updated.get("build_report", {})
            .get("verifier_summary", {})
            .get("newly_rejected_by_l1_temporal_grounding", 0)
        )
        backup = path.with_name("causal_temporal_overlay.pre_l1_temporal_reverify.json")
        if not backup.exists():
            backup.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        totals["video_count"] += 1
        totals["newly_rejected"] += int(newly_rejected)
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
