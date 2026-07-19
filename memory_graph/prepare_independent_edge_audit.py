"""Prepare blinded candidate-edge packets for independent human annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SEMANTIC_RELATIONS = {
    "same_entity",
    "same_object",
    "same_instance_candidate",
    "reappears_candidate",
    "state_transition",
    "observation_support",
    "transition_support",
    "response_candidate",
    "explains",
    "enables",
    "contradicts",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def prepare_packets(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    annotation_items: list[dict[str, Any]] = []
    model_items: list[dict[str, Any]] = []
    for graph_path in sorted(run_dir.glob("*/causal_temporal_overlay.json")):
        overlay = _load(graph_path)
        video_id = str(overlay.get("video_id") or graph_path.parent.name)
        events = {
            str(node["node_id"]): node
            for node in overlay.get("atomic_events") or []
            if isinstance(node, dict) and node.get("node_id")
        }
        l1 = {
            str(node["node_id"]): node
            for node in overlay.get("l1_observations") or []
            if isinstance(node, dict) and node.get("node_id")
        }
        candidates = _candidate_rows(overlay)
        for ordinal, candidate in enumerate(candidates, start=1):
            src_id = candidate["src"]
            dst_id = candidate["dst"]
            src = events.get(src_id)
            dst = events.get(dst_id)
            if src is None or dst is None:
                continue
            item_id = f"{video_id}:edge:{ordinal:04d}"
            annotation_items.append(
                {
                    "item_id": item_id,
                    "video_id": video_id,
                    "relation": candidate["relation"],
                    "source_event": _event_context(src, l1),
                    "destination_event": _event_context(dst, l1),
                    "annotation": {
                        "judgment": None,
                        "reason": None,
                        "minimal_support_sufficient": None,
                    },
                }
            )
            model_items.append(
                {
                    "item_id": item_id,
                    "video_id": video_id,
                    **candidate,
                    "graph_path": str(graph_path),
                }
            )

    packet = {
        "schema_version": "steam-independent-edge-audit/v0.1",
        "instructions": {
            "annotator_independence": (
                "Do not inspect model_key.json, GPT self-audits, probabilities, "
                "or verifier decisions before labeling."
            ),
            "judgment_values": [
                "supported",
                "unsupported",
                "contradicted",
                "unclear",
            ],
            "supported": (
                "The visible quoted events are sufficient for this exact typed "
                "relation; temporal order or shared narrative alone is insufficient."
            ),
            "unclear": "Insufficient visible evidence; excluded from calibration.",
        },
        "annotator": None,
        "protocol_version": "independent-edge-audit/v0.1",
        "items": annotation_items,
    }
    model_key = {
        "schema_version": "steam-edge-model-key/v0.1",
        "warning": "Keep hidden from annotators until labels are locked.",
        "items": model_items,
    }
    return packet, model_key


def _candidate_rows(overlay: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in overlay.get("relations") or []:
        if not isinstance(edge, dict) or edge.get("status") == "deterministic":
            continue
        for relation, probability in (edge.get("relation_probabilities") or {}).items():
            if relation not in SEMANTIC_RELATIONS:
                continue
            key = (str(edge.get("src")), str(edge.get("dst")), str(relation))
            rows[key] = {
                "src": key[0],
                "dst": key[1],
                "relation": key[2],
                "probability": float(probability),
                "post_verifier": True,
                "verifier_reasons": [],
                "teacher_warrant": edge.get("warrant"),
            }
    for rejected in (overlay.get("build_report") or {}).get("rejected_relations") or []:
        if not isinstance(rejected, dict):
            continue
        relation = str(rejected.get("relation"))
        if relation not in SEMANTIC_RELATIONS:
            continue
        key = (str(rejected.get("src")), str(rejected.get("dst")), relation)
        rows.setdefault(
            key,
            {
                "src": key[0],
                "dst": key[1],
                "relation": relation,
                "probability": float(rejected.get("probability") or 0.0),
                "post_verifier": False,
                "verifier_reasons": list(rejected.get("reasons") or []),
                "teacher_warrant": None,
            },
        )
    return [rows[key] for key in sorted(rows)]


def _event_context(
    event: dict[str, Any],
    l1_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    refs = [str(value) for value in event.get("source_segments") or []]
    return {
        "event_id": event.get("node_id"),
        "time_span": event.get("time_span"),
        "predicate": (event.get("metadata") or {}).get("predicate") or event.get("text"),
        "participants": (event.get("metadata") or {}).get("participants") or [],
        "states": (event.get("metadata") or {}).get("states") or [],
        "l1_evidence": [
            {
                "node_id": ref,
                "time_span": (l1_by_id.get(ref) or {}).get("time_span"),
                "text": (l1_by_id.get(ref) or {}).get("text"),
            }
            for ref in refs
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    packet, model_key = prepare_packets(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "annotation_packet.json").write_text(
        json.dumps(packet, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "model_key.json").write_text(
        json.dumps(model_key, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "item_count": len(packet["items"]),
                "annotation_packet": str(args.output_dir / "annotation_packet.json"),
                "model_key": str(args.output_dir / "model_key.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
