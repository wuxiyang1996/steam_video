"""Prepare and evaluate blinded human labels for native L1 navigation relations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


JUDGMENTS = frozenset({"supported", "unsupported", "contradicted", "unclear"})
IDENTITY_RELATIONS = frozenset(
    {"same_entity", "same_object", "same_instance_candidate", "reappears_candidate"}
)


def prepare_l1_relation_packet(
    overlay: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = {
        str(node["node_id"]): node
        for node in overlay.get("l1_observations") or []
        if isinstance(node, dict) and node.get("node_id")
    }
    packet_items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    for ordinal, edge in enumerate(overlay.get("l1_structural_relations") or [], 1):
        if not isinstance(edge, dict):
            continue
        probabilities = edge.get("relation_probabilities") or {}
        if len(probabilities) != 1:
            continue
        relation = str(next(iter(probabilities)))
        src_id, dst_id = str(edge.get("src") or ""), str(edge.get("dst") or "")
        if src_id not in nodes or dst_id not in nodes:
            continue
        item_id = f"l1-edge:{ordinal:04d}"
        packet_items.append(
            {
                "item_id": item_id,
                "relation": relation,
                "source": _node_context(nodes[src_id]),
                "destination": _node_context(nodes[dst_id]),
                "annotation": {
                    "judgment": None,
                    "reason": None,
                    "identity_evidence": None,
                    "state_delta_evidence": None,
                },
            }
        )
        key_items.append(
            {
                "item_id": item_id,
                "edge_id": edge.get("edge_id"),
                "src": src_id,
                "dst": dst_id,
                "relation": relation,
                "probability": probabilities[relation],
                "provenance": edge.get("provenance") or {},
            }
        )
    packet = {
        "schema_version": "steam-l1-relation-audit/v0.1",
        "protocol_version": "l1-relation-precision/v1",
        "annotator": None,
        "instructions": {
            "independence": "Do not inspect model_key.json or prior model audits.",
            "judgments": sorted(JUDGMENTS),
            "identity": (
                "Support only with compatible type, stable attributes, trajectory/context "
                "continuity, and no simultaneous-distinct or impossible-motion evidence."
            ),
            "state_transition": (
                "Support only when both endpoints ground the same accepted identity, the "
                "same attribute, and visibly different before/after values."
            ),
            "unclear": "Use when visible evidence is insufficient; excluded from precision.",
        },
        "items": packet_items,
    }
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    key = {
        "schema_version": "steam-l1-relation-model-key/v0.1",
        "packet_sha256": hashlib.sha256(encoded).hexdigest(),
        "warning": "Keep hidden until the annotation packet is locked.",
        "items": key_items,
    }
    return packet, key


def evaluate_l1_relation_packet(packet: dict[str, Any]) -> dict[str, Any]:
    annotator = str(packet.get("annotator") or "").strip()
    if not annotator:
        raise ValueError("independent audit requires a non-empty annotator")
    counts: dict[str, Counter[str]] = {
        "identity": Counter(),
        "state_transition": Counter(),
        "other": Counter(),
    }
    for item in packet.get("items") or []:
        if not isinstance(item, dict):
            continue
        annotation = item.get("annotation") or {}
        judgment = str(annotation.get("judgment") or "").casefold()
        if judgment not in JUDGMENTS:
            raise ValueError(f"{item.get('item_id')} has invalid or missing judgment")
        relation = str(item.get("relation") or "")
        group = (
            "identity"
            if relation in IDENTITY_RELATIONS
            else "state_transition"
            if relation == "state_transition"
            else "other"
        )
        counts[group][judgment] += 1
    groups = {name: _precision_report(counter) for name, counter in counts.items()}
    required = (groups["identity"], groups["state_transition"])
    return {
        "labels_source": "independent_human",
        "annotator": annotator,
        "groups": groups,
        "acceptance_target": 0.9,
        "acceptance_passed": all(
            report["labeled_count"] > 0
            and report["strict_precision"] is not None
            and report["strict_precision"] >= 0.9
            for report in required
        ),
    }


def _precision_report(counts: Counter[str]) -> dict[str, Any]:
    decided = counts["supported"] + counts["unsupported"] + counts["contradicted"]
    return {
        "counts": dict(sorted(counts.items())),
        "labeled_count": sum(counts.values()),
        "decided_count": decided,
        "strict_precision": counts["supported"] / decided if decided else None,
    }


def _node_context(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "time_span": node.get("time_span"),
        "text": node.get("text"),
        "source_node_id": node.get("source_node_id"),
        "source_segments": node.get("source_segments") or [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--overlay", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--packet", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = json.loads(args.overlay.read_text(encoding="utf-8"))
        packet, key = prepare_l1_relation_packet(payload)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "annotation_packet.json").write_text(
            json.dumps(packet, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "model_key.json").write_text(
            json.dumps(key, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"item_count": len(packet["items"])}))
        return 0
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    report = evaluate_l1_relation_packet(packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
