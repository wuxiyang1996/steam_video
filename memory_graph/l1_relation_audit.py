"""Prepare and evaluate blinded human labels for native L1 navigation relations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


JUDGMENTS = frozenset({"supported", "unsupported", "contradicted", "unclear"})
IDENTITY_RELATIONS = frozenset(
    {"same_entity", "same_object", "same_instance_candidate", "reappears_candidate"}
)
DECISION_COVERAGE_TARGET = 0.95


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
            "unclear": (
                "Use when visible evidence is insufficient; it lowers conservative "
                "precision and decision coverage."
            ),
        },
        "metric_contract": {
            "decided_precision": (
                "supported / (supported + unsupported + contradicted)"
            ),
            "conservative_precision": "supported / all admitted, including unclear",
            "decision_coverage": "decided / all admitted",
            "acceptance": (
                "conservative_precision >= 0.90 and decision_coverage >= 0.95"
            ),
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
    labels_source = str(
        packet.get("labels_source") or "independent_human"
    ).strip()
    if labels_source not in {"independent_human", "model_provisional"}:
        raise ValueError(f"unsupported labels_source: {labels_source!r}")
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
    target_met = all(
        report["labeled_count"] > 0
        and report["conservative_precision"] is not None
        and report["conservative_precision"] >= 0.9
        and report["decision_coverage"] is not None
        and report["decision_coverage"] >= DECISION_COVERAGE_TARGET
        for report in required
    )
    return {
        "labels_source": labels_source,
        "annotator": annotator,
        "groups": groups,
        "acceptance_target": 0.9,
        "decision_coverage_target": DECISION_COVERAGE_TARGET,
        "provisional_target_met": target_met,
        "acceptance_passed": labels_source == "independent_human" and target_met,
    }


def create_locked_audit_split(
    packet: dict[str, Any],
    *,
    development_fraction: float = 0.4,
    salt: str,
) -> dict[str, Any]:
    """Create a hidden, deterministic relation-stratified development split."""

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one")
    if not salt.strip():
        raise ValueError("split salt must be non-empty")
    grouped: dict[str, list[str]] = {
        "identity": [],
        "state_transition": [],
        "other": [],
    }
    for item in packet.get("items") or []:
        if not isinstance(item, dict) or not item.get("item_id"):
            continue
        relation = str(item.get("relation") or "")
        group = (
            "identity"
            if relation in IDENTITY_RELATIONS
            else "state_transition"
            if relation == "state_transition"
            else "other"
        )
        grouped[group].append(str(item["item_id"]))

    development: list[str] = []
    held_out: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for group, item_ids in grouped.items():
        ordered = sorted(
            item_ids,
            key=lambda item_id: hashlib.sha256(
                f"{salt}\0{group}\0{item_id}".encode()
            ).hexdigest(),
        )
        if len(ordered) <= 1:
            development_count = 0
        else:
            development_count = min(
                len(ordered) - 1,
                max(1, math.floor(len(ordered) * development_fraction + 0.5)),
            )
        development.extend(ordered[:development_count])
        held_out.extend(ordered[development_count:])
        counts[group] = {
            "total": len(ordered),
            "development": development_count,
            "held_out": len(ordered) - development_count,
        }
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": "steam-l1-relation-audit-split/v0.1",
        "packet_sha256": hashlib.sha256(encoded).hexdigest(),
        "development_fraction": development_fraction,
        "warning": "Keep hidden from annotators and do not tune on held-out labels.",
        "group_counts": counts,
        "development_item_ids": sorted(development),
        "held_out_item_ids": sorted(held_out),
    }


def _precision_report(counts: Counter[str]) -> dict[str, Any]:
    decided = counts["supported"] + counts["unsupported"] + counts["contradicted"]
    total = sum(counts.values())
    decided_precision = counts["supported"] / decided if decided else None
    conservative_precision = counts["supported"] / total if total else None
    return {
        "counts": dict(sorted(counts.items())),
        "labeled_count": total,
        "decided_count": decided,
        "decided_precision": decided_precision,
        "conservative_precision": conservative_precision,
        "decision_coverage": decided / total if total else None,
        "strict_precision": conservative_precision,
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
    split = subparsers.add_parser("split")
    split.add_argument("--packet", required=True, type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--development-fraction", type=float, default=0.4)
    split.add_argument("--salt", required=True)
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
    if args.command == "split":
        manifest = create_locked_audit_split(
            packet,
            development_fraction=args.development_fraction,
            salt=args.salt,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest["group_counts"], indent=2))
        return 0
    report = evaluate_l1_relation_packet(packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
