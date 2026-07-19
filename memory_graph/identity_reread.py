"""Prepare and apply grounded raw-video identity reread decisions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .identity_verifier import TRUSTED_REREAD_SOURCES, verify_identity_candidates


def prepare_identity_reread_packet(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = {
        str(node["node_id"]): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and node.get("node_id")
    }
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    _, report = verify_identity_candidates(nodes, edges)
    edge_by_id = {
        str(edge.get("edge_id")): edge for edge in edges if edge.get("edge_id")
    }
    items: list[dict[str, Any]] = []
    for row in report.targeted_reread_queue:
        edge_id = str(row["edge_id"])
        edge = edge_by_id[edge_id]
        src, dst = nodes[str(edge["src"])], nodes[str(edge["dst"])]
        items.append(
            {
                "item_id": f"identity-reread:{len(items) + 1:04d}",
                "edge_id": edge_id,
                "relation": edge.get("edge_type"),
                "source": _node_context(src),
                "destination": _node_context(dst),
                "annotation": {
                    "passed": None,
                    "src_evidence_ref": None,
                    "dst_evidence_ref": None,
                    "matched_attributes": [],
                    "conflicts": [],
                    "reason": None,
                },
            }
        )
    graph_digest = hashlib.sha256(
        json.dumps(
            {"nodes": graph.get("nodes") or [], "edges": graph.get("edges") or []},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema_version": "steam-identity-reread/v0.1",
        "graph_sha256": graph_digest,
        "annotator": None,
        "labels_source": None,
        "instructions": {
            "scope": "Inspect only the raw-video spans referenced by both endpoints.",
            "pass": (
                "Pass only with visible positive identity evidence and no type, stable "
                "attribute, simultaneous-distinct, or motion conflict."
            ),
            "evidence": "A passed item requires one concrete evidence ref per endpoint.",
        },
        "items": items,
    }


def apply_identity_reread_packet(
    graph: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    annotator = str(packet.get("annotator") or "").strip()
    labels_source = str(packet.get("labels_source") or "").strip()
    if not annotator:
        raise ValueError("identity reread requires a non-empty annotator")
    if labels_source not in TRUSTED_REREAD_SOURCES:
        raise ValueError(f"untrusted identity reread labels_source: {labels_source!r}")
    expected_digest = prepare_identity_reread_packet(graph)["graph_sha256"]
    if packet.get("graph_sha256") != expected_digest:
        raise ValueError("identity reread packet does not match the input graph")
    decisions: dict[str, dict[str, Any]] = {}
    for item in packet.get("items") or []:
        if not isinstance(item, dict) or not item.get("edge_id"):
            continue
        if str(item["edge_id"]) in decisions:
            raise ValueError(f"duplicate identity reread edge: {item['edge_id']}")
        annotation = item.get("annotation") or {}
        if not isinstance(annotation.get("passed"), bool):
            raise ValueError(f"{item.get('item_id')} has no boolean passed decision")
        if annotation["passed"]:
            if not annotation.get("src_evidence_ref") or not annotation.get("dst_evidence_ref"):
                raise ValueError(f"{item.get('item_id')} passed without endpoint evidence")
            matched = annotation.get("matched_attributes")
            if not isinstance(matched, list) or not any(str(value).strip() for value in matched):
                raise ValueError(f"{item.get('item_id')} passed without matched attributes")
            if annotation.get("conflicts"):
                raise ValueError(f"{item.get('item_id')} passed with conflicts")
        decisions[str(item["edge_id"])] = {
            **annotation,
            "annotator": annotator,
            "labels_source": labels_source,
        }
    output = copy.deepcopy(graph)
    found: set[str] = set()
    for edge in output.get("edges") or []:
        edge_id = str(edge.get("edge_id") or "")
        if edge_id not in decisions:
            continue
        found.add(edge_id)
        payload = dict(edge.get("payload") or {})
        payload["targeted_reread"] = decisions[edge_id]
        edge["payload"] = payload
        edge["targeted_reread"] = decisions[edge_id]
    missing = set(decisions) - found
    if missing:
        raise ValueError(f"identity reread references unknown edges: {sorted(missing)}")
    return output


def _node_context(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "mention_id": node.get("mention_id"),
        "entity_type": node.get("entity_type"),
        "attributes": node.get("attributes") or {},
        "clip_id": node.get("clip_id"),
        "time_span": node.get("time_span"),
        "text": node.get("text"),
        "evidence_refs": node.get("evidence_refs") or [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--graph", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--graph", type=Path, required=True)
    apply.add_argument("--packet", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    if args.command == "prepare":
        payload = prepare_identity_reread_packet(graph)
    else:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        payload = apply_identity_reread_packet(graph, packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(payload.get("items") or []), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
