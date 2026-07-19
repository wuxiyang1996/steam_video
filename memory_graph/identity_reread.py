"""Prepare and apply grounded raw-video identity reread decisions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .identity_verifier import TRUSTED_REREAD_SOURCES, verify_identity_candidates


def prepare_identity_reread_packet(
    graph: dict[str, Any],
    *,
    video_path: str | None = None,
) -> dict[str, Any]:
    nodes = {
        str(node["node_id"]): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and node.get("node_id")
    }
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    _, report = verify_identity_candidates(nodes, edges)
    edge_by_id = {
        str(edge.get("edge_id") or f"identity-edge:{index}"): edge
        for index, edge in enumerate(edges)
    }
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row in report.targeted_reread_queue:
        edge_id = str(row["edge_id"])
        edge = edge_by_id[edge_id]
        pair = tuple(sorted((str(edge["src"]), str(edge["dst"]))))
        grouped.setdefault(pair, []).append((row, edge))
    items: list[dict[str, Any]] = []
    for pair_rows in grouped.values():
        row, edge = pair_rows[0]
        src, dst = nodes[str(edge["src"])], nodes[str(edge["dst"])]
        edge_ids = [str(value[0]["edge_id"]) for value in pair_rows]
        relations = sorted({str(value[1].get("edge_type") or "") for value in pair_rows})
        items.append(
            {
                "item_id": f"identity-reread:{len(items) + 1:04d}",
                "edge_id": edge_ids[0],
                "edge_ids": edge_ids,
                "relation": relations[0] if len(relations) == 1 else "identity",
                "relations": relations,
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
        "graph_id": graph.get("graph_id"),
        "example_id": graph.get("example_id"),
        "video_id": graph.get("video_id"),
        "source_video_path": video_path,
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
        edge_ids = item.get("edge_ids") or [item["edge_id"]]
        if not isinstance(edge_ids, list) or not edge_ids:
            raise ValueError(f"{item.get('item_id')} has no edge IDs")
        for edge_id in edge_ids:
            normalized_edge_id = str(edge_id)
            if normalized_edge_id in decisions:
                raise ValueError(f"duplicate identity reread edge: {normalized_edge_id}")
            decisions[normalized_edge_id] = {
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


def apply_identity_reread_to_artifact(
    artifact: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Apply decisions to either a graph or a canonical Video_Skills artifact."""

    graph = _graph_from_artifact(artifact)
    applied = apply_identity_reread_packet(graph, packet)
    if graph is artifact:
        return applied
    output = copy.deepcopy(artifact)
    output["metadata"]["clue_memory_graph"] = applied
    return output


def _graph_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    if isinstance(artifact.get("nodes"), list) and isinstance(
        artifact.get("edges"), list
    ):
        return artifact
    metadata = artifact.get("metadata")
    nested = metadata.get("clue_memory_graph") if isinstance(metadata, dict) else None
    if isinstance(nested, dict) and isinstance(nested.get("nodes"), list) and isinstance(
        nested.get("edges"), list
    ):
        return nested
    raise ValueError("input contains neither a clue-memory graph nor a canonical nested graph")


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
    prepare.add_argument("--video-path", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--graph", type=Path, required=True)
    apply.add_argument("--packet", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = json.loads(args.graph.read_text(encoding="utf-8"))
    graph = _graph_from_artifact(artifact)
    if args.command == "prepare":
        payload = prepare_identity_reread_packet(
            graph,
            video_path=str(args.video_path.resolve()) if args.video_path else None,
        )
    else:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        payload = apply_identity_reread_to_artifact(artifact, packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(payload.get("items") or []), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
