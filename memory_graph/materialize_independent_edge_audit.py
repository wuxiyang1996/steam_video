"""Join locked human labels with the hidden model key after annotation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


VALID_JUDGMENTS = {"supported", "unsupported", "contradicted", "unclear"}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def materialize(
    packet: dict[str, Any],
    model_key: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    annotator = str(packet.get("annotator") or "").strip()
    protocol = str(packet.get("protocol_version") or "").strip()
    if not annotator or not protocol:
        raise ValueError("locked packet requires annotator and protocol_version")
    labels: dict[str, dict[str, Any]] = {}
    for item in packet.get("items") or []:
        if not isinstance(item, dict):
            continue
        annotation = item.get("annotation") or {}
        judgment = str(annotation.get("judgment") or "").lower()
        if judgment not in VALID_JUDGMENTS:
            raise ValueError(f"{item.get('item_id')} has invalid or missing judgment")
        labels[str(item.get("item_id"))] = annotation

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key_item in model_key.get("items") or []:
        if not isinstance(key_item, dict):
            continue
        item_id = str(key_item.get("item_id"))
        if item_id not in labels:
            raise ValueError(f"model key item {item_id} has no locked human label")
        annotation = labels[item_id]
        grouped[str(key_item.get("video_id"))].append(
            {
                "item_id": item_id,
                "src": key_item.get("src"),
                "dst": key_item.get("dst"),
                "relation": key_item.get("relation"),
                "judgment": annotation.get("judgment"),
                "reason": annotation.get("reason"),
                "minimal_support_sufficient": annotation.get(
                    "minimal_support_sufficient"
                ),
                "post_verifier": key_item.get("post_verifier"),
                "probability": key_item.get("probability"),
            }
        )
    return dict(grouped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-packet", required=True, type=Path)
    parser.add_argument("--model-key", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    packet = _load(args.annotation_packet)
    grouped = materialize(packet, _load(args.model_key))
    for video_id, rows in grouped.items():
        video_dir = args.run_dir / video_id
        if not video_dir.is_dir():
            raise ValueError(f"missing run directory for {video_id}: {video_dir}")
        result = {
            "schema_version": "steam-independent-edge-audit/v0.1",
            "labels_source": "independent_human",
            "annotator": packet["annotator"],
            "protocol_version": packet["protocol_version"],
            "candidate_edge_audit": rows,
            "computed_summary": {"audit_complete": True},
        }
        (video_dir / "independent_audit.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"video_count": len(grouped), "item_count": sum(map(len, grouped.values()))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
