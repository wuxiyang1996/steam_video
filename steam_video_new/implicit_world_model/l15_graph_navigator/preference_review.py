"""Apply externally produced categorical decisions to a blinded preference packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .preference_data import ALLOWED_PREFERENCE_LABELS, lock_annotation_packet


def apply_preference_review(
    packet: dict[str, Any],
    review: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if packet.get("annotation_status") != "unlabeled":
        raise ValueError("preference input must be an unlabeled packet")
    if review.get("labels_source") != "model_provisional":
        raise ValueError("preference review labels_source must be model_provisional")
    annotator = str(review.get("annotator") or "").strip()
    if not annotator:
        raise ValueError("preference review requires an annotator")
    decisions: dict[str, dict[str, Any]] = {}
    for row in review.get("decisions") or []:
        comparison_id = str(row.get("comparison_id") or "")
        label = str(row.get("label") or "")
        rationale = str(row.get("rationale") or "").strip()
        if not comparison_id or comparison_id in decisions:
            raise ValueError(f"duplicate or empty comparison_id: {comparison_id!r}")
        if label not in ALLOWED_PREFERENCE_LABELS or not rationale:
            raise ValueError(f"invalid preference decision: {comparison_id}")
        decisions[comparison_id] = row
    expected = {str(row["comparison_id"]) for row in packet.get("comparisons") or []}
    if set(decisions) != expected:
        raise ValueError(
            "preference review coverage mismatch; "
            f"missing={sorted(expected - set(decisions))}, "
            f"unknown={sorted(set(decisions) - expected)}"
        )
    filled = json.loads(json.dumps(packet))
    for comparison in filled["comparisons"]:
        decision = decisions[str(comparison["comparison_id"])]
        comparison["label"] = decision["label"]
        comparison["rationale"] = decision["rationale"]
    locked = lock_annotation_packet(
        filled,
        annotation_status="ai_provisional",
        annotator=annotator,
    )
    label_counts = {
        label: sum(row["label"] == label for row in locked["comparisons"])
        for label in ALLOWED_PREFERENCE_LABELS
    }
    return locked, {
        "schema_version": "steam-preference-review-report/v0.1",
        "packet_id": locked["packet_id"],
        "labels_source": "model_provisional",
        "annotator": annotator,
        "comparison_count": len(locked["comparisons"]),
        "label_counts": label_counts,
        "locked_sha256": locked["locked_sha256"],
        "numeric_reward_present": False,
        "formal_gate_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    locked, report = apply_preference_review(
        json.loads(args.packet.read_text(encoding="utf-8")),
        json.loads(args.review.read_text(encoding="utf-8")),
    )
    for path, payload in ((args.output, locked), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
