"""Categorical prediction confusion reports against provisional or human audits."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from .l1_relation_audit import IDENTITY_RELATIONS, JUDGMENTS


OUTCOMES = ("supports", "rejects", "inconclusive")
_REFERENCE = {
    "supported": "supports",
    "unsupported": "rejects",
    "contradicted": "rejects",
    "unclear": "inconclusive",
}


def admitted_candidate_predictions(packet: dict[str, Any]) -> dict[str, Any]:
    """Represent admission of every audited candidate without inventing scores."""

    return {
        "schema_version": "steam-categorical-relation-predictions/v0.1",
        "predictor": "native_l1_candidate_admission",
        "semantics": (
            "Every packet row existed in the admitted candidate set, so the baseline "
            "categorically predicts supports. This is not a post-read verifier."
        ),
        "predictions": [
            {
                "item_id": str(item["item_id"]),
                "outcome": "supports",
                "reason": "relation was present in the admitted native L1 candidate set",
            }
            for item in packet.get("items") or []
        ],
    }


def categorical_confusion_report(
    packet: dict[str, Any],
    predictions: dict[str, Any],
) -> dict[str, Any]:
    """Compute a three-way matrix; neither labels nor predictions contain scores."""

    labels_source = str(packet.get("labels_source") or "independent_human")
    by_item: dict[str, str] = {}
    for row in predictions.get("predictions") or []:
        item_id = str(row.get("item_id") or "")
        outcome = str(row.get("outcome") or "")
        if not item_id or item_id in by_item:
            raise ValueError(f"duplicate or empty prediction item_id: {item_id!r}")
        if outcome not in OUTCOMES:
            raise ValueError(f"invalid categorical outcome for {item_id}: {outcome!r}")
        by_item[item_id] = outcome
    items = list(packet.get("items") or [])
    item_ids = {str(item.get("item_id") or "") for item in items}
    if set(by_item) != item_ids:
        raise ValueError(
            "prediction coverage mismatch; "
            f"missing={sorted(item_ids - set(by_item))}, "
            f"unknown={sorted(set(by_item) - item_ids)}"
        )
    matrices: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for item in items:
        item_id = str(item["item_id"])
        judgment = str((item.get("annotation") or {}).get("judgment") or "")
        if judgment not in JUDGMENTS:
            raise ValueError(f"invalid or missing reference judgment: {item_id}")
        relation = str(item.get("relation") or "")
        group = (
            "identity"
            if relation in IDENTITY_RELATIONS
            else "state_transition"
            if relation == "state_transition"
            else "other"
        )
        pair = (_REFERENCE[judgment], by_item[item_id])
        matrices["all"][pair] += 1
        matrices[group][pair] += 1
        matrices[f"relation:{relation}"][pair] += 1
    return {
        "schema_version": "steam-categorical-relation-confusion/v0.1",
        "predictor": str(predictions.get("predictor") or "unknown"),
        "labels_source": labels_source,
        "formal_gate_eligible": labels_source == "independent_human",
        "outcomes": list(OUTCOMES),
        "groups": {
            group: _matrix_payload(matrix)
            for group, matrix in sorted(matrices.items())
        },
        "limitations": [
            "model_provisional references cannot satisfy a production gate"
            if labels_source != "independent_human"
            else "formal eligibility still depends on packet independence and lock audit",
            str(predictions.get("semantics") or ""),
        ],
    }


def _matrix_payload(counts: Counter[tuple[str, str]]) -> dict[str, Any]:
    total = sum(counts.values())
    correct = sum(counts[(outcome, outcome)] for outcome in OUTCOMES)
    return {
        "reference_by_prediction": {
            reference: {
                prediction: counts[(reference, prediction)]
                for prediction in OUTCOMES
            }
            for reference in OUTCOMES
        },
        "item_count": total,
        "categorical_accuracy": correct / total if total else None,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("admission-baseline")
    baseline.add_argument("--packet", required=True, type=Path)
    baseline.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--packet", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    payload = (
        admitted_candidate_predictions(packet)
        if args.command == "admission-baseline"
        else categorical_confusion_report(
            packet,
            json.loads(args.predictions.read_text(encoding="utf-8")),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
