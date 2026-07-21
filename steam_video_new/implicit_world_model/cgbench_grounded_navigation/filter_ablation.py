"""Filter matched-ablation cases using frozen blinded audit decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from .builder import _write_json


def filter_ablation_cases(
    ablation: dict[str, Any], review_packet: dict[str, Any],
    audit: dict[str, Any], hidden_review_key: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = {row["review_id"]: row for row in audit.get("decisions") or []}
    roles = {row["review_id"]: row["gt_interval_role"]
             for row in hidden_review_key.get("items") or []}
    items_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in review_packet.get("items") or []:
        items_by_case[str(item["case_id"])].append(item)
    accepted: set[str] = set()
    exclusions: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for case_id, items in sorted(items_by_case.items()):
        reasons: set[str] = set()
        for item in items:
            review_id = str(item["review_id"])
            decision = decisions.get(review_id)
            if decision is None or decision.get("request_status") != "completed":
                reasons.add("missing_or_failed_model_review")
                continue
            if decision.get("media_alignment") != "accept":
                reasons.add("media_alignment_not_accepted")
            if decision.get("descriptor_grounding") != "accept":
                reasons.add("descriptor_grounding_not_accepted")
            relevance = decision.get("question_relevance")
            role = roles.get(review_id)
            if role == "clue" and relevance != "relevant":
                reasons.add("clue_not_confirmed_relevant")
            elif role == "control" and relevance != "unrelated":
                reasons.add("control_contaminated_or_inconclusive")
            elif role not in {"clue", "control"}:
                reasons.add("hidden_role_missing")
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
            exclusions.append({"case_id": case_id, "reasons": sorted(reasons)})
        else:
            accepted.add(case_id)
    result = deepcopy(ablation)
    result["cases"] = [case for case in result.get("cases") or [] if case.get("case_id") in accepted]
    result["annotation_status"] = "gpt56_filtered_provisional"
    result["training_ready"] = False
    result["formal_eligible"] = False
    report = {
        "schema_version": "steam-cgbench-provisional-ablation-filter-report/v0.1",
        "labels_source": "model_provisional", "input_review_case_count": len(items_by_case),
        "accepted_case_count": len(accepted), "excluded_case_count": len(exclusions),
        "exclusion_reason_case_counts": dict(sorted(reason_counts.items())),
        "exclusions": exclusions, "formal_eligible": False,
        "formal_blocker": "independent human review must confirm every retained case",
    }
    return result, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--review-packet", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--hidden-review-key", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in
                (args.ablation, args.review_packet, args.audit, args.hidden_review_key)]
    result, report = filter_ablation_cases(*payloads)
    _write_json(args.output, result); _write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
