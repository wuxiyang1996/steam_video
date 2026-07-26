"""Apply auditable categorical AI review to a draft navigation case set."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .preference_data import lock_navigation_case_set, require_valid_navigation_case_set


ALLOWED_CASE_JUDGMENTS = {"supported", "rejected", "inconclusive"}


def apply_provisional_case_review(
    draft: dict[str, Any],
    review: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep only supported cases and lock them as explicitly AI-provisional."""

    require_valid_navigation_case_set(draft)
    if draft.get("annotation_status") != "draft":
        raise ValueError("provisional review input must be a draft case set")
    if review.get("labels_source") != "model_provisional":
        raise ValueError("case review labels_source must be model_provisional")
    annotator = str(review.get("annotator") or "").strip()
    if not annotator:
        raise ValueError("case review requires an annotator")
    decisions: dict[str, dict[str, Any]] = {}
    for row in review.get("decisions") or []:
        case_id = str(row.get("case_id") or "")
        judgment = str(row.get("judgment") or "")
        rationale = str(row.get("rationale") or "").strip()
        evidence_refs = tuple(str(value) for value in row.get("evidence_refs") or [])
        if not case_id or case_id in decisions:
            raise ValueError(f"duplicate or empty reviewed case_id: {case_id!r}")
        if judgment not in ALLOWED_CASE_JUDGMENTS:
            raise ValueError(f"invalid case judgment for {case_id}: {judgment!r}")
        if not rationale or not evidence_refs:
            raise ValueError(f"review {case_id} requires rationale and evidence_refs")
        decisions[case_id] = row
    case_ids = {str(case["case_id"]) for case in draft["cases"]}
    if set(decisions) != case_ids:
        missing = sorted(case_ids - set(decisions))
        unknown = sorted(set(decisions) - case_ids)
        raise ValueError(f"case review coverage mismatch; missing={missing}, unknown={unknown}")

    kept: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for case in draft["cases"]:
        decision = decisions[str(case["case_id"])]
        judgment = str(decision["judgment"])
        counts[judgment] += 1
        if judgment != "supported":
            continue
        expected = {str(value) for value in case["gold_event_ids"]}
        cited = {str(value) for value in decision["evidence_refs"]}
        if not expected <= cited:
            raise ValueError(f"supported review does not cite all gold events: {case['case_id']}")
        reviewed = json.loads(json.dumps(case))
        reviewed["notes"] = (
            f"GPT-5.6 model-provisional categorical review: "
            f"{decision['rationale']} Formal human review remains required."
        )
        kept.append(reviewed)
    if not kept:
        raise ValueError("provisional review rejected every case")
    reviewed_set = json.loads(json.dumps(draft))
    reviewed_set["cases"] = kept
    locked = lock_navigation_case_set(
        reviewed_set,
        annotation_status="ai_provisional",
        annotator=annotator,
    )
    report = {
        "schema_version": "steam-navigation-case-review-report/v0.1",
        "case_set_id": locked["case_set_id"],
        "labels_source": "model_provisional",
        "annotator": annotator,
        "input_case_count": len(draft["cases"]),
        "locked_case_count": len(locked["cases"]),
        "judgment_counts": dict(sorted(counts.items())),
        "locked_sha256": locked["locked_sha256"],
        "formal_gate_eligible": False,
        "formal_blocker": "independent human review is still required",
    }
    return locked, report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    locked, report = apply_provisional_case_review(draft, review)
    for path, payload in ((args.output, locked), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
