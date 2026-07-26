#!/usr/bin/env python3
"""Classify L1/L2/repair failures in final acceptance reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACCEPTED_STATUSES = {"accepted_strong", "accepted_bridge"}
LONG_DATASETS = {"cg_bench", "vrbench"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "missing")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _contains_any(values: list[str], needles: tuple[str, ...]) -> bool:
    text = " ".join(str(value).lower() for value in values)
    return any(needle in text for needle in needles)


def _missing_requirements(row: dict[str, Any]) -> list[str]:
    hints = row.get("repair_hints") or {}
    requirements = [str(item) for item in hints.get("missing_requirements") or [] if item]
    commonsense = hints.get("commonsense_repair") or {}
    requirements.extend(str(item) for item in commonsense.get("missing_requirements") or [] if item)
    repair = row.get("repair_report") or {}
    requirements.extend(str(item) for item in repair.get("gap_types") or [] if item)
    return list(dict.fromkeys(requirements))


def _selector_status(row: dict[str, Any]) -> str:
    repair = row.get("repair_report") or {}
    if repair.get("selector_status"):
        return str(repair.get("selector_status"))
    selector = repair.get("option_evidence_selector") or {}
    return str(selector.get("selector_status") or selector.get("status") or "")


def _final_status(row: dict[str, Any]) -> str:
    return str(row.get("final_acceptance_status") or ((row.get("L2_status") or {}).get("acceptance_status")) or "missing")


def _repair_status(row: dict[str, Any]) -> str:
    repair = row.get("repair_report") or {}
    return str(repair.get("repair_status") or ((row.get("L2_status") or {}).get("repair_status")) or "")


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset = str(row.get("dataset") or "unknown")
    final_status = _final_status(row)
    l1 = row.get("L1_quality") or {}
    l2 = row.get("L2_status") or {}
    hints = row.get("repair_hints") or {}
    commonsense = hints.get("commonsense_repair") or {}
    commonsense_present = bool(commonsense.get("present"))
    strict = row.get("strict_vlm_perception") or {}
    requirements = _missing_requirements(row)
    failure_reasons = [str(item) for item in hints.get("failure_reasons") or [] if item]
    verifier_reason = str(row.get("verifier_reason") or "")
    support_ref_count = int(l2.get("support_ref_count") or 0)
    repair_applied = bool(row.get("final_repair_applied"))
    repair_needed = bool(row.get("final_repair_needed") or row.get("repair_needed"))
    repair_status = _repair_status(row)
    selector_status = _selector_status(row)

    if final_status in ACCEPTED_STATUSES:
        failure_stage = "none"
        missing_evidence_type = "none"
        can_repair = False
        needs_dataset_replacement = False
        recommended_next_action = "accepted; no repair needed"
    elif not strict.get("qwen_only", True):
        failure_stage = "perception"
        missing_evidence_type = "strict_vlm_perception_gap"
        can_repair = True
        needs_dataset_replacement = False
        recommended_next_action = "retry failed or fallback clip schemas with the selected VLM backbone"
    elif l1.get("grade") != "high":
        failure_stage = "l1_graph"
        missing_evidence_type = "low_l1_semantic_density"
        can_repair = True
        needs_dataset_replacement = False
        recommended_next_action = "rerun L1 graph composition with denser clip schemas and neighbor graph compose"
    elif repair_applied and repair_status in {"needs_more_evidence", "rejected", "missing"}:
        if selector_status == "error":
            failure_stage = "repair_selector_error"
        elif selector_status in {"abstained", "no_positive_refs_selected", "not_run"}:
            failure_stage = "repair_selector"
        else:
            failure_stage = "repair_verifier"
        if commonsense_present:
            missing_evidence_type = "commonsense_bridge_without_discriminative_visual_anchor"
        elif dataset in LONG_DATASETS:
            missing_evidence_type = "long_video_retrieval_or_fine_evidence_gap"
        else:
            missing_evidence_type = "discriminative_visual_evidence_gap"
        can_repair = True
        needs_dataset_replacement = False
        recommended_next_action = "run another bounded repair round with targeted evidence selection and verifier-only commit"
    elif repair_needed:
        failure_stage = "repair_not_run"
        missing_evidence_type = "unrepaired_l2_failure"
        can_repair = True
        needs_dataset_replacement = False
        recommended_next_action = "send the sample through the repair protocol before judging benchmark fit"
    elif final_status in {"rejected", "missing"} or "no_final_answer" in failure_reasons:
        failure_stage = "l2_planner"
        missing_evidence_type = "reasoning_plan_no_supported_answer"
        can_repair = True
        needs_dataset_replacement = False
        recommended_next_action = "rerun L2 with gated evidence pack and require verifier support before commit"
    elif support_ref_count == 0 or "unsupported" in verifier_reason.lower():
        failure_stage = "verification"
        missing_evidence_type = "unsupported_answer_claim"
        can_repair = True
        needs_dataset_replacement = False
        recommended_next_action = "do not accept; repair or abstain until positive support refs exist"
    elif _contains_any(requirements, ("temporal", "before", "after", "sequence")):
        failure_stage = "retrieval"
        missing_evidence_type = "temporal_evidence_gap"
        can_repair = True
        needs_dataset_replacement = False
        recommended_next_action = "expand adjacent clips and add temporal_next/same_entity evidence"
    elif _contains_any(requirements, ("entity", "coref", "identity", "same")):
        failure_stage = "l1_graph"
        missing_evidence_type = "entity_coreference_gap"
        can_repair = True
        needs_dataset_replacement = False
        recommended_next_action = "repair entity linking across neighboring clips before L2 commit"
    else:
        failure_stage = "verification"
        missing_evidence_type = "insufficient_verified_evidence"
        can_repair = True
        needs_dataset_replacement = False
        recommended_next_action = "inspect evidence pack and run one bounded verifier-guided repair round"

    if (
        final_status not in ACCEPTED_STATUSES
        and dataset == "video_holmes"
        and commonsense_present
        and not requirements
        and support_ref_count == 0
    ):
        missing_evidence_type = "benchmark_requires_nonvisual_or_under_specified_bridge"
        needs_dataset_replacement = True
        recommended_next_action = "keep as abstain/diagnostic or replace with short benchmark whose answer is visually grounded"

    return {
        "dataset": dataset,
        "example_id": row.get("example_id"),
        "final_acceptance_status": final_status,
        "failure_stage": failure_stage,
        "missing_evidence_type": missing_evidence_type,
        "can_repair": can_repair,
        "needs_dataset_replacement": needs_dataset_replacement,
        "recommended_next_action": recommended_next_action,
        "repair_applied": repair_applied,
        "repair_needed_after_final": bool(row.get("final_repair_needed")),
        "l1_grade": l1.get("grade"),
        "support_ref_count": support_ref_count,
        "verifier_reason": verifier_reason,
        "failure_reasons": failure_reasons,
        "missing_requirements": requirements,
    }


def build_taxonomy(final_report: dict[str, Any]) -> dict[str, Any]:
    rows = [classify_row(row) for row in final_report.get("reports") or []]
    failures = [row for row in rows if row["final_acceptance_status"] not in ACCEPTED_STATUSES]
    summary = {
        "examples": len(rows),
        "accepted": len(rows) - len(failures),
        "failures": len(failures),
        "failure_stage_counts": _count(failures, "failure_stage"),
        "missing_evidence_type_counts": _count(failures, "missing_evidence_type"),
        "dataset_failure_counts": _count(failures, "dataset"),
        "repairable_failure_count": sum(1 for row in failures if row.get("can_repair")),
        "needs_dataset_replacement_count": sum(1 for row in failures if row.get("needs_dataset_replacement")),
    }
    return {"summary": summary, "reports": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify failure modes from a final acceptance report.")
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = build_taxonomy(_read_json(args.final_report))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
