#!/usr/bin/env python3
"""Smoke test: fault localization and repair loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from atomic_skills.common import SCHEMA_VERSION
from dataset_clip_wrapper.fault_repair import (
    FaultType,
    localize_faults,
    generate_repair_plan,
    attempt_repair,
)


def _sample_graph():
    return {
        "schema_version": SCHEMA_VERSION,
        "nodes": [
            {"node_id": "event:e1", "node_type": "event", "text": "A man leaves the fence.", "time_span": {"start_s": 10, "end_s": 12}},
            {"node_id": "event:e2", "node_type": "event", "text": "The man returns.", "time_span": {"start_s": 120, "end_s": 125}},
            {"node_id": "state:s1", "node_type": "state", "text": "Iron fence original position.", "time_span": {"start_s": 10, "end_s": 125}},
        ],
        "edges": [
            {"edge_id": "edge:1", "src": "event:e1", "dst": "state:s1", "edge_type": "same_location"},
        ],
    }


def test_fault_localization():
    """Fault localization correctly identifies root causes."""
    trace = [
        {"step_id": "r1", "skill_id": "parse_question_target", "ok": True},
        {"step_id": "r2", "skill_id": "retrieve_by_event", "ok": False, "failure_code": "no_event_match"},
        {"step_id": "r3", "skill_id": "localize_clue", "ok": False, "failure_code": "no_evidence_match"},
        {"step_id": "r4", "skill_id": "infer_causal_relation", "ok": False, "failure_code": "low_confidence"},
        {"step_id": "r5", "skill_id": "verify_claim_support", "ok": False, "failure_code": "insufficient_evidence"},
    ]

    faults = localize_faults(trace)
    assert len(faults) == 4, f"Expected 4 faults, got {len(faults)}"

    root_faults = [f for f in faults if f["is_root_cause"]]
    assert len(root_faults) >= 1, "Should have at least 1 root cause"
    assert root_faults[0]["step_id"] == "r2", f"Root cause should be r2, got {root_faults[0]['step_id']}"
    assert root_faults[0]["fault_type"] == FaultType.WRONG_RETRIEVAL
    assert root_faults[0]["repair_strategy"] == "broaden_retrieval"

    print(f"  fault localization: {len(faults)} faults, {len(root_faults)} root causes")
    print(f"  root cause: {root_faults[0]['skill_id']} → {root_faults[0]['repair_strategy']}")
    print("PASS: fault localization")


def test_repair_plan_generation():
    """Generate repair plan from faults."""
    trace = [
        {"step_id": "r1", "skill_id": "parse_question_target", "ok": True},
        {"step_id": "r2", "skill_id": "retrieve_by_event", "ok": False, "failure_code": "no_event_match"},
        {"step_id": "r3", "skill_id": "extract_observation", "ok": False, "failure_code": "empty_observation"},
    ]

    original_plan = [
        {"step_id": "r1", "skill_id": "parse_question_target", "args": {"question_text": "Why?"}, "depends_on": []},
        {"step_id": "r2", "skill_id": "retrieve_by_event", "args": {"event_description": "man leaves"}, "depends_on": ["r1"]},
        {"step_id": "r3", "skill_id": "extract_observation", "args": {"clip_or_text_ref": "clip:1", "modality": "visual", "text": ""}, "depends_on": ["r2"]},
    ]

    faults = localize_faults(trace)
    repair_plan = generate_repair_plan(
        faults, original_plan, _sample_graph(),
        {"question_text": "Why did the man return?", "options": []},
    )

    assert len(repair_plan) > 0, "Should generate at least one repair step"
    assert any(s["repair_for"] == "r2" for s in repair_plan), "Should target root cause r2"

    for step in repair_plan:
        assert step.get("repair_strategy"), "Each repair step must have a strategy"
        assert step.get("skill_id"), "Each repair step must have a skill_id"

    print(f"  repair plan: {len(repair_plan)} steps")
    for s in repair_plan:
        print(f"    {s['step_id']}: {s['skill_id']} (for {s['repair_for']}, strategy={s['repair_strategy']})")
    print("PASS: repair plan generation")


def test_full_repair_loop():
    """Full repair attempt with rule-based execution."""
    trace = [
        {"step_id": "r1", "skill_id": "parse_question_target", "ok": True},
        {"step_id": "r2", "skill_id": "retrieve_by_event", "ok": False, "failure_code": "no_event_match"},
    ]
    step_outputs = {"r1": {"target_entities": ["man"], "evidence_refs": []}}

    original_plan = [
        {"step_id": "r1", "skill_id": "parse_question_target", "args": {"question_text": "Where is the man?"}, "depends_on": []},
        {"step_id": "r2", "skill_id": "retrieve_by_event", "args": {"event_description": "xyz_nonexistent"}, "depends_on": ["r1"]},
    ]

    result = attempt_repair(
        trace, step_outputs, original_plan,
        clue_memory_graph=_sample_graph(),
        question={"question_text": "Where is the man?", "options": []},
    )

    assert result["attempted"], "Should have attempted repair"
    assert result["faults"], "Should have identified faults"
    assert result["repair_plan"], "Should have generated repair plan"
    assert result["repair_trace"], "Should have executed repair"

    print(f"  full repair: attempted={result['attempted']}, "
          f"repaired={result['repaired_count']}, "
          f"still_failed={result['still_failed_count']}")
    print(f"  fault types: {[f['fault_type'] for f in result['faults']]}")
    print("PASS: full repair loop")


def test_no_repair_when_all_pass():
    """No repair attempted when trace is fully successful."""
    trace = [
        {"step_id": "r1", "skill_id": "parse_question_target", "ok": True},
        {"step_id": "r2", "skill_id": "retrieve_by_event", "ok": True},
    ]

    result = attempt_repair(
        trace, {}, [], _sample_graph(),
        {"question_text": "test", "options": []},
    )

    assert not result["attempted"], "Should not attempt repair when all pass"
    assert result["repaired_count"] == 0
    print("PASS: no repair when all pass")


def main() -> int:
    errors = []
    for name, fn in [
        ("fault_localization", test_fault_localization),
        ("repair_plan_generation", test_repair_plan_generation),
        ("full_repair_loop", test_full_repair_loop),
        ("no_repair_when_all_pass", test_no_repair_when_all_pass),
    ]:
        try:
            fn()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            import traceback
            traceback.print_exc()

    if errors:
        print(f"\nFAILED ({len(errors)} tests):")
        for e in errors:
            print(f"  - {e}")
        return 2
    print(f"\nAll tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
