"""Fault localization and repair for L1/L2 graph construction.

When reasoning or perception skills fail, this module:
1. Diagnoses which skill failed and why (fault localization)
2. Generates a targeted repair plan (repair strategy)
3. Re-executes the repaired sub-plan (local repair)

This is a Stage 0.5 implementation: still open-loop per repair attempt,
but allows one retry cycle before falling back to deterministic.

The full Stage 1 closed-loop MDP controller will handle iterative repair
with action masks and budget constraints.
"""

from __future__ import annotations

import json
from typing import Any

from atomic_skills.common import make_result


# --- Fault Localization ---

class FaultType:
    MISSING_EVIDENCE = "missing_evidence"
    WRONG_RETRIEVAL = "wrong_retrieval"
    INFERENCE_FAILURE = "inference_failure"
    VERIFICATION_FAILURE = "verification_failure"
    PERCEPTION_FAILURE = "perception_failure"
    ARGUMENT_ERROR = "argument_error"
    PLAN_STRUCTURE_ERROR = "plan_structure_error"


_FAILURE_CODE_TO_FAULT: dict[str, str] = {
    "no_event_match": FaultType.WRONG_RETRIEVAL,
    "no_entity_match": FaultType.WRONG_RETRIEVAL,
    "no_evidence_match": FaultType.MISSING_EVIDENCE,
    "empty_observation": FaultType.PERCEPTION_FAILURE,
    "empty_dialogue": FaultType.PERCEPTION_FAILURE,
    "no_entity_mentions": FaultType.PERCEPTION_FAILURE,
    "insufficient_evidence": FaultType.VERIFICATION_FAILURE,
    "low_confidence": FaultType.INFERENCE_FAILURE,
    "invalid_skill_args": FaultType.ARGUMENT_ERROR,
    "unknown_skill_id": FaultType.PLAN_STRUCTURE_ERROR,
    "llm_backend_error": FaultType.INFERENCE_FAILURE,
    "vlm_backend_error": FaultType.PERCEPTION_FAILURE,
}

_RETRIEVAL_SKILLS = frozenset({
    "retrieve_by_event", "retrieve_by_entity", "retrieve_by_time",
    "retrieve_by_relation", "retrieve_evidence_for_hypothesis",
    "search_counterevidence", "localize_clue",
})

_INFERENCE_SKILLS = frozenset({
    "infer_causal_relation", "infer_temporal_relation", "infer_state_change",
    "infer_intention_or_motive", "infer_social_contradiction",
})

_PERCEPTION_SKILLS = frozenset({
    "extract_observation", "extract_dialogue_span", "detect_entity_mention",
})

_VERIFICATION_SKILLS = frozenset({
    "verify_claim_support", "verify_temporal_social_consistency",
    "score_hypothesis_support", "compare_hypotheses",
})


def localize_faults(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyze execution trace and identify faults with root causes.

    Returns a list of fault records:
    [
        {
            "step_id": "r3",
            "skill_id": "retrieve_by_event",
            "fault_type": "wrong_retrieval",
            "failure_code": "no_event_match",
            "is_root_cause": True,
            "downstream_affected": ["r4", "r5"],
            "repair_strategy": "broaden_retrieval",
        }
    ]
    """
    faults: list[dict[str, Any]] = []
    failed_step_ids: set[str] = set()

    for step in trace:
        if step.get("ok"):
            continue

        step_id = step.get("step_id", "")
        skill_id = step.get("skill_id", "")
        failure_code = step.get("failure_code") or "unknown"

        fault_type = _FAILURE_CODE_TO_FAULT.get(failure_code)
        if not fault_type:
            if skill_id in _RETRIEVAL_SKILLS:
                fault_type = FaultType.WRONG_RETRIEVAL
            elif skill_id in _INFERENCE_SKILLS:
                fault_type = FaultType.INFERENCE_FAILURE
            elif skill_id in _PERCEPTION_SKILLS:
                fault_type = FaultType.PERCEPTION_FAILURE
            elif skill_id in _VERIFICATION_SKILLS:
                fault_type = FaultType.VERIFICATION_FAILURE
            else:
                fault_type = FaultType.ARGUMENT_ERROR

        failed_step_ids.add(step_id)
        faults.append({
            "step_id": step_id,
            "skill_id": skill_id,
            "fault_type": fault_type,
            "failure_code": failure_code,
            "messages": step.get("messages", []),
        })

    # Determine root causes: a fault is root if no prior fault feeds into it
    step_order = {step.get("step_id"): i for i, step in enumerate(trace)}
    for fault in faults:
        fault_idx = step_order.get(fault["step_id"], 999)
        prior_faults = [f for f in faults if step_order.get(f["step_id"], 999) < fault_idx]
        fault["is_root_cause"] = len(prior_faults) == 0 or fault["fault_type"] in (
            FaultType.PERCEPTION_FAILURE, FaultType.PLAN_STRUCTURE_ERROR
        )

    # Tag downstream affected steps
    for fault in faults:
        if fault["is_root_cause"]:
            fault_idx = step_order.get(fault["step_id"], 999)
            downstream = [
                f["step_id"] for f in faults
                if step_order.get(f["step_id"], 999) > fault_idx and not f["is_root_cause"]
            ]
            fault["downstream_affected"] = downstream
        else:
            fault["downstream_affected"] = []

    # Assign repair strategies
    for fault in faults:
        fault["repair_strategy"] = _select_repair_strategy(fault)

    return faults


def _select_repair_strategy(fault: dict[str, Any]) -> str:
    """Select a repair action based on fault type."""
    ft = fault["fault_type"]
    if ft == FaultType.WRONG_RETRIEVAL:
        return "broaden_retrieval"
    elif ft == FaultType.MISSING_EVIDENCE:
        return "alternative_retrieval_path"
    elif ft == FaultType.INFERENCE_FAILURE:
        return "retry_with_more_context"
    elif ft == FaultType.VERIFICATION_FAILURE:
        return "gather_additional_evidence"
    elif ft == FaultType.PERCEPTION_FAILURE:
        return "retry_adjacent_clip"
    elif ft == FaultType.ARGUMENT_ERROR:
        return "fix_arguments"
    elif ft == FaultType.PLAN_STRUCTURE_ERROR:
        return "skip_or_replace_skill"
    return "no_repair"


# --- Repair Plan Generation ---

_REPAIR_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "broaden_retrieval": [
        {"skill_id": "retrieve_by_time", "args_override": {"window_before": 60, "window_after": 60}},
        {"skill_id": "retrieve_by_relation", "args_override": {"hop_limit": 2}},
    ],
    "alternative_retrieval_path": [
        {"skill_id": "retrieve_by_entity", "args_override": {}},
        {"skill_id": "retrieve_by_relation", "args_override": {"relation_type": "temporal_next"}},
    ],
    "retry_with_more_context": [
        {"skill_id": "__retry_same__", "args_override": {"retry": True}},
    ],
    "gather_additional_evidence": [
        {"skill_id": "retrieve_by_event", "args_override": {}},
        {"skill_id": "localize_clue", "args_override": {}},
    ],
    "retry_adjacent_clip": [
        {"skill_id": "__retry_same__", "args_override": {"offset_clip": True}},
    ],
    "fix_arguments": [
        {"skill_id": "__retry_same__", "args_override": {}},
    ],
    "skip_or_replace_skill": [],
}


def generate_repair_plan(
    faults: list[dict[str, Any]],
    original_plan: list[dict[str, Any]],
    clue_memory_graph: dict[str, Any],
    question: dict[str, Any],
    *,
    max_repair_steps: int = 4,
) -> list[dict[str, Any]]:
    """Generate a repair sub-plan targeting root-cause faults.

    Returns a list of repair steps that can be appended to or replace
    failed steps in the original plan.
    """
    root_faults = [f for f in faults if f.get("is_root_cause")]
    if not root_faults:
        return []

    repair_steps: list[dict[str, Any]] = []
    original_by_id = {s.get("step_id"): s for s in original_plan}

    for fault in root_faults[:max_repair_steps]:
        strategy = fault.get("repair_strategy", "no_repair")
        templates = _REPAIR_TEMPLATES.get(strategy, [])

        original_step = original_by_id.get(fault["step_id"]) or {}
        original_args = dict(original_step.get("args") or {})

        for i, tmpl in enumerate(templates):
            skill_id = tmpl["skill_id"]
            if skill_id == "__retry_same__":
                skill_id = fault["skill_id"]

            repair_args = {**original_args, **tmpl.get("args_override", {})}

            # Broaden retrieval: use question text as fallback query
            if strategy == "broaden_retrieval" and not repair_args.get("event_description"):
                repair_args["event_description"] = question.get("question_text", "")

            repair_steps.append({
                "step_id": f"repair_{fault['step_id']}_{i}",
                "skill_id": skill_id,
                "args": repair_args,
                "depends_on": [fault["step_id"]],
                "repair_for": fault["step_id"],
                "repair_strategy": strategy,
            })

    return repair_steps[:max_repair_steps]


# --- Repair Execution ---

def execute_repair(
    repair_plan: list[dict[str, Any]],
    original_trace: list[dict[str, Any]],
    step_outputs: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    question: dict[str, Any],
    *,
    skill_executor: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Execute a repair plan and merge results.

    Returns:
        (repair_trace, updated_step_outputs, repair_summary)
    """
    from .reasoning_planner import execute_reasoning_plan

    repair_trace, repair_outputs = execute_reasoning_plan(
        reasoning_plan=repair_plan,
        clue_memory_graph=clue_memory_graph,
        question=question,
        skill_executor=skill_executor,
    )

    merged_outputs = {**step_outputs, **repair_outputs}

    repair_summary = []
    for step in repair_trace:
        target = next(
            (s.get("repair_for") for s in repair_plan if s.get("step_id") == step.get("step_id")),
            None,
        )
        repair_summary.append({
            "repair_step_id": step.get("step_id"),
            "target_step_id": target,
            "skill_id": step.get("skill_id"),
            "ok": step.get("ok"),
            "repaired": step.get("ok", False),
        })

    return repair_trace, merged_outputs, repair_summary


# --- Full Repair Loop (wraps fault localize + repair + merge) ---

def attempt_repair(
    trace: list[dict[str, Any]],
    step_outputs: dict[str, Any],
    original_plan: list[dict[str, Any]],
    clue_memory_graph: dict[str, Any],
    question: dict[str, Any],
    *,
    skill_executor: Any | None = None,
    max_repair_attempts: int = 1,
) -> dict[str, Any]:
    """Full repair loop: localize faults → generate repair plan → execute.

    Returns a structured repair result:
    {
        "attempted": bool,
        "faults": [...],
        "repair_plan": [...],
        "repair_trace": [...],
        "repair_summary": [...],
        "repaired_count": int,
        "still_failed_count": int,
    }
    """
    faults = localize_faults(trace)
    root_faults = [f for f in faults if f.get("is_root_cause")]

    if not root_faults:
        return {
            "attempted": False,
            "faults": faults,
            "repair_plan": [],
            "repair_trace": [],
            "repair_summary": [],
            "repaired_count": 0,
            "still_failed_count": 0,
        }

    repair_plan = generate_repair_plan(
        faults, original_plan, clue_memory_graph, question,
    )

    if not repair_plan:
        return {
            "attempted": False,
            "faults": faults,
            "repair_plan": [],
            "repair_trace": [],
            "repair_summary": [],
            "repaired_count": 0,
            "still_failed_count": len(root_faults),
        }

    repair_trace, merged_outputs, repair_summary = execute_repair(
        repair_plan, trace, step_outputs, clue_memory_graph, question,
        skill_executor=skill_executor,
    )

    repaired = sum(1 for s in repair_summary if s.get("repaired"))
    still_failed = sum(1 for s in repair_summary if not s.get("repaired"))

    return {
        "attempted": True,
        "faults": faults,
        "repair_plan": repair_plan,
        "repair_trace": repair_trace,
        "repair_summary": repair_summary,
        "repaired_count": repaired,
        "still_failed_count": still_failed,
    }
