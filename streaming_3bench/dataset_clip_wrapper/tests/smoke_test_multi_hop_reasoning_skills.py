#!/usr/bin/env python3
"""Smoke test option-level multi-hop/social reasoning atomic skills."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from atomic_skills.common import SCHEMA_VERSION
from atomic_skills.reasoning_graph_assembly import (
    bridge_evidence_hops,
    compare_hypotheses,
    generate_answer_hypotheses,
    retrieve_evidence_for_hypothesis,
    score_hypothesis_support,
    verify_claim_support,
    verify_temporal_social_consistency,
)
from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import REASONING_SKILL_IDS, execute_reasoning_plan
from atomic_skills.skill_backends import SkillBackendConfig, SkillBackendMode
from atomic_skills.skill_executor import SkillExecutor


class FakeVerifierLlmClient:
    model = "fake-gpt-oss-verifier"

    def reason(self, prompt: str, *, max_tokens: int = 512) -> dict[str, object]:
        supported = (
            ("Claim: 'White'" in prompt and "white RV vehicle" in prompt)
            or ("original position" in prompt and "iron fence" in prompt)
        )
        return {
            "supported": supported,
            "target_aligned": supported,
            "score": 0.95 if supported else 0.05,
            "reasoning": "fake verifier checks claim text and vehicle evidence",
        }


def _graph() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "nodes": [
            {
                "node_id": "event:start",
                "node_type": "event",
                "text": "The man leaves the iron fence and walks away.",
                "time_span": {"start_s": 10.0, "end_s": 12.0},
            },
            {
                "node_id": "event:return",
                "node_type": "event",
                "text": "The man walks back to his original position near the iron fence.",
                "time_span": {"start_s": 124.0, "end_s": 126.0},
            },
            {
                "node_id": "state:position",
                "node_type": "state",
                "text": "The iron fence marks the man's original position.",
                "time_span": {"start_s": 124.0, "end_s": 126.0},
            },
        ],
        "edges": [
            {"edge_id": "edge:1", "src": "event:start", "dst": "state:position", "edge_type": "same_location"},
            {"edge_id": "edge:2", "src": "state:position", "dst": "event:return", "edge_type": "temporal_next"},
        ],
    }


def main() -> int:
    graph = _graph()
    question = {
        "question_text": "What does the repeated iron fence imply about the man?",
        "options": [
            {"label": "A", "text": "The man entered a new building."},
            {"label": "E", "text": "The man walked back to his original position."},
        ],
    }

    hypotheses = generate_answer_hypotheses(question["question_text"], options=question["options"])
    best_option = hypotheses.outputs["hypotheses"][1]
    support = retrieve_evidence_for_hypothesis(graph, hypothesis=best_option)
    bridge = bridge_evidence_hops(
        graph,
        source_evidence=support.evidence_refs[:1],
        target_hypothesis=best_option,
        allowed_hop_types=["same_location", "temporal_next"],
    )
    scored = score_hypothesis_support(best_option, support_evidence=support.outputs)
    compared = compare_hypotheses([scored.outputs["scored_hypothesis"]])
    consistency = verify_temporal_social_consistency(bridge.outputs["multi_hop_chain"], hypothesis=best_option, evidence_graph=graph)
    color_graph = {
        "nodes": [
            {
                "node_id": "obs:white_rv",
                "node_type": "observation",
                "text": "A white RV vehicle is visible in the scene.",
            },
            {
                "node_id": "obs:white_bow",
                "node_type": "observation",
                "text": "A large white bow hangs above the door.",
            },
        ],
        "edges": [],
    }
    white_verify = verify_claim_support(
        {"claim_text": "White", "option_label": "E", "question_text": "What color is the vehicle?"},
        evidence_chain={"evidence_refs": ["obs:white_rv"]},
        evidence_graph=color_graph,
    )
    gray_verify = verify_claim_support(
        {"claim_text": "Light gray", "option_label": "A", "question_text": "What color is the vehicle?"},
        evidence_chain={"evidence_refs": ["obs:white_rv", "obs:white_bow"]},
        evidence_graph=color_graph,
    )
    llm_executor = SkillExecutor(
        llm_client=FakeVerifierLlmClient(),  # type: ignore[arg-type]
        config=SkillBackendConfig(default_mode=SkillBackendMode.LLM),
    )
    llm_white_verify = llm_executor.execute(
        "verify_claim_support",
        args={
            "claim": {"claim_text": "White", "option_label": "E", "question_text": "What color is the vehicle?"},
            "evidence_chain": {"evidence_refs": ["obs:white_rv"]},
            "question_text": "What color is the vehicle?",
        },
        graph=color_graph,
    )

    planned = [
        {"step_id": "r1", "skill_id": "generate_answer_hypotheses", "args": {"question_text": "$bindings.question_text", "options": "$bindings.options"}, "depends_on": []},
        {"step_id": "r2", "skill_id": "retrieve_evidence_for_hypothesis", "args": {"hypothesis": "$step.r1.hypotheses.1", "max_refs": 5}, "depends_on": ["r1"]},
        {"step_id": "r3", "skill_id": "bridge_evidence_hops", "args": {"source_evidence": "$step.r2.evidence_refs", "target_hypothesis": "$step.r1.hypotheses.1", "allowed_hop_types": ["same_location", "temporal_next"]}, "depends_on": ["r2"]},
        {"step_id": "r4", "skill_id": "score_hypothesis_support", "args": {"hypothesis": "$step.r1.hypotheses.1", "support_evidence": "$step.r2"}, "depends_on": ["r2"]},
        {"step_id": "r5", "skill_id": "compare_hypotheses", "args": {"scored_hypotheses": ["$step.r4.scored_hypothesis"]}, "depends_on": ["r4"]},
        {"step_id": "r6", "skill_id": "verify_temporal_social_consistency", "args": {"evidence_chain": "$step.r3.multi_hop_chain", "hypothesis": "$step.r1.hypotheses.1"}, "depends_on": ["r3"]},
    ]
    trace, _ = execute_reasoning_plan(reasoning_plan=planned, clue_memory_graph=graph, question=question)
    verifier_executor = SkillExecutor(
        llm_client=FakeVerifierLlmClient(),  # type: ignore[arg-type]
        config=SkillBackendConfig(default_mode=SkillBackendMode.RULE, llm_skills={"verify_claim_support"}),
    )
    contract_plan = [
        {"step_id": "r1", "skill_id": "generate_answer_hypotheses", "args": {"question_text": "$bindings.question_text", "options": "$bindings.options"}, "depends_on": []},
        {"step_id": "r2", "skill_id": "retrieve_by_event", "args": {"hypothesis": "$step.r1.hypotheses.1"}, "depends_on": ["r1"]},
        {
            "step_id": "r3",
            "skill_id": "verify_claim_support",
            "args": {
                "claim": "$step.r1.hypotheses.1",
                "evidence_chain": {"evidence_refs": "$step.r2.evidence_refs"},
                "support_policy": {"min_evidence_refs": 1},
            },
            "depends_on": ["r2"],
        },
        {
            "step_id": "r4",
            "skill_id": "commit_answer",
            "args": {
                "verified_claim": "$step.r3.verified_claim",
                "options": "$bindings.options",
                "answer_format": "$bindings.question.answer_format",
                "support_chain": {"evidence_refs": "$step.r3.evidence_chain"},
            },
            "depends_on": ["r3"],
        },
    ]
    contract_trace, contract_outputs = execute_reasoning_plan(
        reasoning_plan=contract_plan,
        clue_memory_graph=graph,
        question=question,
        skill_executor=verifier_executor,
    )

    required = {
        "generate_answer_hypotheses",
        "retrieve_evidence_for_hypothesis",
        "score_hypothesis_support",
        "compare_hypotheses",
        "bridge_evidence_hops",
        "verify_temporal_social_consistency",
        "verify_claim_support",
    }
    errors = []
    missing_enum = sorted(required - set(REASONING_SKILL_IDS))
    if missing_enum:
        errors.append(f"missing planner enum skills: {missing_enum}")
    for name, result in [
        ("generate_answer_hypotheses", hypotheses),
        ("retrieve_evidence_for_hypothesis", support),
        ("score_hypothesis_support", scored),
        ("compare_hypotheses", compared),
        ("bridge_evidence_hops", bridge),
        ("verify_temporal_social_consistency", consistency),
        ("verify_claim_support_white", white_verify),
        ("llm_verify_claim_support_white", llm_white_verify),
    ]:
        if not result.ok:
            errors.append(f"{name} failed: {result.failure_code}")
    if gray_verify.ok:
        errors.append("verify_claim_support accepted unsupported Light gray claim")
    executor_failures = [item for item in trace if item.get("failure_code") in {"unknown_skill_id", "invalid_skill_args"}]
    if executor_failures:
        errors.append(f"planner executor failures: {executor_failures}")
    contract_failures = [item for item in contract_trace if item.get("ok") is False]
    if contract_failures:
        errors.append(f"planner contract failures: {contract_failures}")
    if (contract_outputs.get("r4") or {}).get("final_answer") != "E":
        errors.append(f"commit_answer did not map verified claim to option E: {contract_outputs.get('r4')}")

    report = {
        "passed": not errors,
        "errors": errors,
        "hypothesis_count": len(hypotheses.outputs["hypotheses"]),
        "support_refs": support.evidence_refs,
        "bridge_refs": bridge.evidence_refs,
        "best_option": compared.outputs.get("best_hypothesis", {}).get("option_label"),
        "white_verify_score": white_verify.outputs.get("verification_score"),
        "gray_verify_score": gray_verify.outputs.get("verification_score"),
        "llm_verify_backend": llm_white_verify.outputs.get("backend"),
        "planner_trace": trace,
        "contract_trace": contract_trace,
        "contract_final_answer": (contract_outputs.get("r4") or {}).get("final_answer"),
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
