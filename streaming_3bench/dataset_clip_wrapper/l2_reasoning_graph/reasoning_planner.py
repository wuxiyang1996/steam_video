"""L2 reasoning planner: gpt-oss plans question-conditioned reasoning skill programs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atomic_skills import export_skill_ontology  # noqa: E402
from atomic_skills.common import stable_id  # noqa: E402
from atomic_skills.reasoning_graph_assembly import (  # noqa: E402
    assign_evidence_role,
    bridge_evidence_hops,
    commit_answer,
    compare_hypotheses,
    compose_evidence_chain,
    detect_missing_role,
    extract_claim,
    generate_answer_hypotheses,
    infer_causal_relation,
    infer_intention_or_motive,
    infer_social_contradiction,
    infer_state_change,
    infer_temporal_relation,
    localize_clue,
    parse_question_target,
    propose_evidence_roles,
    retrieve_by_entity,
    retrieve_evidence_for_hypothesis,
    retrieve_by_event,
    retrieve_by_relation,
    retrieve_by_time,
    search_counterevidence,
    score_hypothesis_support,
    verify_claim_support,
    verify_temporal_social_consistency,
)

from ..l1_clue_graph.clue_memory import make_reasoning_rollout_shell
from ..l1_clue_graph.graph_plan_validator import resolve_plan_value, _coerce_node_ref
from .l2_recursive_trace import attach_initial_l2_trajectory
from ..perception.openrouter_client import OpenRouterClient, load_openrouter_api_key
from ..schemas import GraphComposerConfig

REASONING_SKILL_EXECUTORS = {
    "parse_question_target": parse_question_target,
    "propose_evidence_roles": propose_evidence_roles,
    "retrieve_by_event": retrieve_by_event,
    "retrieve_by_entity": retrieve_by_entity,
    "retrieve_by_time": retrieve_by_time,
    "retrieve_by_relation": retrieve_by_relation,
    "localize_clue": localize_clue,
    "extract_claim": extract_claim,
    "assign_evidence_role": assign_evidence_role,
    "generate_answer_hypotheses": generate_answer_hypotheses,
    "retrieve_evidence_for_hypothesis": retrieve_evidence_for_hypothesis,
    "score_hypothesis_support": score_hypothesis_support,
    "compare_hypotheses": compare_hypotheses,
    "bridge_evidence_hops": bridge_evidence_hops,
    "verify_temporal_social_consistency": verify_temporal_social_consistency,
    "compose_evidence_chain": compose_evidence_chain,
    "detect_missing_role": detect_missing_role,
    "search_counterevidence": search_counterevidence,
    "infer_temporal_relation": infer_temporal_relation,
    "infer_state_change": infer_state_change,
    "infer_causal_relation": infer_causal_relation,
    "infer_intention_or_motive": infer_intention_or_motive,
    "infer_social_contradiction": infer_social_contradiction,
    "verify_claim_support": verify_claim_support,
    "commit_answer": commit_answer,
}

REASONING_SKILL_IDS = sorted(REASONING_SKILL_EXECUTORS.keys())


def _refs_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("evidence_refs", "support_refs", "clue_refs", "counterevidence_refs", "supporting_evidence"):
            if key in value:
                return _refs_from_value(value.get(key))
        if "multi_hop_chain" in value:
            return _refs_from_value(value.get("multi_hop_chain"))
        node_id = value.get("node_id") or value.get("evidence_ref")
        return [str(node_id)] if node_id else []
    if isinstance(value, list):
        refs: list[str] = []
        for item in value:
            refs.extend(_refs_from_value(item))
        return list(dict.fromkeys(refs))
    return []


def _chain_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if "evidence_refs" in value or "items" in value:
            return {
                **value,
                "evidence_refs": _refs_from_value(value),
                "items": value.get("items") or [],
            }
        if "role_labeled_evidence" in value:
            item = value["role_labeled_evidence"]
            return {
                "items": [item] if isinstance(item, dict) else [],
                "evidence_refs": _refs_from_value(item),
            }
        if "multi_hop_chain" in value:
            return _chain_from_value(value["multi_hop_chain"])
    return {"evidence_refs": _refs_from_value(value), "items": []}


def _augment_step_outputs(skill_id: str, outputs: dict[str, Any], evidence_refs: list[str]) -> dict[str, Any]:
    augmented = {**outputs, "evidence_refs": evidence_refs}
    if skill_id == "parse_question_target":
        augmented.setdefault("parsed_target", outputs)
    if skill_id in ("retrieve_by_event", "localize_clue"):
        augmented.setdefault("support_refs", evidence_refs)
    if skill_id == "assign_evidence_role":
        augmented.setdefault("evidence_chain", _chain_from_value(outputs))
    if skill_id == "bridge_evidence_hops":
        augmented.setdefault("evidence_chain", _chain_from_value(outputs.get("multi_hop_chain")))
    if skill_id == "verify_claim_support":
        augmented.setdefault("claim", outputs.get("verified_claim"))
        augmented.setdefault("evidence_chain", _chain_from_value(evidence_refs))
    if skill_id == "compare_hypotheses" and isinstance(outputs.get("best_hypothesis"), dict):
        best = outputs["best_hypothesis"]
        augmented.setdefault("claim", best)
        augmented.setdefault("verified_claim", best)
        augmented.setdefault("evidence_chain", _chain_from_value(best.get("support_refs") or evidence_refs))
    if skill_id == "score_hypothesis_support" and "scored_hypothesis" in outputs:
        augmented.setdefault("scored_hypotheses", [outputs["scored_hypothesis"]])
    return augmented


def _repair_tokens(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[a-z0-9]+", text) if len(tok) > 2}


def _build_commonsense_repair_pack(
    *,
    question: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    support_refs: list[str],
    answerability_diagnostic: dict[str, Any],
    final_label: Any,
) -> dict[str, Any]:
    """Build an explicit non-committing social/common-sense repair pack.

    This pack is deliberately separated from verified evidence. It may guide a
    repair/retrieval pass, but it must not make an unsupported answer accepted.
    """
    nodes = {
        node.get("node_id"): node
        for node in clue_memory_graph.get("nodes") or []
        if isinstance(node, dict) and node.get("node_id")
    }
    diagnostic_types = {"question_requirement", "required_modality", "answerability_gap", "l2_repair_reminder"}
    visual_refs = [
        ref for ref in support_refs
        if ref in nodes and nodes[ref].get("node_type") not in diagnostic_types
    ]
    if not visual_refs:
        for node in clue_memory_graph.get("nodes") or []:
            if not isinstance(node, dict) or node.get("node_type") in diagnostic_types:
                continue
            text = str(node.get("text") or node.get("event_description") or "")
            if text.strip():
                visual_refs.append(str(node.get("node_id")))
            if len(visual_refs) >= 6:
                break

    question_text = str(question.get("question_text") or "")
    q_tokens = _repair_tokens(question_text)
    options = [option for option in question.get("options") or [] if isinstance(option, dict)]
    option_rows = []
    for option in options:
        text = str(option.get("text") or "")
        tokens = _repair_tokens(text)
        social_terms = tokens & {
            "because", "interested", "curiosity", "desire", "explore", "history",
            "cultural", "understand", "feel", "adventure", "seeking", "protect",
            "fear", "confidential", "misunderstandings", "thrill",
        }
        overlap = len(tokens & q_tokens)
        option_rows.append(
            {
                "label": option.get("label"),
                "text": text,
                "commonsense_score": round(overlap + 0.5 * len(social_terms), 4),
                "bridge": (
                    "This option can be considered as a social/common-sense explanation, "
                    "but it needs visual-context support before commit."
                ),
            }
        )
    option_rows.sort(key=lambda row: row["commonsense_score"], reverse=True)

    missing = answerability_diagnostic.get("missing_requirements") or []
    if not missing and option_rows:
        missing = ["discriminative_visual_evidence"]
    return {
        "status": "repair_candidate",
        "trust_level": "commonsense_hypothesis_not_visual_evidence",
        "trigger": "l2_rejected_or_weak_evidence",
        "missing_requirements": missing,
        "visual_context_refs": list(dict.fromkeys(visual_refs))[:8],
        "commonsense_hypotheses": option_rows[:5],
        "candidate_label_before_repair": final_label,
        "cannot_commit_reason": (
            "Common-sense/social inference is not accepted unless verified by "
            "concrete non-diagnostic video evidence refs."
        ),
        "recommended_next_action": (
            "expand/retrieve adjacent fine clips for the top commonsense hypotheses, "
            "then rerun verify_claim_support before commit."
        ),
    }

_REASONING_SKILL_CONTRACTS = {
    "parse_question_target": "args: question_text, options -> parsed_target object",
    "propose_evidence_roles": "args: question_text, parsed_target, task_family -> role_constraints",
    "generate_answer_hypotheses": "args: question_text, options, parsed_target -> hypotheses list",
    "retrieve_evidence_for_hypothesis": "args: hypothesis, max_refs -> support_refs",
    "score_hypothesis_support": "args: hypothesis, support_evidence, counterevidence -> scored_hypothesis/scored_hypotheses",
    "compare_hypotheses": "args: scored_hypotheses -> best_hypothesis",
    "retrieve_by_event": "args: evidence_graph, event_description, time_range? -> event_nodes, evidence_refs",
    "localize_clue": "args: candidate_evidence, role_constraint, question_context -> clue_refs",
    "extract_claim": "args: evidence_ref, claim_query? -> claim_text, evidence_ref",
    "assign_evidence_role": "args: evidence_ref, role_schema, question_context -> role_labeled_evidence",
    "bridge_evidence_hops": "args: source_evidence, target_hypothesis -> multi_hop_chain",
    "verify_claim_support": "args: claim, evidence_chain, support_policy -> verified_claim",
    "commit_answer": "args: verified_claim, options, answer_format, support_chain -> final_answer",
}

_REASONING_PLAN_PROMPT = """You are an expert video-reasoning planner. Given a question and a Layer-1
clue-memory graph (perception evidence), plan which reasoning skills to execute and in what order.

This is a MULTIPLE-CHOICE skill selection task:
- skill_id MUST be exactly one value from allowed_skill_ids.
- Reference prior step outputs with $step.<step_id>.evidence_refs.N or $step.<step_id>.<output_field>.
- Use $bindings.question_text, $bindings.options, $bindings.graph for inputs.
- L1 node_ids from the clue graph can be used directly as string refs.

Return JSON only:
{
  "reasoning_plan": [
    {
      "step_id": "r1",
      "skill_id": "parse_question_target",
      "args": {"question_text": "$bindings.question_text", "options": "$bindings.options"},
      "depends_on": []
    },
    {
      "step_id": "r2",
      "skill_id": "retrieve_by_event",
      "args": {"evidence_graph": "$bindings.graph", "event_description": "$bindings.question_text"},
      "depends_on": ["r1"]
    }
  ],
  "notes": "short reasoning strategy summary",
  "expected_answer_format": "multiple_choice"
}

Skill execution rules:
1. Always start with parse_question_target and propose_evidence_roles.
2. For multiple-choice or complex social questions, prefer the option-level path:
   generate_answer_hypotheses -> retrieve_evidence_for_hypothesis ->
   score_hypothesis_support -> compare_hypotheses.
3. Use bridge_evidence_hops when the answer requires linking source evidence to
   an option through object/location/action/state evidence.
4. Use verify_temporal_social_consistency before final verification when social
   or temporal plausibility matters.
5. Use retrieve_by_event / retrieve_by_entity / retrieve_by_time to find relevant L1 evidence.
6. Use localize_clue and extract_claim to ground claims in evidence.
7. Use assign_evidence_role + compose_evidence_chain to build the support structure.
8. Use infer_* skills for temporal, causal, state-change, or social reasoning as needed.
9. For ordinary answerable graphs, end with verify_claim_support then commit_answer.
   If the graph contains l2_repair_reminder / answerability_gap nodes or missing
   requirements, treat the run as repair-only: reason over weak visual context,
   keep out-of-scope modalities out of the claim, and do not commit an answer
   unless verify_claim_support returns concrete non-diagnostic evidence refs.
10. Keep plans between 8-18 steps. Do not over-plan.
11. Do not output chain-of-thought.

Stable output contracts:
- parse_question_target outputs the parsed target object directly; refer to $step.r1.parsed_target.
- generate_answer_hypotheses outputs hypotheses; pass the full $step.rN.hypotheses list to option scoring.
- retrieve_evidence_for_hypothesis outputs support_refs and evidence_refs.
- score_hypothesis_support outputs scored_hypothesis for one item or scored_hypotheses for a list.
- compare_hypotheses outputs best_hypothesis.
- verify_claim_support outputs verified_claim; commit_answer.verified_claim MUST use that object, not $step.x.passed.
- commit_answer.support_chain MUST be an object with evidence_refs, not a raw list.
"""


def _build_reasoning_plan_schema(allowed_skill_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "reasoning_skill_plan",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reasoning_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "step_id": {"type": "string"},
                                "skill_id": {"type": "string", "enum": allowed_skill_ids},
                                "args": {"type": "object", "additionalProperties": True},
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["step_id", "skill_id", "args", "depends_on"],
                        },
                    },
                    "notes": {"type": "string"},
                    "expected_answer_format": {"type": "string"},
                },
                "required": ["reasoning_plan", "notes"],
            },
        },
    }


def _summarize_clue_graph(clue_memory_graph: dict[str, Any], max_nodes: int = 20) -> dict[str, Any]:
    """Compact summary of L1 graph for the reasoning planner prompt."""
    nodes = clue_memory_graph.get("nodes") or []
    priority = {
        "clue": 0,
        "event": 1,
        "observation": 2,
        "state": 3,
        "entity_mention": 4,
        "entity": 5,
        "dialogue_span": 6,
        "ocr": 7,
        "object": 8,
        "question_requirement": 20,
        "answerability_gap": 21,
        "clip": 50,
    }
    ordered_nodes = sorted(
        nodes,
        key=lambda node: (
            priority.get(str(node.get("node_type") or ""), 10),
            float((node.get("time_span") or {}).get("start_s") or 0.0),
        ),
    )
    summary_nodes = []
    for node in ordered_nodes[:max_nodes]:
        summary_nodes.append({
            "node_id": node.get("node_id"),
            "node_type": node.get("node_type"),
            "text": (node.get("text") or node.get("event_description") or node.get("surface_form") or "")[:120],
            "time_span": node.get("time_span"),
        })
    return {
        "total_nodes": len(nodes),
        "shown_nodes": summary_nodes,
        "node_types": list({n.get("node_type") for n in nodes if n.get("node_type")}),
        "edge_count": len(clue_memory_graph.get("edges") or []),
    }


def _default_multi_hop_mcq_plan() -> list[dict[str, Any]]:
    """Conservative fallback plan for short/offline multiple-choice reasoning."""
    return [
        {
            "step_id": "r1",
            "skill_id": "parse_question_target",
            "args": {"question_text": "$bindings.question_text", "options": "$bindings.options"},
            "depends_on": [],
        },
        {
            "step_id": "r2",
            "skill_id": "propose_evidence_roles",
            "args": {
                "question_text": "$bindings.question_text",
                "parsed_target": "$step.r1.parsed_target",
                "task_family": "$bindings.task_family",
            },
            "depends_on": ["r1"],
        },
        {
            "step_id": "r3",
            "skill_id": "generate_answer_hypotheses",
            "args": {
                "question_text": "$bindings.question_text",
                "options": "$bindings.options",
                "parsed_target": "$step.r1.parsed_target",
            },
            "depends_on": ["r1"],
        },
        {
            "step_id": "r4",
            "skill_id": "retrieve_evidence_for_hypothesis",
            "args": {"hypothesis": "$step.r3.hypotheses", "max_refs": 6},
            "depends_on": ["r3"],
        },
        {
            "step_id": "r5",
            "skill_id": "score_hypothesis_support",
            "args": {
                "hypothesis": "$step.r3.hypotheses",
                "support_evidence": "$step.r4",
                "counterevidence": [],
            },
            "depends_on": ["r4"],
        },
        {
            "step_id": "r6",
            "skill_id": "compare_hypotheses",
            "args": {"scored_hypotheses": "$step.r5.scored_hypotheses"},
            "depends_on": ["r5"],
        },
        {
            "step_id": "r7",
            "skill_id": "bridge_evidence_hops",
            "args": {
                "source_evidence": "$step.r6.best_hypothesis.support_refs",
                "target_hypothesis": "$step.r6.best_hypothesis",
                "max_hops": 2,
            },
            "depends_on": ["r6"],
        },
        {
            "step_id": "r8",
            "skill_id": "verify_temporal_social_consistency",
            "args": {
                "evidence_chain": "$step.r7.multi_hop_chain",
                "hypothesis": "$step.r6.best_hypothesis",
            },
            "depends_on": ["r7", "r6"],
        },
        {
            "step_id": "r9",
            "skill_id": "verify_claim_support",
            "args": {
                "claim": "$step.r6.best_hypothesis",
                "evidence_chain": "$step.r7.multi_hop_chain",
                "support_policy": {"min_evidence_refs": 1},
            },
            "depends_on": ["r6", "r7", "r8"],
        },
        {
            "step_id": "r10",
            "skill_id": "commit_answer",
            "args": {
                "verified_claim": "$step.r9.verified_claim",
                "options": "$bindings.options",
                "answer_format": "multiple_choice",
                "support_chain": "$step.r9.evidence_chain",
            },
            "depends_on": ["r9"],
        },
    ]


def plan_reasoning_skills(
    *,
    question: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    task_family: str,
    client: OpenRouterClient,
) -> dict[str, Any]:
    """Call gpt-oss to plan a question-conditioned reasoning skill program."""
    ontology = export_skill_ontology()["reasoning_graph_assembly"]
    errors: list[str] = []
    attempts = [
        {"name": "full", "max_nodes": 20, "include_ontology": True},
        {"name": "compact_retry", "max_nodes": 10, "include_ontology": False},
    ]

    for attempt in attempts:
        graph_summary = _summarize_clue_graph(clue_memory_graph, max_nodes=attempt["max_nodes"])
        payload = {
            "task": "plan_reasoning_over_clue_graph",
            "question": question,
            "task_family": task_family,
            "clue_graph_summary": graph_summary,
            "allowed_skill_ids": REASONING_SKILL_IDS,
            "skill_contracts": _REASONING_SKILL_CONTRACTS,
            "instructions": _REASONING_PLAN_PROMPT,
        }
        if attempt["include_ontology"]:
            payload["ontology"] = ontology

        try:
            response = client.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert video-reasoning planner. "
                            "Choose reasoning skills from allowed_skill_ids. "
                            "Return compact valid JSON only; no markdown, no comments, no trailing commas."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format=_build_reasoning_plan_schema(REASONING_SKILL_IDS),
            )
            response["model"] = client.model
            response["planner"] = "gpt_oss_reasoning_planner"
            response["planner_attempt"] = attempt["name"]
            if errors:
                response["planner_retry_errors"] = errors
            return response
        except Exception as exc:
            errors.append(f"{attempt['name']}: {exc}")

    raise RuntimeError("; ".join(errors))


def execute_reasoning_plan(
    *,
    reasoning_plan: list[dict[str, Any]],
    clue_memory_graph: dict[str, Any],
    question: dict[str, Any],
    skill_executor: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute a reasoning skill plan over the clue graph, returning trace + step_outputs.

    If skill_executor is provided (a SkillExecutor instance), skills configured
    for LLM/VLM mode will be dispatched via API calls; otherwise pure rule-based.
    """
    from copy import deepcopy

    graph = {
        "schema_version": clue_memory_graph.get("schema_version"),
        "nodes": deepcopy(clue_memory_graph.get("nodes") or []),
        "edges": deepcopy(clue_memory_graph.get("edges") or []),
    }
    question_text = question.get("question_text") or ""
    options = question.get("options") or []

    bindings = {
        "question_text": question_text,
        "options": options,
        "answer_format": question.get("answer_format") or ("multiple_choice" if options else "free_text"),
        "question": {
            "question_text": question_text,
            "options": options,
            "answer_format": question.get("answer_format") or ("multiple_choice" if options else "free_text"),
        },
        "graph": graph,
        "task_family": question.get("task_family") or "",
    }

    trace: list[dict[str, Any]] = []
    step_outputs: dict[str, Any] = {}

    def _one_hypothesis(value: Any) -> Any:
        if isinstance(value, list):
            return next((item for item in value if isinstance(item, dict)), value[0] if value else {"claim_text": question_text})
        return value or {"claim_text": question_text}

    def _hypothesis_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value or [{"claim_text": question_text}]
        if value:
            return [value]
        return [{"claim_text": question_text}]

    def _refs(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            for key in ("evidence_refs", "support_refs", "clue_refs", "counterevidence_refs", "supporting_evidence"):
                if key in value:
                    return _refs(value.get(key))
            if "multi_hop_chain" in value:
                return _refs(value.get("multi_hop_chain"))
            node_id = value.get("node_id") or value.get("evidence_ref")
            return [str(node_id)] if node_id else []
        if isinstance(value, list):
            refs: list[str] = []
            for item in value:
                refs.extend(_refs(item))
            return list(dict.fromkeys(refs))
        return []

    def _chain(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            if "evidence_refs" in value or "items" in value:
                return {
                    **value,
                    "evidence_refs": _refs(value),
                    "items": value.get("items") or [],
                }
            if "role_labeled_evidence" in value:
                item = value["role_labeled_evidence"]
                return {
                    "items": [item] if isinstance(item, dict) else [],
                    "evidence_refs": _refs(item),
                }
            if "multi_hop_chain" in value:
                return _chain(value["multi_hop_chain"])
        return {"evidence_refs": _refs(value), "items": []}

    def _claim(value: Any, *, fallback: str | None = None) -> dict[str, Any]:
        if isinstance(value, dict):
            if value.get("verified_claim") and isinstance(value["verified_claim"], dict):
                return value["verified_claim"]
            if value.get("best_hypothesis") and isinstance(value["best_hypothesis"], dict):
                value = value["best_hypothesis"]
            text = value.get("claim_text") or value.get("text") or value.get("final_answer") or fallback or question_text
            return {
                **value,
                "claim_text": text,
                "text": value.get("text") or text,
                "claim_status": value.get("claim_status") or "verified",
            }
        if isinstance(value, str):
            return {"claim_text": value, "text": value, "claim_status": "verified"}
        return {"claim_text": fallback or question_text, "text": fallback or question_text, "claim_status": "verified"}

    def _lookup_recent_verified_claim() -> dict[str, Any] | None:
        for outputs in reversed(list(step_outputs.values())):
            verified = outputs.get("verified_claim") if isinstance(outputs, dict) else None
            if isinstance(verified, dict):
                return verified
            best = outputs.get("best_hypothesis") if isinstance(outputs, dict) else None
            if isinstance(best, dict):
                return _claim(best)
        return None

    def _lookup_recent_best_hypothesis() -> dict[str, Any] | None:
        for outputs in reversed(list(step_outputs.values())):
            best = outputs.get("best_hypothesis") if isinstance(outputs, dict) else None
            if isinstance(best, dict):
                return best
            scored = outputs.get("scored_hypothesis") if isinstance(outputs, dict) else None
            if isinstance(scored, dict):
                return scored
        return None

    def _lookup_recent_hypotheses() -> list[Any]:
        for outputs in reversed(list(step_outputs.values())):
            hypotheses = outputs.get("hypotheses") if isinstance(outputs, dict) else None
            if isinstance(hypotheses, list) and hypotheses:
                return hypotheses
        best = _lookup_recent_best_hypothesis()
        return [best] if best else []

    def _lookup_recent_evidence_refs() -> list[str]:
        for outputs in reversed(list(step_outputs.values())):
            refs = _refs(outputs) if isinstance(outputs, dict) else []
            if refs:
                return refs
        return []

    def _augment_step_outputs(skill_id: str, outputs: dict[str, Any], evidence_refs: list[str]) -> dict[str, Any]:
        augmented = {**outputs, "evidence_refs": evidence_refs}
        if skill_id == "parse_question_target":
            augmented.setdefault("parsed_target", outputs)
        if skill_id == "retrieve_by_event":
            augmented.setdefault("support_refs", evidence_refs)
        if skill_id == "localize_clue":
            augmented.setdefault("support_refs", evidence_refs)
        if skill_id == "assign_evidence_role":
            augmented.setdefault("evidence_chain", _chain(outputs))
        if skill_id == "bridge_evidence_hops":
            augmented.setdefault("evidence_chain", _chain(outputs.get("multi_hop_chain")))
        if skill_id == "verify_claim_support":
            augmented.setdefault("claim", outputs.get("verified_claim"))
            augmented.setdefault("evidence_chain", _chain(evidence_refs))
        if skill_id == "compare_hypotheses" and isinstance(outputs.get("best_hypothesis"), dict):
            best = outputs["best_hypothesis"]
            augmented.setdefault("claim", best)
            augmented.setdefault("verified_claim", best)
            augmented.setdefault("evidence_chain", _chain(best.get("support_refs") or evidence_refs))
        if skill_id == "score_hypothesis_support" and "scored_hypothesis" in outputs:
            augmented.setdefault("scored_hypotheses", [outputs["scored_hypothesis"]])
        return augmented

    for step in reasoning_plan:
        step_id = step.get("step_id")
        skill_id = step.get("skill_id")
        raw_args = dict(step.get("args") or {})

        if skill_id not in REASONING_SKILL_EXECUTORS:
            trace.append({"step_id": step_id, "skill_id": skill_id, "ok": False, "failure_code": "unknown_skill_id"})
            continue

        try:
            resolved_args = resolve_plan_value(raw_args, bindings, step_outputs)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            trace.append({
                "step_id": step_id,
                "skill_id": skill_id,
                "ok": False,
                "failure_code": "invalid_step_reference",
                "messages": [str(exc)],
            })
            continue

        if skill_id == "verify_claim_support":
            raw_claim = resolved_args.get("claim")
            if raw_claim in (None, [], {}):
                raw_claim = _lookup_recent_best_hypothesis()
            elif isinstance(raw_claim, str):
                recent_best = _lookup_recent_best_hypothesis()
                if isinstance(recent_best, dict):
                    recent_text = recent_best.get("claim_text") or recent_best.get("text")
                    if recent_text == raw_claim:
                        raw_claim = recent_best
            resolved_args["claim"] = _claim(raw_claim, fallback=question_text)
            support_policy = resolved_args.get("support_policy") or {"min_evidence_refs": 1}
            if isinstance(support_policy, str):
                support_policy = {"min_evidence_refs": 1}
            resolved_args["support_policy"] = support_policy
            evidence_chain = _chain(resolved_args.get("evidence_chain"))
            if not evidence_chain.get("evidence_refs"):
                evidence_chain = _chain(resolved_args["claim"])
            resolved_args["evidence_chain"] = evidence_chain
            resolved_args.setdefault("question_text", resolved_args["claim"].get("question_text") or question_text)
        elif skill_id == "score_hypothesis_support":
            if resolved_args.get("hypothesis") in (None, [], {}):
                resolved_args["hypothesis"] = _lookup_recent_hypotheses()
            counter = resolved_args.get("counterevidence")
            if counter is None:
                resolved_args["counterevidence"] = []
        elif skill_id == "compose_evidence_chain":
            if not resolved_args.get("dependency_template"):
                resolved_args["dependency_template"] = "support_chain"
            labeled = resolved_args.get("role_labeled_evidence")
            if isinstance(labeled, dict):
                resolved_args["role_labeled_evidence"] = [labeled]
        elif skill_id == "retrieve_by_time":
            anchor = resolved_args.get("anchor_event_or_time")
            if isinstance(anchor, (int, float)):
                value = float(anchor)
                resolved_args["anchor_event_or_time"] = {"start_s": value, "end_s": value}
            resolved_args["window_before"] = float(resolved_args.get("window_before") or 0.0)
            resolved_args["window_after"] = float(resolved_args.get("window_after") or 0.0)

        # --- LLM/VLM dispatch via SkillExecutor ---
        if skill_executor is not None:
            from atomic_skills.skill_backends import SkillBackendMode
            mode = skill_executor.config.mode_for(skill_id)
            backend_safe = True
            if skill_id == "score_hypothesis_support" and len(_hypothesis_list(resolved_args.get("hypothesis"))) > 1:
                # The LLM scorer prompt is a single-hypothesis contract. Let the
                # deterministic branch split and aggregate option scores, then
                # keep the LLM verifier for the final support check.
                backend_safe = False
            if mode in (SkillBackendMode.LLM, SkillBackendMode.VLM):
                has_client = (
                    (mode == SkillBackendMode.LLM and skill_executor.llm_client)
                    or (mode == SkillBackendMode.VLM and skill_executor.vlm_client)
                )
                if has_client and backend_safe:
                    try:
                        result = skill_executor.execute(skill_id, args=resolved_args, graph=graph)
                        trace.append({
                            "step_id": step_id,
                            "skill_id": skill_id,
                            "ok": result.ok,
                            "failure_code": result.failure_code,
                            "evidence_refs": result.evidence_refs,
                            "confidence": result.confidence,
                            "backend": mode.value,
                        })
                        if step_id:
                            step_outputs[step_id] = _augment_step_outputs(skill_id, result.outputs, result.evidence_refs)
                        continue
                    except Exception as exc:
                        trace.append({
                            "step_id": step_id,
                            "skill_id": skill_id,
                            "ok": False,
                            "failure_code": f"{mode.value}_backend_error",
                            "messages": [str(exc)],
                            "backend": mode.value,
                        })
                        continue

        executor = REASONING_SKILL_EXECUTORS[skill_id]

        try:
            if skill_id in ("retrieve_by_event", "retrieve_by_entity", "retrieve_by_time",
                            "retrieve_by_relation", "assign_evidence_role",
                            "search_counterevidence"):
                filtered = {k: v for k, v in resolved_args.items() if k != "evidence_graph"}
                if skill_id == "retrieve_by_event":
                    if "event_description" not in filtered:
                        hypothesis = _one_hypothesis(filtered.pop("hypothesis", None) or filtered.pop("claim", None))
                        if isinstance(hypothesis, dict):
                            filtered["event_description"] = (
                                hypothesis.get("claim_text")
                                or hypothesis.get("text")
                                or filtered.get("question_context")
                                or question_text
                            )
                        else:
                            filtered["event_description"] = str(hypothesis or filtered.get("question_context") or question_text)
                    filtered.pop("query", None)
                    filtered.pop("question_context", None)
                    filtered = {
                        key: filtered[key]
                        for key in ("event_description", "time_range", "entity_filter")
                        if key in filtered
                    }
                if skill_id == "assign_evidence_role":
                    ev_ref = filtered.get("evidence_ref")
                    if isinstance(ev_ref, list):
                        filtered["evidence_ref"] = ev_ref[0] if ev_ref else "missing"
                    elif isinstance(ev_ref, dict):
                        filtered["evidence_ref"] = ev_ref.get("node_id") or ev_ref.get("evidence_ref") or "missing"
                result = executor(graph, **filtered)
            elif skill_id == "extract_claim":
                filtered = {k: v for k, v in resolved_args.items() if k != "evidence_graph"}
                ev_ref = filtered.get("evidence_ref")
                if isinstance(ev_ref, list):
                    filtered["evidence_ref"] = ev_ref[0] if ev_ref else "missing"
                elif isinstance(ev_ref, dict):
                    filtered["evidence_ref"] = ev_ref.get("node_id") or ev_ref.get("evidence_ref") or "missing"
                result = executor(graph, **filtered)
            elif skill_id == "infer_state_change":
                filtered = {k: v for k, v in resolved_args.items() if k != "evidence_graph"}
                ba_refs = filtered.get("before_after_refs") or []
                if isinstance(ba_refs, str):
                    ba_refs = [ba_refs]
                ba_refs = [r if isinstance(r, str) else (r.get("node_id") if isinstance(r, dict) else str(r)) for r in ba_refs]
                filtered["before_after_refs"] = ba_refs
                result = executor(graph, **filtered)
            elif skill_id == "infer_temporal_relation":
                event_refs = resolved_args.get("event_refs") or []
                if isinstance(event_refs, str):
                    event_refs = [event_refs]
                elif isinstance(event_refs, dict):
                    event_refs = [event_refs.get("node_id") or str(event_refs)]
                event_refs = [r if isinstance(r, str) else (r.get("node_id") if isinstance(r, dict) else str(r)) for r in event_refs]
                result = executor(event_refs, evidence_graph=graph)
            elif skill_id == "localize_clue":
                candidate_evidence = resolved_args.get("candidate_evidence") or [
                    n for n in graph.get("nodes", []) if n.get("node_type") in ("observation", "event")
                ]
                result = executor(
                    candidate_evidence,
                    role_constraint=resolved_args.get("role_constraint") or "supporting_evidence",
                    question_context=resolved_args.get("question_context") or question_text,
                )
            elif skill_id == "parse_question_target":
                result = executor(
                    resolved_args.get("question_text") or question_text,
                    options=resolved_args.get("options") or options or None,
                )
            elif skill_id == "propose_evidence_roles":
                parsed_target = resolved_args.get("parsed_target")
                if not isinstance(parsed_target, dict):
                    parsed_target = step_outputs.get(reasoning_plan[0].get("step_id"), {})
                    if not isinstance(parsed_target, dict):
                        parsed_target = {}
                result = executor(
                    resolved_args.get("question_text") or question_text,
                    parsed_target,
                    task_family=resolved_args.get("task_family") or "",
                )
            elif skill_id == "generate_answer_hypotheses":
                parsed_target = resolved_args.get("parsed_target")
                if not isinstance(parsed_target, dict):
                    parsed_target = {}
                result = executor(
                    resolved_args.get("question_text") or question_text,
                    options=resolved_args.get("options") or options or None,
                    parsed_target=parsed_target,
                )
            elif skill_id == "retrieve_evidence_for_hypothesis":
                hypotheses = _hypothesis_list(resolved_args.get("hypothesis"))
                if len(hypotheses) == 1:
                    result = executor(
                        graph,
                        hypothesis=hypotheses[0],
                        max_refs=int(resolved_args.get("max_refs") or 6),
                    )
                    if isinstance(result.outputs, dict):
                        result.outputs.setdefault("hypothesis", hypotheses[0])
                else:
                    support_by_hypothesis = []
                    all_refs: list[str] = []
                    best_confidence = 0.0
                    for hypothesis in hypotheses:
                        partial = executor(graph, hypothesis=hypothesis, max_refs=int(resolved_args.get("max_refs") or 6))
                        all_refs.extend(partial.evidence_refs)
                        best_confidence = max(best_confidence, partial.confidence)
                        support_by_hypothesis.append({
                            "hypothesis": hypothesis,
                            "support_refs": partial.outputs.get("support_refs") or [],
                            "retrieval_scores": partial.outputs.get("retrieval_scores") or {},
                            "ok": partial.ok,
                        })
                    all_refs = list(dict.fromkeys(all_refs))
                    from atomic_skills.common import make_result
                    result = make_result(
                        "retrieve_evidence_for_hypothesis",
                        {"support_by_hypothesis": support_by_hypothesis, "support_refs": all_refs},
                        all_refs,
                        ok=bool(all_refs),
                        failure_code=None if all_refs else "no_hypothesis_evidence",
                        confidence=best_confidence,
                    )
            elif skill_id == "score_hypothesis_support":
                counter = resolved_args.get("counterevidence") or []
                if isinstance(counter, dict):
                    counter = counter.get("counterevidence_refs") or counter.get("evidence_refs") or []
                if resolved_args.get("hypothesis") in (None, [], {}):
                    resolved_args["hypothesis"] = _lookup_recent_hypotheses()
                hypotheses = _hypothesis_list(resolved_args.get("hypothesis"))
                support_arg = resolved_args.get("support_evidence") or []
                support_by_hypothesis = support_arg.get("support_by_hypothesis") if isinstance(support_arg, dict) else None
                llm_score_each = False
                if skill_executor is not None:
                    from atomic_skills.skill_backends import SkillBackendMode
                    llm_score_each = (
                        skill_executor.config.mode_for("score_hypothesis_support") == SkillBackendMode.LLM
                        and bool(skill_executor.llm_client)
                    )
                if len(hypotheses) == 1:
                    support = support_arg if isinstance(support_arg, dict) else _refs(support_arg)
                    if llm_score_each:
                        result = skill_executor.execute(
                            "score_hypothesis_support",
                            args={
                                "hypothesis": hypotheses[0],
                                "support_evidence": support,
                                "counterevidence": _refs(counter),
                            },
                            graph=graph,
                        )
                    else:
                        result = executor(
                            hypotheses[0],
                            support_evidence=support,
                            counterevidence=_refs(counter),
                            evidence_graph=graph,
                        )
                else:
                    scored = []
                    refs: list[str] = []
                    for index, hypothesis in enumerate(hypotheses):
                        support = []
                        if isinstance(support_by_hypothesis, list) and index < len(support_by_hypothesis):
                            support = support_by_hypothesis[index].get("support_refs") or []
                        elif isinstance(support_arg, dict):
                            support = support_arg.get("support_refs") or support_arg.get("evidence_refs") or []
                        else:
                            support = _refs(support_arg)
                        if llm_score_each:
                            partial = skill_executor.execute(
                                "score_hypothesis_support",
                                args={
                                    "hypothesis": hypothesis,
                                    "support_evidence": support,
                                    "counterevidence": _refs(counter),
                                },
                                graph=graph,
                            )
                        else:
                            partial = executor(
                                hypothesis,
                                support_evidence=support,
                                counterevidence=_refs(counter),
                                evidence_graph=graph,
                            )
                        scored.append(partial.outputs.get("scored_hypothesis") or {})
                        refs.extend(partial.evidence_refs)
                    refs = list(dict.fromkeys(refs))
                    from atomic_skills.common import make_result
                    result = make_result(
                        "score_hypothesis_support",
                        {"scored_hypotheses": scored, "scored_hypothesis": scored[0] if scored else {}},
                        refs,
                        ok=any(item.get("support_refs") for item in scored),
                        failure_code=None if any(item.get("support_refs") for item in scored) else "missing_support_evidence",
                        confidence=max((item.get("overall_score", 0.0) for item in scored), default=0.0),
                    )
            elif skill_id == "compare_hypotheses":
                scored = resolved_args.get("scored_hypotheses") or []
                if isinstance(scored, dict):
                    scored = [scored]
                scored = [item for item in scored if isinstance(item, dict)]
                if not scored:
                    for outputs in step_outputs.values():
                        if not isinstance(outputs, dict):
                            continue
                        if isinstance(outputs.get("scored_hypotheses"), list):
                            scored.extend(item for item in outputs["scored_hypotheses"] if isinstance(item, dict))
                        elif isinstance(outputs.get("scored_hypothesis"), dict):
                            scored.append(outputs["scored_hypothesis"])
                result = executor(
                    scored,
                    decision_policy=resolved_args.get("decision_policy") if isinstance(resolved_args.get("decision_policy"), dict) else None,
                )
            elif skill_id == "bridge_evidence_hops":
                source = resolved_args.get("source_evidence") or []
                if isinstance(source, dict):
                    source = source.get("evidence_refs") or source.get("support_refs") or []
                result = executor(
                    graph,
                    source_evidence=source,
                    target_hypothesis=_one_hypothesis(resolved_args.get("target_hypothesis")),
                    allowed_hop_types=resolved_args.get("allowed_hop_types") if isinstance(resolved_args.get("allowed_hop_types"), list) else None,
                    max_hops=int(resolved_args.get("max_hops") or 2),
                )
            elif skill_id == "verify_temporal_social_consistency":
                result = executor(
                    _chain(resolved_args.get("evidence_chain")),
                    hypothesis=_one_hypothesis(resolved_args.get("hypothesis")),
                    evidence_graph=graph,
                )
            elif skill_id == "compose_evidence_chain":
                labeled = resolved_args.get("role_labeled_evidence") or []
                labeled = [item for item in labeled if isinstance(item, dict)]
                result = executor(
                    labeled or [{"role": "supporting_evidence", "evidence_ref": "unknown", "text": "", "confidence": 0.0}],
                    dependency_template=resolved_args.get("dependency_template") or "support_chain",
                )
            elif skill_id == "detect_missing_role":
                result = executor(
                    _chain(resolved_args.get("evidence_chain")),
                    required_roles=resolved_args.get("required_roles") or [],
                )
            elif skill_id == "infer_causal_relation":
                result = executor(
                    resolved_args.get("candidate_cause") or "cause",
                    resolved_args.get("candidate_effect") or "effect",
                    evidence_chain=_chain(resolved_args.get("evidence_chain")),
                )
            elif skill_id == "infer_intention_or_motive":
                hypothesis = _one_hypothesis(resolved_args.get("hypothesis"))
                if isinstance(hypothesis, dict):
                    action = hypothesis.get("claim_text") or hypothesis.get("text") or "action"
                    agent = hypothesis.get("agent") or resolved_args.get("agent") or "person"
                else:
                    action = str(hypothesis or "action")
                    agent = resolved_args.get("agent") or "person"
                context_refs = _refs(resolved_args.get("context_evidence") or resolved_args.get("support_evidence"))
                if not context_refs:
                    context_refs = _refs(hypothesis) or _lookup_recent_evidence_refs()
                result = executor(
                    agent,
                    resolved_args.get("actions") or [action],
                    context_evidence=context_refs,
                )
            elif skill_id == "infer_social_contradiction":
                result = executor(
                    _claim(resolved_args.get("claim_or_alibi"), fallback=question_text),
                    evidence_chain=_chain(resolved_args.get("evidence_chain")),
                    counterevidence=resolved_args.get("counterevidence") or [],
                )
            elif skill_id == "verify_claim_support":
                raw_claim = resolved_args.get("claim")
                if raw_claim in (None, [], {}):
                    raw_claim = _lookup_recent_best_hypothesis()
                elif isinstance(raw_claim, str):
                    recent_best = _lookup_recent_best_hypothesis()
                    if isinstance(recent_best, dict):
                        recent_text = recent_best.get("claim_text") or recent_best.get("text")
                        if recent_text == raw_claim:
                            raw_claim = recent_best
                claim_arg = _claim(raw_claim, fallback=question_text)
                support_policy = resolved_args.get("support_policy") or {"min_evidence_refs": 1}
                if isinstance(support_policy, str):
                    support_policy = {"min_evidence_refs": 1}
                evidence_chain = _chain(resolved_args.get("evidence_chain"))
                if not evidence_chain.get("evidence_refs"):
                    evidence_chain = _chain(claim_arg)
                result = executor(
                    claim_arg,
                    evidence_chain=evidence_chain,
                    support_policy=support_policy,
                    evidence_graph=graph,
                    question_text=claim_arg.get("question_text") or question_text,
                )
            elif skill_id == "commit_answer":
                verified_claim = resolved_args.get("verified_claim")
                if not isinstance(verified_claim, dict):
                    verified_claim = _lookup_recent_verified_claim() or _claim(verified_claim, fallback=question_text)
                support_chain = _chain(resolved_args.get("support_chain"))
                if not support_chain.get("evidence_refs"):
                    support_chain = _chain(verified_claim.get("supported_by_refs") or verified_claim.get("evidence_refs"))
                if not support_chain.get("evidence_refs"):
                    support_chain = _chain(_lookup_recent_evidence_refs())
                result = executor(
                    verified_claim,
                    options=resolved_args.get("options") or options or None,
                    answer_format=resolved_args.get("answer_format") or ("multiple_choice" if options else "free_text"),
                    support_chain=support_chain,
                )
            else:
                result = executor(**resolved_args)
        except (TypeError, KeyError, AttributeError, ValueError) as exc:
            trace.append({
                "step_id": step_id,
                "skill_id": skill_id,
                "ok": False,
                "failure_code": "invalid_skill_args",
                "messages": [str(exc)],
            })
            continue

        trace.append({
            "step_id": step_id,
            "skill_id": skill_id,
            "ok": result.ok,
            "failure_code": result.failure_code,
            "evidence_refs": result.evidence_refs,
            "confidence": result.confidence,
        })
        if step_id:
            step_outputs[step_id] = _augment_step_outputs(skill_id, result.outputs, result.evidence_refs)

    has_successful_commit = any(
        item.get("skill_id") == "commit_answer" and item.get("ok") for item in trace
    )
    if not has_successful_commit:
        best_hypothesis = _lookup_recent_best_hypothesis()
        support_refs = _refs(best_hypothesis) or _lookup_recent_evidence_refs()
        if isinstance(best_hypothesis, dict) and support_refs:
            claim_arg = _claim(best_hypothesis, fallback=question_text)
            evidence_chain = _chain(support_refs)
            try:
                if skill_executor is not None:
                    verify_result = skill_executor.execute(
                        "verify_claim_support",
                        args={
                            "claim": claim_arg,
                            "evidence_chain": evidence_chain,
                            "support_policy": {"min_evidence_refs": 1},
                            "question_text": question_text,
                        },
                        graph=graph,
                    )
                else:
                    verify_result = verify_claim_support(
                        claim_arg,
                        evidence_chain=evidence_chain,
                        support_policy={"min_evidence_refs": 1},
                        evidence_graph=graph,
                        question_text=question_text,
                    )
                trace.append({
                    "step_id": "auto_verify_final",
                    "skill_id": "verify_claim_support",
                    "ok": verify_result.ok,
                    "failure_code": verify_result.failure_code,
                    "evidence_refs": verify_result.evidence_refs,
                    "confidence": verify_result.confidence,
                    "auto_finalized": True,
                })
                step_outputs["auto_verify_final"] = _augment_step_outputs(
                    "verify_claim_support",
                    verify_result.outputs,
                    verify_result.evidence_refs,
                )
                verified_claim = verify_result.outputs.get("verified_claim") if isinstance(verify_result.outputs, dict) else None
                if verify_result.ok and isinstance(verified_claim, dict):
                    commit_result = commit_answer(
                        verified_claim,
                        options=options or None,
                        answer_format=question.get("answer_format") or ("multiple_choice" if options else "free_text"),
                        support_chain=_chain(verify_result.evidence_refs),
                    )
                    trace.append({
                        "step_id": "auto_commit_final",
                        "skill_id": "commit_answer",
                        "ok": commit_result.ok,
                        "failure_code": commit_result.failure_code,
                        "evidence_refs": commit_result.evidence_refs,
                        "confidence": commit_result.confidence,
                        "auto_finalized": True,
                    })
                    step_outputs["auto_commit_final"] = _augment_step_outputs(
                        "commit_answer",
                        commit_result.outputs,
                        commit_result.evidence_refs,
                    )
            except Exception as exc:
                trace.append({
                    "step_id": "auto_commit_final",
                    "skill_id": "commit_answer",
                    "ok": False,
                    "failure_code": "auto_finalize_failed",
                    "messages": [str(exc)],
                    "auto_finalized": True,
                })

    return trace, step_outputs


def build_llm_reasoning_rollout(
    example: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    *,
    client: OpenRouterClient,
    skill_executor: Any | None = None,
) -> dict[str, Any]:
    """Full L2: plan with gpt-oss then execute reasoning skills. Falls back to deterministic.

    Args:
        skill_executor: Optional SkillExecutor for LLM-backed skill dispatch.
            If provided, skills configured for LLM mode will call the model API.
    """
    question = example.get("question") or {}
    input_mode = ((example.get("available_inputs") or {}).get("mode") or "").strip()
    planner_example = example
    if input_mode == "video_only" and isinstance(question, dict) and "answer" in question:
        question = {key: value for key, value in question.items() if key != "answer"}
        planner_example = {**example, "question": question}
    task_family = example.get("task_family") or ""

    try:
        plan_response = plan_reasoning_skills(
            question=question,
            clue_memory_graph=clue_memory_graph,
            task_family=task_family,
            client=client,
        )
        reasoning_plan = plan_response.get("reasoning_plan") or []
    except Exception as exc:
        reasoning_plan = _default_multi_hop_mcq_plan()
        plan_response = {
            "reasoning_plan": reasoning_plan,
            "notes": "planner failed; using deterministic short-video multi-hop MCQ fallback",
            "planner_error": str(exc),
            "planner": "deterministic_multi_hop_fallback",
            "fallback_reason": "planner_failed",
        }

    if not reasoning_plan:
        reasoning_plan = _default_multi_hop_mcq_plan()
        plan_response = {
            **plan_response,
            "reasoning_plan": reasoning_plan,
            "planner": plan_response.get("planner") or "deterministic_multi_hop_fallback",
            "fallback_reason": plan_response.get("fallback_reason") or "empty_reasoning_plan",
        }

    trace, step_outputs = execute_reasoning_plan(
        reasoning_plan=reasoning_plan,
        clue_memory_graph=clue_memory_graph,
        question=question,
        skill_executor=skill_executor,
    )

    failed_steps = [t for t in trace if t.get("ok") is False]
    ok_steps = [t for t in trace if t.get("ok")]
    crash_steps = [t for t in failed_steps if t.get("failure_code") in ("unknown_skill_id", "invalid_skill_args")]
    has_successful_commit = any(
        item.get("skill_id") == "commit_answer" and item.get("ok") for item in trace
    )

    # --- Fault localization + repair attempt ---
    repair_result = None
    if failed_steps and not crash_steps:
        from .fault_repair import attempt_repair
        repair_result = attempt_repair(
            trace, step_outputs, reasoning_plan, clue_memory_graph, question,
            skill_executor=skill_executor,
            max_repair_attempts=1,
        )
        if repair_result.get("attempted") and repair_result.get("repaired_count", 0) > 0:
            trace = trace + repair_result["repair_trace"]
            ok_steps = [t for t in trace if t.get("ok")]
            failed_steps = [t for t in trace if t.get("ok") is False]

    if not has_successful_commit and (crash_steps or (not ok_steps and len(failed_steps) > 3)):
        rollout = make_reasoning_rollout_shell(planner_example, clue_memory_graph, rollout_source="gpt_oss_execution_failed")
        rollout.setdefault("metadata", {})
        rollout["metadata"]["llm_plan"] = plan_response
        rollout["metadata"]["llm_trace"] = trace
        rollout["metadata"]["fallback_suppressed"] = True
        rollout["metadata"]["fallback_reason"] = "too_many_failures"
        trace_refs: list[str] = []
        for item in reversed(trace):
            trace_refs.extend(str(ref) for ref in item.get("evidence_refs") or [] if ref)
            if trace_refs:
                break
        rollout["metadata"]["commonsense_repair_pack"] = _build_commonsense_repair_pack(
            question=question,
            clue_memory_graph=clue_memory_graph,
            support_refs=list(dict.fromkeys(trace_refs)),
            answerability_diagnostic=(planner_example.get("metadata") or {}).get("answerability_diagnostic") or {},
            final_label=None,
        )
        rollout["failure_reasons"] = ["planner_execution_failed"]
        if repair_result:
            rollout["metadata"]["repair"] = repair_result
        attach_initial_l2_trajectory(rollout, clue_memory_graph)
        return rollout

    answerability_diagnostic = (planner_example.get("metadata") or {}).get("answerability_diagnostic") or {}
    l2_route = answerability_diagnostic.get("l2_route")
    query_memory_commit_allowed = l2_route not in {"repair_only", "abstain_only"}

    if query_memory_commit_allowed and not any(item.get("skill_id") == "commit_answer" and item.get("ok") for item in trace):
        try:
            from ..verification.evaluate_l1_query_memory import evaluate_example

            diagnostic_example = {
                **planner_example,
                "metadata": {
                    **(planner_example.get("metadata") or {}),
                    "clue_memory_graph": clue_memory_graph,
                },
            }
            l1_report = evaluate_example(diagnostic_example, topk=8)
            qa_answerability = l1_report.get("qa_answerability") or {}
            option_scores = l1_report.get("option_scores") or []
            if qa_answerability.get("grade") == "answerable" and option_scores:
                best_option = option_scores[0]
                support_refs = best_option.get("top_refs") or []
                claim_arg = {
                    "claim_text": best_option.get("text") or best_option.get("label"),
                    "text": best_option.get("text") or best_option.get("label"),
                    "option_label": best_option.get("label"),
                    "question_text": question.get("question_text") or "",
                    "supported_by_refs": support_refs,
                }
                evidence_graph = {
                    "schema_version": clue_memory_graph.get("schema_version"),
                    "nodes": clue_memory_graph.get("nodes") or [],
                    "edges": clue_memory_graph.get("edges") or [],
                }
                evidence_chain = {"evidence_refs": support_refs, "items": []}
                if skill_executor is not None:
                    verify_result = skill_executor.execute(
                        "verify_claim_support",
                        args={
                            "claim": claim_arg,
                            "evidence_chain": evidence_chain,
                            "support_policy": {"min_evidence_refs": 1},
                            "question_text": question.get("question_text") or "",
                        },
                        graph=evidence_graph,
                    )
                else:
                    verify_result = verify_claim_support(
                        claim_arg,
                        evidence_chain=evidence_chain,
                        support_policy={"min_evidence_refs": 1},
                        evidence_graph=evidence_graph,
                        question_text=question.get("question_text") or "",
                    )
                trace.append({
                    "step_id": "query_memory_verify_final",
                    "skill_id": "verify_claim_support",
                    "ok": verify_result.ok,
                    "failure_code": verify_result.failure_code,
                    "evidence_refs": verify_result.evidence_refs,
                    "confidence": verify_result.confidence,
                    "auto_finalized": "query_memory",
                })
                step_outputs["query_memory_verify_final"] = _augment_step_outputs(
                    "verify_claim_support",
                    verify_result.outputs,
                    verify_result.evidence_refs,
                )
                verified_claim = verify_result.outputs.get("verified_claim") if isinstance(verify_result.outputs, dict) else None
                if verify_result.ok and isinstance(verified_claim, dict):
                    commit_result = commit_answer(
                        verified_claim,
                        options=question.get("options") or None,
                        answer_format=question.get("answer_format") or ("multiple_choice" if question.get("options") else "free_text"),
                        support_chain={"evidence_refs": verify_result.evidence_refs, "items": []},
                    )
                    trace.append({
                        "step_id": "query_memory_commit_final",
                        "skill_id": "commit_answer",
                        "ok": commit_result.ok,
                        "failure_code": commit_result.failure_code,
                        "evidence_refs": commit_result.evidence_refs,
                        "confidence": commit_result.confidence,
                        "auto_finalized": "query_memory",
                    })
                    step_outputs["query_memory_commit_final"] = _augment_step_outputs(
                        "commit_answer",
                        commit_result.outputs,
                        commit_result.evidence_refs,
                    )
                plan_response["query_memory_finalizer"] = {
                    "attempted": True,
                    "qa_answerability": qa_answerability,
                    "selected_option": {
                        "label": best_option.get("label"),
                        "score": best_option.get("score"),
                        "top_refs": support_refs,
                    },
                    "verified": bool(verify_result.ok),
                }
        except Exception as exc:
            plan_response["query_memory_finalizer"] = {"attempted": True, "error": str(exc)}
        ok_steps = [t for t in trace if t.get("ok")]
        failed_steps = [t for t in trace if t.get("ok") is False]
    elif not query_memory_commit_allowed:
        plan_response["query_memory_finalizer"] = {
            "attempted": False,
            "suppressed_by_l2_route": l2_route,
        }

    rollout = make_reasoning_rollout_shell(planner_example, clue_memory_graph, rollout_source="gpt_oss_reasoning_planner")
    rollout["rollout_id"] = f"skill_rollout:{example.get('example_id')}:llm_v1"

    executed_skills: list[str] = []
    prev: str | None = None
    for step_trace in trace:
        node_id = stable_id("skill", step_trace.get("skill_id"), len(rollout["nodes"]))
        rollout["nodes"].append({
            "node_id": node_id,
            "skill_id": step_trace.get("skill_id"),
            "step_id": step_trace.get("step_id"),
            "evidence_refs": step_trace.get("evidence_refs") or [],
            "status": "verified" if step_trace.get("ok") else "failed",
            "failure_code": step_trace.get("failure_code"),
            "confidence": step_trace.get("confidence", 0.0),
        })
        if prev:
            rollout["edges"].append({
                "edge_id": stable_id("edge", prev, node_id),
                "src": prev,
                "dst": node_id,
                "edge_type": "data",
            })
        prev = node_id
        if step_trace.get("skill_id"):
            executed_skills.append(step_trace["skill_id"])

    commit_traces = [
        item
        for item in trace
        if item.get("skill_id") == "commit_answer" and item.get("ok")
    ]
    last_trace = commit_traces[-1] if commit_traces else (trace[-1] if trace else {})
    last_output = step_outputs.get(last_trace.get("step_id", ""), {}) if trace else {}
    final_answer = last_output.get("final_answer")
    support_chain = last_output.get("answer_support_chain") if isinstance(last_output.get("answer_support_chain"), dict) else {}
    if not support_chain:
        support_chain = {"evidence_refs": last_output.get("evidence_refs") or [], "items": []}
    support_refs = support_chain.get("evidence_refs") or last_output.get("evidence_refs") or []
    commit_ok = bool(last_trace.get("ok") and final_answer and support_refs)
    options = question.get("options") or []
    final_label = final_answer
    final_text = str(final_answer) if final_answer is not None else ""
    if isinstance(final_answer, dict):
        final_label = final_answer.get("label") or final_answer.get("answer")
        final_text = str(final_answer.get("text") or final_label or "")
    elif options:
        answer_string = str(final_answer or "").strip()
        by_label = {str(option.get("label", "")).strip(): option for option in options}
        if answer_string in by_label:
            final_label = answer_string
            final_text = str(by_label[answer_string].get("text") or answer_string)
        else:
            for option in options:
                if answer_string and answer_string == str(option.get("text", "")).strip():
                    final_label = option.get("label") or answer_string
                    final_text = answer_string
                    break

    query_memory_consistency: dict[str, Any] | None = None
    try:
        from ..verification.evaluate_l1_query_memory import evaluate_example

        diagnostic_example = {
            **planner_example,
            "metadata": {
                **(planner_example.get("metadata") or {}),
                "clue_memory_graph": clue_memory_graph,
            },
        }
        l1_report = evaluate_example(diagnostic_example, topk=8)
        qa_answerability = l1_report.get("qa_answerability") or {}
        option_scores = l1_report.get("option_scores") or []
        if qa_answerability.get("grade") == "answerable" and option_scores:
            l1_best = option_scores[0]
            l1_label = str(l1_best.get("label") or "")
            l1_margin = float(qa_answerability.get("option_margin") or 0.0)
            if l1_label and final_label and str(final_label) != l1_label and l1_margin >= 0.75:
                l1_refs = l1_best.get("top_refs") or []
                l1_claim = {
                    "claim_text": l1_best.get("text") or l1_label,
                    "text": l1_best.get("text") or l1_label,
                    "option_label": l1_label,
                    "question_text": question.get("question_text") or "",
                    "supported_by_refs": l1_refs,
                }
                evidence_graph = {
                    "schema_version": clue_memory_graph.get("schema_version"),
                    "nodes": clue_memory_graph.get("nodes") or [],
                    "edges": clue_memory_graph.get("edges") or [],
                }
                evidence_chain = {"evidence_refs": l1_refs, "items": []}
                if skill_executor is not None:
                    verify_result = skill_executor.execute(
                        "verify_claim_support",
                        args={
                            "claim": l1_claim,
                            "evidence_chain": evidence_chain,
                            "support_policy": {"min_evidence_refs": 1},
                            "question_text": question.get("question_text") or "",
                        },
                        graph=evidence_graph,
                    )
                else:
                    verify_result = verify_claim_support(
                        l1_claim,
                        evidence_chain=evidence_chain,
                        support_policy={"min_evidence_refs": 1},
                        evidence_graph=evidence_graph,
                        question_text=question.get("question_text") or "",
                    )
                query_memory_consistency = {
                    "conflict": True,
                    "l2_label": final_label,
                    "l1_label": l1_label,
                    "l1_margin": l1_margin,
                    "verified_l1_override": bool(verify_result.ok),
                    "l1_refs": l1_refs,
                }
                if verify_result.ok:
                    final_label = l1_label
                    final_text = str(l1_best.get("text") or l1_label)
                    support_refs = verify_result.evidence_refs
                    support_chain = {"evidence_refs": support_refs, "items": []}
                    final_answer = {"label": final_label, "text": final_text}
                    commit_ok = bool(support_refs)
                    last_output = {**last_output, "confidence": verify_result.confidence}
                else:
                    commit_ok = False
            elif l1_label:
                query_memory_consistency = {
                    "conflict": bool(final_label and str(final_label) != l1_label),
                    "l2_label": final_label,
                    "l1_label": l1_label,
                    "l1_margin": l1_margin,
                }
    except Exception as exc:
        query_memory_consistency = {"error": str(exc)}

    unique_support_refs = list(dict.fromkeys(str(ref) for ref in support_refs if ref))
    video_regime = (planner_example.get("metadata") or {}).get("video_regime") or rollout.get("video_regime")
    is_multiple_choice = bool(question.get("options"))
    answerability_diagnostic = (planner_example.get("metadata") or {}).get("answerability_diagnostic") or {}
    missing_requirements = answerability_diagnostic.get("missing_requirements") or []
    min_support_refs = 2 if video_regime == "long" and is_multiple_choice else 1
    acceptance_failures: list[str] = []
    if commit_ok and len(unique_support_refs) < min_support_refs:
        commit_ok = False
        acceptance_failures.append("insufficient_support_refs")
    if commit_ok and video_regime == "long" and len(failed_steps) > len(ok_steps):
        commit_ok = False
        acceptance_failures.append("unstable_l2_trace")

    strong_min_refs = 4 if video_regime == "long" else 3
    trace_total = len(ok_steps) + len(failed_steps)
    trace_fail_ratio = (len(failed_steps) / trace_total) if trace_total else 0.0
    strong_trace_ok = trace_fail_ratio <= 0.2
    strong_accept = (
        commit_ok
        and len(unique_support_refs) >= strong_min_refs
        and strong_trace_ok
        and not missing_requirements
    )
    acceptance_status = "accepted_strong" if strong_accept else ("accepted_weak" if commit_ok else "rejected")
    verifier_reason = (
        "strong_verified_evidence_pack"
        if strong_accept
        else ("weak_verified_evidence_pack" if commit_ok else (acceptance_failures[0] if acceptance_failures else ("no_supported_final_answer" if final_answer else "no_final_answer")))
    )
    verified_evidence_pack = {
        "claim_text": final_text,
        "final_label": final_label,
        "support_refs": unique_support_refs if commit_ok else [],
        "support_ref_count": len(unique_support_refs) if commit_ok else 0,
        "min_support_refs": min_support_refs,
        "strong_min_refs": strong_min_refs,
        "trace_ok": len(ok_steps),
        "trace_fail": len(failed_steps),
        "missing_requirements": missing_requirements,
        "verifier_reason": verifier_reason,
    }
    commonsense_repair_pack = None
    if (not strong_accept) and (not commit_ok or missing_requirements or video_regime == "long"):
        trace_refs: list[str] = []
        for item in reversed(trace):
            trace_refs.extend(str(ref) for ref in item.get("evidence_refs") or [] if ref)
            if trace_refs:
                break
        commonsense_repair_pack = _build_commonsense_repair_pack(
            question=question,
            clue_memory_graph=clue_memory_graph,
            support_refs=unique_support_refs or list(dict.fromkeys(trace_refs)),
            answerability_diagnostic=answerability_diagnostic,
            final_label=final_label,
        )

    rollout["claims"] = [{
        "claim_id": stable_id("claim", final_text),
        "text": final_text,
        "claim_status": "verified" if commit_ok else "insufficient",
        "supported_by_refs": unique_support_refs if commit_ok else [],
    }]
    rollout["answer_support_chain"] = [support_chain] if commit_ok else []
    rollout["final_answer"] = {
        "label": final_label,
        "text": final_text,
        "confidence": last_output.get("confidence", 0.0) if commit_ok else 0.0,
    }
    rollout["acceptance_status"] = acceptance_status
    rollout["failure_reasons"] = [] if commit_ok else (
        acceptance_failures or ["no_supported_final_answer" if final_answer else "no_final_answer"]
    )
    rollout["verified_evidence_pack"] = verified_evidence_pack
    rollout["metadata"] = {
        "executed_skill_ids": list(dict.fromkeys(executed_skills)),
        "executed_skill_count": len(set(executed_skills)),
        "expected_reasoning_skill_count": len(REASONING_SKILL_IDS),
        "llm_plan": plan_response,
        "llm_trace_ok": len(ok_steps),
        "llm_trace_fail": len(failed_steps),
        "acceptance_status_detail": {
            "status": acceptance_status,
            "reason": verifier_reason,
            "strong_trace_ok": strong_trace_ok,
            "trace_fail_ratio": round(trace_fail_ratio, 4),
            "strong_min_refs": strong_min_refs,
        },
    }
    if repair_result:
        rollout["metadata"]["repair"] = repair_result
    if query_memory_consistency:
        rollout["metadata"]["query_memory_consistency"] = query_memory_consistency
    if commonsense_repair_pack:
        rollout["metadata"]["commonsense_repair_pack"] = commonsense_repair_pack
    attach_initial_l2_trajectory(rollout, clue_memory_graph)
    return rollout
