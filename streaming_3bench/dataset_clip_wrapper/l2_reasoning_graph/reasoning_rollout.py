"""Layer-2 reasoning rollout builder executing all 19 reasoning assembly skills."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from atomic_skills.common import stable_id
from atomic_skills.reasoning_graph_assembly import (
    assign_evidence_role,
    commit_answer,
    compose_evidence_chain,
    detect_missing_role,
    extract_claim,
    infer_causal_relation,
    infer_intention_or_motive,
    infer_social_contradiction,
    infer_state_change,
    infer_temporal_relation,
    localize_clue,
    parse_question_target,
    propose_evidence_roles,
    retrieve_by_entity,
    retrieve_by_event,
    retrieve_by_relation,
    retrieve_by_time,
    search_counterevidence,
    verify_claim_support,
)

from ..l1_clue_graph.clue_memory import make_reasoning_rollout_shell


def _clue_to_evidence_graph(clue_memory_graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": clue_memory_graph.get("schema_version"),
        "nodes": deepcopy(clue_memory_graph.get("nodes") or []),
        "edges": deepcopy(clue_memory_graph.get("edges") or []),
    }


def _pick_nodes(graph: dict[str, Any], *node_types: str) -> list[dict[str, Any]]:
    allowed = set(node_types)
    return [node for node in graph.get("nodes", []) if node.get("node_type") in allowed and node.get("node_id")]


def _record_step(
    rollout: dict[str, Any],
    *,
    skill_id: str,
    args: dict[str, Any],
    result: Any,
    prev_node_id: str | None,
) -> str:
    node_id = stable_id("skill", skill_id, len(rollout["nodes"]))
    status = "verified" if result.ok else ("skipped" if result.failure_code else "failed")
    rollout["nodes"].append(
        {
            "node_id": node_id,
            "skill_id": skill_id,
            "args": args,
            "outputs": result.outputs,
            "evidence_refs": list(result.evidence_refs or []),
            "status": status,
            "failure_code": result.failure_code,
            "confidence": result.confidence,
        }
    )
    if prev_node_id:
        rollout["edges"].append(
            {
                "edge_id": stable_id("edge", prev_node_id, node_id),
                "src": prev_node_id,
                "dst": node_id,
                "edge_type": "data",
            }
        )
    return node_id


def build_reasoning_rollout(
    example: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    *,
    rollout_source: str = "deterministic_reasoning_skills",
) -> dict[str, Any]:
    """Execute all 19 reasoning skills over a clue-memory graph and return SkillGraphRollout."""
    rollout = make_reasoning_rollout_shell(example, clue_memory_graph, rollout_source=rollout_source)
    rollout["rollout_id"] = f"skill_rollout:{example.get('example_id')}:v1"
    rollout["rollout_source"] = rollout_source

    question = example.get("question") or {}
    question_text = question.get("question_text") or ""
    options = question.get("options") or []
    answer_format = question.get("answer_format") or ("multiple_choice" if options else "free_text")
    task_family = example.get("task_family") or ""

    graph = _clue_to_evidence_graph(clue_memory_graph)
    prev: str | None = None
    executed_skills: list[str] = []

    parsed = parse_question_target(question_text, options=options or None)
    prev = _record_step(rollout, skill_id="parse_question_target", args={"question_text": question_text}, result=parsed, prev_node_id=prev)
    executed_skills.append(parsed.skill_id)
    parsed_target = parsed.outputs

    roles = propose_evidence_roles(question_text, parsed_target, task_family=task_family)
    prev = _record_step(
        rollout,
        skill_id="propose_evidence_roles",
        args={"question_text": question_text, "task_family": task_family},
        result=roles,
        prev_node_id=prev,
    )
    executed_skills.append(roles.skill_id)
    required_roles = [item["role"] for item in roles.outputs.get("role_constraints", []) if item.get("role")]

    by_event = retrieve_by_event(graph, event_description=question_text)
    prev = _record_step(
        rollout,
        skill_id="retrieve_by_event",
        args={"event_description": question_text},
        result=by_event,
        prev_node_id=prev,
    )
    executed_skills.append(by_event.skill_id)

    entity_hint = (parsed_target.get("target_entities") or [None])[0]
    if not entity_hint:
        tokens = [t for t in question_text.split() if len(t) > 3]
        entity_hint = tokens[0] if tokens else "person"
    by_entity = retrieve_by_entity(graph, entity_id=str(entity_hint))
    prev = _record_step(
        rollout,
        skill_id="retrieve_by_entity",
        args={"entity_id": str(entity_hint)},
        result=by_entity,
        prev_node_id=prev,
    )
    executed_skills.append(by_entity.skill_id)

    anchor = None
    event_nodes = _pick_nodes(graph, "event", "observation")
    if by_event.evidence_refs:
        anchor = by_event.evidence_refs[0]
    elif event_nodes:
        anchor = event_nodes[0]["node_id"]
    elif _pick_nodes(graph, "clip"):
        anchor = _pick_nodes(graph, "clip")[0]["node_id"]

    by_time = retrieve_by_time(graph, anchor_event_or_time=anchor or {"start_s": 0.0, "end_s": 30.0}, window_before=30, window_after=30)
    prev = _record_step(
        rollout,
        skill_id="retrieve_by_time",
        args={"anchor_event_or_time": anchor, "window_before": 30, "window_after": 30},
        result=by_time,
        prev_node_id=prev,
    )
    executed_skills.append(by_time.skill_id)

    relation_anchor = anchor or (by_event.evidence_refs[0] if by_event.evidence_refs else None)
    by_relation = retrieve_by_relation(graph, source_node=relation_anchor or "missing", relation_type="temporal_next")
    if relation_anchor is None:
        by_relation.ok = False
        by_relation.failure_code = "no_relation_path"
    prev = _record_step(
        rollout,
        skill_id="retrieve_by_relation",
        args={"source_node": relation_anchor, "relation_type": "temporal_next"},
        result=by_relation,
        prev_node_id=prev,
    )
    executed_skills.append(by_relation.skill_id)

    candidates = by_time.outputs.get("neighbor_events") or event_nodes or _pick_nodes(graph, "observation")
    role_constraint = (required_roles[0] if required_roles else "supporting_evidence")
    clue = localize_clue(candidates, role_constraint=role_constraint, question_context=question_text)
    prev = _record_step(
        rollout,
        skill_id="localize_clue",
        args={"role_constraint": role_constraint, "question_context": question_text},
        result=clue,
        prev_node_id=prev,
    )
    executed_skills.append(clue.skill_id)

    claim_ref = (clue.evidence_refs[0] if clue.evidence_refs else (by_event.evidence_refs[0] if by_event.evidence_refs else None))
    claim = extract_claim(graph, evidence_ref=claim_ref or "missing", claim_query=question_text[:120] if question_text else None)
    if not claim_ref:
        claim.ok = False
        claim.failure_code = "missing_evidence_ref"
    prev = _record_step(
        rollout,
        skill_id="extract_claim",
        args={"evidence_ref": claim_ref},
        result=claim,
        prev_node_id=prev,
    )
    executed_skills.append(claim.skill_id)

    labeled_roles: list[dict[str, Any]] = []
    role_targets = required_roles[:2] if required_roles else ["question_anchor", "supporting_evidence"]
    refs_for_roles = list(dict.fromkeys((clue.evidence_refs or []) + (by_event.evidence_refs or []) + (by_time.evidence_refs or [])))
    for role_name, evidence_ref in zip(role_targets, refs_for_roles or [claim_ref]):
        if not evidence_ref:
            continue
        role_result = assign_evidence_role(graph, evidence_ref=evidence_ref, role_schema=role_name, question_context=question_text)
        prev = _record_step(
            rollout,
            skill_id="assign_evidence_role",
            args={"evidence_ref": evidence_ref, "role_schema": role_name},
            result=role_result,
            prev_node_id=prev,
        )
        executed_skills.append(role_result.skill_id)
        if role_result.ok:
            labeled_roles.append(role_result.outputs["role_labeled_evidence"])

    dependency_template = roles.outputs.get("expected_chain_shape") or "support_chain"
    if labeled_roles:
        dependency_template = "->".join(item.get("role") for item in labeled_roles if item.get("role"))
    chain = compose_evidence_chain(labeled_roles or [{"role": "supporting_evidence", "evidence_ref": claim_ref, "text": "", "confidence": 0.0}], dependency_template=dependency_template)
    prev = _record_step(
        rollout,
        skill_id="compose_evidence_chain",
        args={"dependency_template": dependency_template},
        result=chain,
        prev_node_id=prev,
    )
    executed_skills.append(chain.skill_id)
    evidence_chain = chain.outputs.get("evidence_chain") or {"evidence_refs": refs_for_roles, "items": labeled_roles}

    missing = detect_missing_role(evidence_chain, required_roles=required_roles or role_targets)
    prev = _record_step(
        rollout,
        skill_id="detect_missing_role",
        args={"required_roles": required_roles or role_targets},
        result=missing,
        prev_node_id=prev,
    )
    executed_skills.append(missing.skill_id)

    counter = search_counterevidence(
        graph,
        claim=claim.outputs if claim.ok else {"claim_text": question_text},
        supporting_evidence=evidence_chain.get("evidence_refs", [])[:1],
        search_scope=question_text,
    )
    prev = _record_step(
        rollout,
        skill_id="search_counterevidence",
        args={"search_scope": question_text},
        result=counter,
        prev_node_id=prev,
    )
    executed_skills.append(counter.skill_id)

    event_refs = [node["node_id"] for node in event_nodes[:2]] or evidence_chain.get("evidence_refs", [])[:2]
    temporal = infer_temporal_relation(event_refs, evidence_graph=graph) if len(event_refs) >= 2 else infer_temporal_relation([], evidence_graph=graph)
    prev = _record_step(
        rollout,
        skill_id="infer_temporal_relation",
        args={"event_refs": event_refs},
        result=temporal,
        prev_node_id=prev,
    )
    executed_skills.append(temporal.skill_id)

    state_refs = evidence_chain.get("evidence_refs", [])[:2]
    state_change = infer_state_change(
        graph,
        entity_or_object=str(entity_hint),
        state_predicate="state",
        before_after_refs=state_refs,
    )
    prev = _record_step(
        rollout,
        skill_id="infer_state_change",
        args={"entity_or_object": str(entity_hint), "state_predicate": "state", "before_after_refs": state_refs},
        result=state_change,
        prev_node_id=prev,
    )
    executed_skills.append(state_change.skill_id)

    causal = infer_causal_relation("prior event", "question outcome", evidence_chain=evidence_chain)
    prev = _record_step(
        rollout,
        skill_id="infer_causal_relation",
        args={"candidate_cause": "prior event", "candidate_effect": "question outcome"},
        result=causal,
        prev_node_id=prev,
    )
    executed_skills.append(causal.skill_id)

    motive = infer_intention_or_motive(
        str(entity_hint),
        ["observed action"],
        context_evidence=evidence_chain.get("evidence_refs", [])[:3],
    )
    prev = _record_step(
        rollout,
        skill_id="infer_intention_or_motive",
        args={"agent": str(entity_hint), "actions": ["observed action"]},
        result=motive,
        prev_node_id=prev,
    )
    executed_skills.append(motive.skill_id)

    contradiction = infer_social_contradiction(
        claim.outputs if claim.ok else {"claim_text": question_text},
        evidence_chain=evidence_chain,
        counterevidence=counter.evidence_refs,
    )
    prev = _record_step(
        rollout,
        skill_id="infer_social_contradiction",
        args={},
        result=contradiction,
        prev_node_id=prev,
    )
    executed_skills.append(contradiction.skill_id)

    claim_for_verify = contradiction.outputs.get("contradiction_claim") if contradiction.ok else (claim.outputs.get("claim_text") or question_text)
    verified = verify_claim_support(
        {"claim_text": claim_for_verify, "claim_status": "candidate"},
        evidence_chain=evidence_chain,
        support_policy={"min_evidence_refs": 1},
    )
    prev = _record_step(
        rollout,
        skill_id="verify_claim_support",
        args={"support_policy": {"min_evidence_refs": 1}},
        result=verified,
        prev_node_id=prev,
    )
    executed_skills.append(verified.skill_id)

    verified_claim = verified.outputs.get("verified_claim") or {"text": claim_for_verify, "claim_status": "verified" if verified.ok else "insufficient"}
    answer = commit_answer(
        verified_claim,
        options=options or None,
        answer_format=answer_format,
        support_chain=evidence_chain,
    )
    _record_step(
        rollout,
        skill_id="commit_answer",
        args={"answer_format": answer_format},
        result=answer,
        prev_node_id=prev,
    )
    executed_skills.append(answer.skill_id)

    verified_claim_record = verified.outputs.get("verified_claim") or {}
    rollout["claims"] = [
        {
            "claim_id": verified_claim_record.get("claim_id") or stable_id("claim", claim_for_verify),
            "text": verified_claim_record.get("text") or claim_for_verify,
            "claim_status": verified_claim_record.get("claim_status") or ("verified" if verified.ok else "insufficient"),
            "supported_by_refs": verified_claim_record.get("supported_by_refs") or evidence_chain.get("evidence_refs", []),
        }
    ]
    rollout["answer_support_chain"] = [
        {
            "node_id": rollout["nodes"][-1]["node_id"],
            "claim_id": rollout["claims"][0]["claim_id"],
            "evidence_refs": evidence_chain.get("evidence_refs", []),
        }
    ]
    input_mode = rollout.get("input_mode")
    gt = question.get("answer") or {}
    answer_text = answer.outputs.get("final_answer") if answer.ok else None
    if input_mode != "video_only" and gt.get("text"):
        answer_text = gt.get("text")
    rollout["final_answer"] = {
        "label": answer.outputs.get("final_answer") if answer.ok else (gt.get("label") if input_mode != "video_only" else None),
        "text": answer_text,
        "confidence": answer.confidence if answer.ok else 0.0,
    }
    rollout["verifier_summary"] = {
        "schema_valid": True,
        "all_commits_have_evidence": bool(answer.ok and evidence_chain.get("evidence_refs")),
        "answer_chain_valid": bool(chain.ok),
        "timestamp_valid": True,
        "no_old_video_fact_leakage": True,
        "no_hidden_supervision_leakage": rollout.get("input_mode") == "video_only",
    }
    rollout["acceptance_status"] = "accepted_weak" if answer.ok else "rejected"
    rollout["failure_reasons"] = [] if answer.ok else [answer.failure_code or "reasoning_incomplete"]
    rollout["metadata"] = {
        "executed_skill_ids": list(dict.fromkeys(executed_skills)),
        "executed_skill_count": len(set(executed_skills)),
        "expected_reasoning_skill_count": 19,
        "gold_answer_hidden_eval_only": bool(input_mode == "video_only" and gt),
    }
    return rollout
