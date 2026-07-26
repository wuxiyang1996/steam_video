"""Unified runtime verifier: hard acceptance gates over both graph layers.

These are NOT planner actions. They run after rollout construction and produce
verifier_summary, failure_reasons, and acceptance_status fields.

Atomic verification skills (verify_claim_support, verify_temporal_social_consistency,
score_hypothesis_support, compare_hypotheses) live in the L2 skill ontology and are
planner-selectable. This module handles system-level invariants only.
"""

from __future__ import annotations

from typing import Any

from atomic_skills.common import normalize_time_span

HIDDEN_SOURCE_TYPES = frozenset({
    "segment_description",
    "inference_shot",
    "key_relationship",
    "clue_interval",
    "clue_clip",
    "reasoning_process_step",
    "video_summary",
    "qa_answer",
})


def _check_schema_valid(clue_graph: dict[str, Any], rollout: dict[str, Any]) -> list[str]:
    """Basic structural checks without jsonschema dependency."""
    errors: list[str] = []
    if not clue_graph.get("graph_id"):
        errors.append("clue_graph missing graph_id")
    if clue_graph.get("layer") != "clue_memory":
        errors.append("clue_graph missing layer=clue_memory")
    if rollout.get("layer") != "reasoning":
        errors.append("rollout missing layer=reasoning")
    if not rollout.get("rollout_id"):
        errors.append("rollout missing rollout_id")
    ref = rollout.get("clue_memory_ref") or {}
    if ref.get("graph_id") != clue_graph.get("graph_id"):
        errors.append("rollout clue_memory_ref.graph_id mismatch")
    return errors


def _check_evidence_ref_existence(rollout: dict[str, Any], clue_graph: dict[str, Any]) -> list[str]:
    """Every evidence_ref in rollout nodes must exist in the clue graph."""
    clue_node_ids = {n.get("node_id") for n in clue_graph.get("nodes", []) if n.get("node_id")}
    errors: list[str] = []
    for node in rollout.get("nodes", []):
        for ref in node.get("evidence_refs") or []:
            if ref and ref not in clue_node_ids:
                errors.append(f"rollout node {node.get('node_id')} refs missing L1 node: {ref}")
                break
    for claim in rollout.get("claims") or []:
        for ref in claim.get("supported_by_refs") or []:
            if ref and ref not in clue_node_ids:
                errors.append(f"claim {claim.get('claim_id')} refs missing L1 node: {ref}")
                break
    return errors


def _check_hidden_supervision_leakage(
    clue_graph: dict[str, Any],
    rollout: dict[str, Any],
    mode: str,
) -> list[str]:
    """In video_only mode, no hidden supervision sources may appear."""
    if mode != "video_only":
        return []
    errors: list[str] = []
    for node in clue_graph.get("nodes", []):
        if node.get("source_type") in HIDDEN_SOURCE_TYPES:
            errors.append(f"L1 leaked hidden source: {node.get('source_type')} ({node.get('node_id')})")
            break
    for node in rollout.get("nodes", []):
        for ref in node.get("evidence_refs") or []:
            clue_node = next((n for n in clue_graph.get("nodes", []) if n.get("node_id") == ref), None)
            if clue_node and clue_node.get("source_type") in HIDDEN_SOURCE_TYPES:
                errors.append(f"L2 node {node.get('node_id')} cites hidden source via {ref}")
                break
    return errors


def _check_streaming_visibility(clue_graph: dict[str, Any]) -> list[str]:
    """In streaming regime, all nodes must respect observation_end_s."""
    obs_end = clue_graph.get("observation_end_s")
    if obs_end is None:
        if clue_graph.get("video_regime") == "streaming":
            return ["streaming clue_graph missing observation_end_s"]
        return []
    errors: list[str] = []
    for node in clue_graph.get("nodes", []):
        span = normalize_time_span(node.get("time_span"))
        if span and span["end_s"] > obs_end + 1e-6:
            errors.append(
                f"node {node.get('node_id')} time_span.end_s={span['end_s']:.1f} "
                f"exceeds observation_end_s={obs_end:.1f}"
            )
            break
    return errors


def _check_retrieval_not_support(rollout: dict[str, Any]) -> list[str]:
    """Retrieval scores alone cannot serve as answer support."""
    errors: list[str] = []
    for claim in rollout.get("claims") or []:
        refs = claim.get("supported_by_refs") or []
        if not refs and claim.get("claim_status") == "verified":
            errors.append(f"claim {claim.get('claim_id')} verified without supported_by_refs")
    answer_chain = rollout.get("answer_support_chain") or []
    for entry in answer_chain:
        if not entry.get("evidence_refs"):
            errors.append("answer_support_chain entry has no evidence_refs")
            break
    return errors


def _check_commit_has_evidence(rollout: dict[str, Any]) -> list[str]:
    """Every committed answer must trace to evidence."""
    errors: list[str] = []
    final = rollout.get("final_answer") or {}
    if final.get("label") and not (rollout.get("claims") or []):
        errors.append("final_answer present but no claims recorded")
    for claim in rollout.get("claims") or []:
        if claim.get("claim_status") == "verified" and not claim.get("supported_by_refs"):
            errors.append(f"verified claim {claim.get('claim_id')} has empty supported_by_refs")
    return errors


def verify_rollout(
    clue_graph: dict[str, Any],
    rollout: dict[str, Any],
    *,
    mode: str = "video_only",
) -> dict[str, Any]:
    """Run all runtime verifier invariants and return a structured result.

    Returns a dict with:
      - passed: bool
      - verifier_summary: dict of individual check results
      - failure_reasons: list of error strings
      - acceptance_status: "accepted" | "accepted_weak" | "rejected"
    """
    schema_errors = _check_schema_valid(clue_graph, rollout)
    ref_errors = _check_evidence_ref_existence(rollout, clue_graph)
    leakage_errors = _check_hidden_supervision_leakage(clue_graph, rollout, mode)
    streaming_errors = _check_streaming_visibility(clue_graph)
    retrieval_errors = _check_retrieval_not_support(rollout)
    commit_errors = _check_commit_has_evidence(rollout)

    all_errors = schema_errors + ref_errors + leakage_errors + streaming_errors + retrieval_errors + commit_errors

    verifier_summary = {
        "schema_valid": not schema_errors,
        "evidence_refs_exist": not ref_errors,
        "no_hidden_supervision_leakage": not leakage_errors,
        "streaming_visibility_ok": not streaming_errors,
        "retrieval_not_used_as_support": not retrieval_errors,
        "all_commits_have_evidence": not commit_errors,
    }

    hard_errors = schema_errors + ref_errors + leakage_errors + retrieval_errors + commit_errors

    if not all_errors:
        acceptance = "accepted"
    elif not hard_errors:
        acceptance = "accepted_weak"
    else:
        acceptance = "rejected"

    return {
        "passed": not all_errors,
        "verifier_summary": verifier_summary,
        "failure_reasons": all_errors,
        "acceptance_status": acceptance,
    }
