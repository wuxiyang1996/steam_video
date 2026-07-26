"""POMDP-compatible trajectory helpers for bounded L2 repair loops."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from atomic_skills.common import stable_id

L2_TRAJECTORY_SCHEMA = "video-skills-relaunch/l2-trajectory-v0.1"
REPAIR_SUBGRAPH_SCHEMA = "video-skills-relaunch/l2-repair-subgraph-v0.1"

STRONG_TERMINAL_STATUSES = {"accepted_strong", "resolved_strong", "accepted_bridge"}


def graph_snapshot(graph: dict[str, Any] | None) -> dict[str, Any]:
    """Compact state snapshot; keep full graph content out of recursive logs."""
    graph = graph or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    return {
        "graph_id": graph.get("graph_id"),
        "layer": graph.get("layer"),
        "dataset": graph.get("dataset"),
        "example_id": graph.get("example_id"),
        "video_regime": graph.get("video_regime"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "index_stats": deepcopy(graph.get("index_stats") or {}),
        "observation_end_s": graph.get("observation_end_s"),
    }


def rollout_snapshot(rollout: dict[str, Any] | None) -> dict[str, Any]:
    rollout = rollout or {}
    return {
        "rollout_id": rollout.get("rollout_id"),
        "layer": rollout.get("layer"),
        "node_count": len(rollout.get("nodes") or []),
        "edge_count": len(rollout.get("edges") or []),
        "claim_count": len(rollout.get("claims") or []),
        "acceptance_status": rollout.get("acceptance_status"),
        "failure_reasons": deepcopy(rollout.get("failure_reasons") or []),
        "final_answer": deepcopy(rollout.get("final_answer") or {}),
        "verified_evidence_pack": deepcopy(rollout.get("verified_evidence_pack") or {}),
    }


def reward_proxy_from_status(status: str | None, *, repair_needed: bool = False) -> dict[str, Any]:
    status = str(status or "")
    if status in {"accepted_strong", "resolved_strong"}:
        value = 1.0
    elif status == "accepted_bridge":
        value = 0.65
    elif repair_needed:
        value = -0.25
    elif status in {"rejected", "needs_more_evidence"}:
        value = -0.5
    else:
        value = 0.0
    return {
        "value": value,
        "components": {
            "strong_or_resolved": status in {"accepted_strong", "resolved_strong"},
            "accepted_bridge": status == "accepted_bridge",
            "repair_needed": repair_needed,
            "unsupported_commit_penalty": status in {"accepted_weak", "rejected", "needs_more_evidence"},
        },
    }


def initial_l2_round(
    rollout: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    *,
    max_repair_rounds: int = 2,
) -> dict[str, Any]:
    """Build the round-0 trace from an initial L2 reasoning rollout."""
    status = str(rollout.get("acceptance_status") or "")
    repair_needed = status != "accepted_strong"
    metadata = rollout.get("metadata") or {}
    plan = metadata.get("llm_plan") or {}
    return {
        "round_index": 0,
        "round_type": "initial_l2_reasoning",
        "state_snapshot": {
            "l1": graph_snapshot(clue_memory_graph),
            "l2": rollout_snapshot(rollout),
        },
        "action": {
            "action_type": "call_gptoss_reasoning_planner",
            "tool_backend": plan.get("model") or "unknown",
            "planner": plan.get("planner"),
            "planner_attempt": plan.get("planner_attempt"),
            "max_repair_rounds": max_repair_rounds,
        },
        "observation_summary": {
            "executed_skill_count": metadata.get("executed_skill_count"),
            "llm_trace_ok": metadata.get("llm_trace_ok"),
            "llm_trace_fail": metadata.get("llm_trace_fail"),
            "acceptance_status": status,
            "failure_reasons": deepcopy(rollout.get("failure_reasons") or []),
        },
        "graph_delta": {
            "l2_nodes_added": len(rollout.get("nodes") or []),
            "l2_edges_added": len(rollout.get("edges") or []),
            "claims_added": len(rollout.get("claims") or []),
        },
        "verifier_signal": {
            "status": status,
            "reason": (metadata.get("acceptance_status_detail") or {}).get("reason"),
            "verified_evidence_pack": deepcopy(rollout.get("verified_evidence_pack") or {}),
        },
        "reward_proxy": reward_proxy_from_status(status, repair_needed=repair_needed),
        "terminal_status": "repair_requested" if repair_needed else "accepted_strong",
    }


def attach_initial_l2_trajectory(
    rollout: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    *,
    max_repair_rounds: int = 2,
) -> dict[str, Any]:
    """Attach a POMDP-compatible trajectory shell to an L2 rollout in-place."""
    trajectory = {
        "schema_version": L2_TRAJECTORY_SCHEMA,
        "process_model": "pomdp_compatible_bounded_recursive_graph_agent",
        "is_training_mdp": False,
        "max_repair_rounds": max_repair_rounds,
        "state_definition": [
            "question",
            "current_l1_clue_graph_snapshot",
            "current_l2_reasoning_graph_snapshot",
            "verifier_status",
            "budget_state",
        ],
        "action_definition": [
            "llm_reasoning_plan",
            "evidence_selector",
            "repair_clip_schema",
            "l1_patch",
            "claim_verification",
            "objective_bridge",
            "commit_or_abstain",
        ],
        "rounds": [initial_l2_round(rollout, clue_memory_graph, max_repair_rounds=max_repair_rounds)],
    }
    rollout.setdefault("metadata", {})["l2_trajectory"] = trajectory
    return rollout


def repair_artifacts_to_subgraph(
    *,
    plan: dict[str, Any],
    patch: dict[str, Any],
    l2: dict[str, Any],
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert repair protocol artifacts into explicit L2 repair graph nodes."""
    report = report or {}
    example_id = plan.get("example_id") or l2.get("example_id") or report.get("example_id")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def add_node(node_type: str, payload: dict[str, Any]) -> str:
        node_id = stable_id("l2.repair", example_id, node_type, len(nodes), payload)
        nodes.append({"node_id": node_id, "node_type": node_type, **payload})
        return node_id

    def add_edge(src: str, dst: str, edge_type: str) -> None:
        edges.append({"edge_id": stable_id("l2.repair.edge", src, dst, edge_type), "src": src, "dst": dst, "edge_type": edge_type})

    gap_id = add_node(
        "l2_gap_diagnosis",
        {
            "gap_types": deepcopy(plan.get("gap_types") or report.get("gap_types") or []),
            "failure_type": report.get("failure_type"),
            "recommended_next_action": report.get("recommended_next_action"),
        },
    )
    plan_id = add_node(
        "repair_plan",
        {
            "strategy": plan.get("strategy"),
            "repair_mode": plan.get("repair_mode"),
            "selection_mode": (plan.get("span_selection") or {}).get("selection_mode") or report.get("selection_mode"),
            "selected_coarse_indices": deepcopy(report.get("selected_coarse_indices") or []),
            "span_count": len(plan.get("spans") or []),
            "clue_need_spec": deepcopy(plan.get("clue_need_spec") or {}),
        },
    )
    add_edge(gap_id, plan_id, "requests_repair")

    patch_id = add_node(
        "l1_patch",
        {
            "patch_counts": deepcopy(patch.get("counts") or report.get("patch_counts") or {}),
            "negative_target_evidence_nodes": report.get("negative_target_evidence_nodes", 0),
        },
    )
    add_edge(plan_id, patch_id, "patches_l1")

    selector_id = add_node(
        "option_evidence_selector",
        {
            "selector": deepcopy(l2.get("option_evidence_selector") or report.get("option_evidence_selector") or {}),
            "option_evidence_packs": deepcopy(report.get("option_evidence_packs") or []),
        },
    )
    add_edge(patch_id, selector_id, "selects_evidence")

    verifier_id = add_node(
        "option_verifier",
        {
            "backend": l2.get("backend") or report.get("verifier_backend"),
            "repair_status": l2.get("repair_status") or report.get("repair_status"),
            "best_option": deepcopy(l2.get("best_option") or report.get("best_option") or {}),
            "option_verifier_policy": deepcopy(l2.get("option_verifier_policy") or report.get("option_verifier_policy") or {}),
        },
    )
    add_edge(selector_id, verifier_id, "verifies_options")

    last_id = verifier_id
    bridge = l2.get("background_bridge_verification") or report.get("background_bridge_verification")
    if isinstance(bridge, dict) and bridge:
        bridge_id = add_node(
            "commonsense_bridge_verifier",
            {
                "bridge_status": bridge.get("bridge_status"),
                "not_direct_visual_evidence": bridge.get("not_direct_visual_evidence"),
                "visual_anchor_refs": deepcopy(bridge.get("visual_anchor_refs") or []),
                "best_option": deepcopy(bridge.get("best_option") or {}),
            },
        )
        add_edge(verifier_id, bridge_id, "bridges_with_objective_context")
        last_id = bridge_id

    final_status = l2.get("repair_status") or report.get("repair_status")
    final_id = add_node(
        "final_commit_or_abstain",
        {
            "terminal_status": final_status,
            "repair_needed_after_round": report.get("repair_needed_after_round", final_status not in STRONG_TERMINAL_STATUSES),
            "best_option": deepcopy(l2.get("best_option") or report.get("best_option") or {}),
            "verifier_reason": report.get("verifier_reason"),
        },
    )
    add_edge(last_id, final_id, "commits_or_abstains")

    return {
        "schema_version": REPAIR_SUBGRAPH_SCHEMA,
        "example_id": example_id,
        "dataset": plan.get("dataset") or l2.get("dataset") or report.get("dataset"),
        "nodes": nodes,
        "edges": edges,
    }


def repair_artifacts_to_round(
    *,
    plan: dict[str, Any],
    patch: dict[str, Any],
    l2: dict[str, Any],
    report: dict[str, Any],
    round_index: int = 1,
) -> dict[str, Any]:
    status = str(l2.get("repair_status") or report.get("repair_status") or "")
    repair_needed = bool(report.get("repair_needed_after_round", status not in STRONG_TERMINAL_STATUSES))
    return {
        "round_index": round_index,
        "round_type": "repair_l2_reasoning",
        "state_snapshot": {
            "repair_plan": {
                "gap_types": deepcopy(plan.get("gap_types") or []),
                "span_count": len(plan.get("spans") or []),
                "retrieval_round_count": report.get("retrieval_round_count", 0),
            },
            "l1_patch": deepcopy(patch.get("counts") or {}),
        },
        "action": {
            "action_type": "bounded_recursive_repair",
            "tool_backend": l2.get("backend"),
            "repair_mode": plan.get("repair_mode"),
            "selection_mode": report.get("selection_mode"),
        },
        "observation_summary": {
            "repair_status": status,
            "best_option": deepcopy(l2.get("best_option") or {}),
            "selector_abstained": bool(report.get("selector_abstained")),
            "option_pack_count": len(report.get("option_evidence_packs") or []),
        },
        "graph_delta": {
            "l1_patch_counts": deepcopy(patch.get("counts") or {}),
            "repair_subgraph_nodes_added": 0,
        },
        "verifier_signal": {
            "status": status,
            "verifier_backend": l2.get("backend"),
            "verifier_reason": report.get("verifier_reason"),
            "option_verifier_policy": deepcopy(l2.get("option_verifier_policy") or {}),
            "background_bridge_verification": deepcopy(l2.get("background_bridge_verification") or {}),
        },
        "reward_proxy": reward_proxy_from_status(status, repair_needed=repair_needed),
        "terminal_status": status if not repair_needed else "needs_more_evidence",
    }


def repair_artifacts_to_trajectory(
    *,
    plan: dict[str, Any],
    patch: dict[str, Any],
    l2: dict[str, Any],
    report: dict[str, Any],
    max_repair_rounds: int = 2,
) -> dict[str, Any]:
    subgraph = repair_artifacts_to_subgraph(plan=plan, patch=patch, l2=l2, report=report)
    repair_round = repair_artifacts_to_round(plan=plan, patch=patch, l2=l2, report=report)
    repair_round["graph_delta"]["repair_subgraph_nodes_added"] = len(subgraph.get("nodes") or [])
    return {
        "schema_version": L2_TRAJECTORY_SCHEMA,
        "process_model": "pomdp_compatible_bounded_recursive_graph_agent",
        "is_training_mdp": False,
        "max_repair_rounds": max_repair_rounds,
        "rounds": [repair_round],
        "repair_subgraph": subgraph,
    }
