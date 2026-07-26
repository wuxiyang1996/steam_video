#!/usr/bin/env python3
"""Smoke test for L2 recursive trajectory and repair-subgraph encoding."""

from __future__ import annotations

from dataset_clip_wrapper.l1_clue_graph.clue_memory import make_reasoning_rollout_shell
from dataset_clip_wrapper.l2_reasoning_graph.l2_recursive_trace import (
    attach_initial_l2_trajectory,
    repair_artifacts_to_trajectory,
)


def test_initial_l2_trajectory() -> None:
    example = {
        "schema_version": "video-skills-relaunch/v0.1",
        "example_id": "toy:001",
        "question": {"question_text": "What happened?", "options": [{"label": "A", "text": "one"}]},
        "available_inputs": {"mode": "video_only"},
    }
    graph = {
        "graph_id": "clue_memory:toy:001",
        "layer": "clue_memory",
        "example_id": "toy:001",
        "dataset": "toy",
        "video_regime": "short",
        "nodes": [{"node_id": "evidence.observation:1"}],
        "edges": [],
        "index_stats": {"node_count": 1},
    }
    rollout = make_reasoning_rollout_shell(example, graph, rollout_source="test")
    rollout["acceptance_status"] = "accepted_weak"
    rollout["failure_reasons"] = []
    rollout["metadata"] = {
        "llm_plan": {"planner": "gpt_oss_reasoning_planner", "model": "openai/gpt-oss-120b"},
        "executed_skill_count": 2,
        "llm_trace_ok": 2,
        "llm_trace_fail": 0,
    }
    attach_initial_l2_trajectory(rollout, graph)
    trajectory = rollout["metadata"]["l2_trajectory"]
    assert trajectory["process_model"] == "pomdp_compatible_bounded_recursive_graph_agent"
    assert trajectory["is_training_mdp"] is False
    assert trajectory["rounds"][0]["terminal_status"] == "repair_requested"


def test_repair_artifacts_to_trajectory() -> None:
    plan = {
        "example_id": "toy:001",
        "dataset": "toy",
        "strategy": "option_verification",
        "repair_mode": "existing_l1_option_verification",
        "gap_types": ["insufficient_support_refs"],
        "spans": [],
        "span_selection": {"selection_mode": "existing_l1_option_verification"},
    }
    patch = {"counts": {"nodes_added": 1, "edges_added": 1}}
    l2 = {
        "example_id": "toy:001",
        "dataset": "toy",
        "backend": "gptoss_verifier",
        "repair_status": "resolved_strong",
        "best_option": {"label": "A", "confidence": 0.8},
        "option_evidence_selector": {"selector_backend": "openai/gpt-oss-120b"},
        "option_verifier_policy": {"supported_option_count": 1},
    }
    report = {
        "example_id": "toy:001",
        "dataset": "toy",
        "repair_status": "resolved_strong",
        "repair_needed_after_round": False,
        "option_evidence_packs": [{"option_label": "A", "positive_refs": ["evidence.observation:1"]}],
        "verifier_reason": "strong repair evidence verified",
    }
    trajectory = repair_artifacts_to_trajectory(plan=plan, patch=patch, l2=l2, report=report)
    assert trajectory["rounds"][0]["terminal_status"] == "resolved_strong"
    assert trajectory["rounds"][0]["reward_proxy"]["value"] == 1.0
    subgraph = trajectory["repair_subgraph"]
    assert len(subgraph["nodes"]) >= 5
    assert any(node["node_type"] == "option_verifier" for node in subgraph["nodes"])


if __name__ == "__main__":
    test_initial_l2_trajectory()
    test_repair_artifacts_to_trajectory()
    print("l2 recursive trace smoke test passed")
