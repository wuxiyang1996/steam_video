from __future__ import annotations

import json

from steam_video_new.implicit_world_model.cgbench_grounded_navigation.iwm_supervision import (
    build_iwm_supervision_artifacts,
)


def test_iwm_supervision_keeps_gt_alignment_hidden_and_unmatched_pairs_unlabeled(
    tmp_path,
) -> None:
    dataset = {
        "dataset_id": "test",
        "cases": [
            {
                "case_id": "case:1",
                "video_id": "video:1",
                "video_ref": "video.mp4",
                "video_duration_s": 30.0,
                "split": "test",
                "planner_input": {"question": "what happened?", "choices": ["a", "b"]},
                "executed_transitions": [
                    {
                        "transition_id": "transition:1",
                        "checkpoint": {"required_clue_coverage": "none"},
                        "action": {"action_id": "read:1"},
                        "real_observation": {
                            "descriptor_status": "pending_qwen_vl_grounded_read",
                            "descriptor": None,
                        },
                        "target": {"belief_delta": {"evidence_progress": "advances"}},
                    }
                ],
                "trajectory_preference": {"label": "prefer_right"},
            }
        ],
    }
    hidden = {
        "cases": [
            {
                "case_id": "case:1",
                "clue_intervals": [
                    {"start_s": 1.0, "end_s": 2.0},
                    {"start_s": 10.0, "end_s": 11.0},
                ],
            }
        ]
    }
    graph = {
        "graph_id": "graph:1",
        "nodes": [
            {
                "node_id": "n1",
                "video_id": "video:1",
                "time_span": {"start_s": 0.0, "end_s": 3.0},
            },
            {
                "node_id": "n2",
                "video_id": "video:1",
                "time_span": {"start_s": 9.0, "end_s": 12.0},
            },
            {
                "node_id": "n3",
                "video_id": "video:1",
                "time_span": {"start_s": 20.0, "end_s": 21.0},
            },
        ],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    supervision, alignment, readiness, queue = build_iwm_supervision_artifacts(
        dataset,
        hidden,
        graph_paths=(graph_path,),
    )

    assert "clue_intervals" not in supervision["cases"][0]
    assert "l1.5_graph_builder" in supervision["forbidden_consumers"]
    assert supervision["numeric_reward_present"] is False
    bridge = alignment["cases"][0]["consecutive_bridge_groups"][0]
    assert bridge["status"] == "aligned_multi_positive"
    assert bridge["positive_node_pairs"] == [["n1", "n2"]]
    assert bridge["unmatched_same_video_pairs_are_negative"] is False
    assert readiness["heldout_aligned_bridge_group_count"] == 1
    assert readiness["grounded_observation_descriptor_ready_transition_count"] == 0
    assert readiness["phase_3_training_ready"] is False
    assert "clue_intervals" in queue["forbidden_builder_inputs"]
    assert queue["videos"][0]["required_graph_scope"] == (
        "full_video_question_independent"
    )
