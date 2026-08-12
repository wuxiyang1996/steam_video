import json
from pathlib import Path

from steam_video_new.implicit_world_model.cgbench_grounded_navigation.fixed_l15_cohort import (
    build_fixed_heldout_cohort,
    evaluate_fixed_l15_cohort,
)


def test_fixed_cohort_selection_is_complete_and_graph_worker_safe(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "cg_videos"
    video_root.mkdir()
    for video_id in ("video-a", "video-b"):
        (video_root / f"{video_id}.mp4").write_bytes(b"video")
    dataset = {
        "dataset_id": "cgbench:test",
        "cases": [
            {
                "case_id": "case-a",
                "video_id": "video-a",
                "video_duration_s": 10.0,
                "split": "validation",
                "planner_input": {"question": "hidden from worker"},
            },
            {
                "case_id": "case-b",
                "video_id": "video-b",
                "video_duration_s": 12.0,
                "split": "test",
                "planner_input": {"question": "hidden from worker"},
            },
        ],
    }
    manifest = {
        "dataset_id": "cgbench:test",
        "videos": [
            {
                "video_id": "video-a",
                "video_ref": "cg_videos/video-a.mp4",
                "current_candidate_source": "unavailable",
                "cases": [{"case_id": "case-a", "split": "validation"}],
            },
            {
                "video_id": "video-b",
                "video_ref": "cg_videos/video-b.mp4",
                "current_candidate_source": "subtitle",
                "cases": [{"case_id": "case-b", "split": "test"}],
            },
        ],
    }

    selection, protocol = build_fixed_heldout_cohort(
        dataset,
        manifest,
        dataset_root=tmp_path,
        minimum_cases=2,
        maximum_cases=3,
    )

    assert protocol["case_count"] == 2
    assert protocol["video_count"] == 2
    assert [row["case_id"] for row in protocol["cases"]] == ["case-b", "case-a"]
    serialized_selection = json.dumps(selection)
    assert "hidden from worker" not in serialized_selection
    assert all(
        set(row)
        == {
            "video_id",
            "video_ref",
            "split",
            "fallback_source",
            "duration_s",
            "observation_horizon_s",
        }
        for row in selection["videos"]
    )
    assert all(
        row["observation_horizon_s"] == row["duration_s"] for row in selection["videos"]
    )


def test_fixed_cohort_gate_separates_raw_consolidation_and_delayed(
    tmp_path: Path,
) -> None:
    selection = {
        "schema_version": "selection",
        "dataset_id": "cgbench:test",
        "videos": [
            {
                "video_id": "video-a",
                "video_ref": "cg_videos/video-a.mp4",
                "split": "test",
                "duration_s": 10.0,
                "observation_horizon_s": 10.0,
            }
        ],
    }
    protocol = {
        "schema_version": "protocol",
        "dataset_id": "cgbench:test",
        "minimum_locked_cases": 1,
        "maximum_locked_cases": 3,
        "cases": [
            {"case_id": "pass", "video_id": "video-a", "split": "test"},
            {
                "case_id": "consolidation",
                "video_id": "video-a",
                "split": "test",
            },
            {"case_id": "raw-miss", "video_id": "video-a", "split": "test"},
        ],
    }
    dataset = {
        "dataset_id": "cgbench:test",
        "cases": [
            {"case_id": case_id, "video_id": "video-a", "split": "test"}
            for case_id in ("pass", "consolidation", "raw-miss")
        ],
    }
    hidden = {
        "cases": [
            {
                "case_id": "pass",
                "clue_intervals": [
                    {"start_s": 0.0, "end_s": 1.0},
                    {"start_s": 4.0, "end_s": 5.0},
                ],
            },
            {
                "case_id": "consolidation",
                "clue_intervals": [{"start_s": 6.0, "end_s": 7.0}],
            },
            {
                "case_id": "raw-miss",
                "clue_intervals": [{"start_s": 8.0, "end_s": 9.0}],
            },
        ]
    }
    sample_dir = tmp_path / "video-a"
    sample_dir.mkdir()
    source_nodes = [
        _node("n0", 0.0, 1.0),
        _node("n1", 2.0, 3.0),
        _node("n2", 4.0, 5.0),
        _node("n3", 6.0, 7.0),
    ]
    (sample_dir / "causal_temporal_overlay.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "question_independent_contract": True,
                    "full_video_scope": True,
                },
                "l1_observations": source_nodes,
            }
        )
    )
    (sample_dir / "l1_l15_navigation_graph.json").write_text(
        json.dumps(
            {
                "metadata": {"question_independent": True},
                "nodes": source_nodes[:3],
                "temporal_edges": [
                    {"src": "n0", "dst": "n1"},
                    {"src": "n1", "dst": "n2"},
                ],
                "correlation_edges": [],
            }
        )
    )

    report, details = evaluate_fixed_l15_cohort(
        dataset,
        hidden,
        selection,
        protocol,
        graph_root=tmp_path,
    )

    assert report["gate_passed"] is True
    assert report["locked_case_ids"] == ["pass"]
    assert report["structural_delayed_candidate_count"] == 1
    assert report["failure_counts"] == {
        "bounded_consolidation_dropped_clue": 1,
        "raw_l1_missing_clue": 1,
    }
    rows = {row["case_id"]: row for row in details["cases"]}
    assert rows["pass"]["consecutive_clue_shortest_path_hops"] == [2]
    assert rows["consolidation"]["failure_reason"] == (
        "bounded_consolidation_dropped_clue"
    )
    assert rows["raw-miss"]["failure_reason"] == "raw_l1_missing_clue"


def _node(node_id: str, start_s: float, end_s: float) -> dict[str, object]:
    return {
        "node_id": node_id,
        "time_span": {"start_s": start_s, "end_s": end_s},
    }
