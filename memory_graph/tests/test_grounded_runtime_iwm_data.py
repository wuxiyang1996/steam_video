from __future__ import annotations

from copy import deepcopy

from memory_graph.types import EmbeddingRef, MemoryNode, TimeSpan
from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
    RetainedEvidenceGraph,
    TemporalNavigationEdge,
)
from steam_video_new.implicit_world_model.iwm_9b.calibration_gate import (
    evaluate_transition_predictions,
)
from steam_video_new.implicit_world_model.iwm_9b.grounded_runtime_data import (
    _transition_counts,
    _transition_records,
    build_grounded_runtime_data,
)
from steam_video_new.implicit_world_model.iwm_9b.predict_validation import (
    _categorical_prediction,
    _extract_json,
)
from steam_video_new.implicit_world_model.iwm_9b.targeted_delayed_collection import (
    _case_candidates,
    _selected_path_to_run,
)


def _observation(node_id: str, start_s: float) -> dict:
    return {
        "node_id": node_id,
        "node_type": "observation",
        "text": f"grounded observation {node_id}",
        "time_span": {"start_s": start_s, "end_s": start_s + 1.0},
        "embedding_ref": {
            "model": "Qwen/Qwen3-VL-Embedding-2B",
            "dimension": 2048,
            "dtype": "float32",
            "normalized": True,
            "row_index": int(start_s),
            "checksum": "abc",
        },
        "metadata": {
            "predicate": "grounded_event",
            "participants": [{"role": "actor", "surface": "person"}],
            "states": [],
            "state_change": None,
            "modality": "visual",
        },
    }


def _belief(question: str, evidence: list[str]) -> dict:
    return {
        "question": question,
        "required_roles": ["first", "second"],
        "missing_roles": ["first", "second"],
        "grounded_role_evidence": [],
        "contradictions": [],
        "answerability": "not_ready",
        "acquired_evidence": evidence,
    }


def _step(
    question: str,
    node_id: str,
    *,
    index: int,
    correlation: bool,
) -> dict:
    before = _belief(question, [] if index == 0 else ["earlier"])
    after = deepcopy(before)
    after["acquired_evidence"] = [*before["acquired_evidence"], node_id]
    return {
        "observation_id": node_id,
        "decision": {
            "selected_action": {
                "action_id": f"read:{node_id}",
                "kind": "follow_correlation" if correlation else "read_node",
                "source_id": "source",
                "target_id": node_id,
                "edge_id": "edge" if correlation else None,
                "relation": "l1.5_correlation" if correlation else None,
            },
            "imagined_paths": [
                {
                    "transitions": [
                        {"action": {"action_id": f"read:{node_id}"}}
                    ]
                }
            ],
        },
        "pool_before": {
            "trajectories": [
                {
                    "trajectory_id": "trajectory:a",
                    "hypothesis": "answer-a",
                    "belief": before,
                }
            ]
        },
        "pool_after": {
            "trajectories": [
                {
                    "trajectory_id": "trajectory:a",
                    "hypothesis": "answer-a",
                    "belief": after,
                }
            ]
        },
    }


def _run(case_id: str, video_id: str, arm: str) -> dict:
    question = f"question for {case_id}"
    negative_id = f"visual_l1:{video_id}:negative:{arm}"
    positive_id = f"visual_l1:{video_id}:positive:{arm}"
    return {
        "case_id": case_id,
        "arm": arm,
        "graph_fingerprint": f"fingerprint:{video_id}",
        "real_observations": [
            _observation(negative_id, 1.0),
            _observation(positive_id, 2.0),
        ],
        "realized_labels_evaluator_only": [
            {"observation_id": negative_id, "newly_covered_clue_indices": []},
            {"observation_id": positive_id, "newly_covered_clue_indices": [0]},
        ],
        "steps": [
            _step(question, negative_id, index=0, correlation=True),
            _step(question, positive_id, index=1, correlation=False),
        ],
    }


def _fixture() -> tuple[list[dict], dict, dict]:
    public_cases = []
    hidden_cases = []
    runs = []
    for split in ("train", "validation"):
        for index in range(2):
            case_id = f"{split}-case-{index}"
            video_id = f"{split}-video-{index}"
            public_cases.append(
                {"case_id": case_id, "video_id": video_id, "split": split}
            )
            hidden_cases.append(
                {"case_id": case_id, "video_id": video_id, "clue_intervals": [{}]}
            )
            runs.append(_run(case_id, video_id, "world_model_guided"))
            runs.append(_run(case_id, video_id, "no_world_model"))
    return (
        [{"runs": runs}],
        {"dataset_id": "fixture", "cases": public_cases},
        {"cases": hidden_cases},
    )


def test_grounded_runtime_data_gates_real_delayed_and_hard_negative_records() -> None:
    artifacts, dataset, hidden = _fixture()
    result = build_grounded_runtime_data(
        artifacts,
        dataset,
        hidden,
        minimum_train_videos=2,
        minimum_train_records=4,
        minimum_validation_videos=2,
        minimum_validation_records=4,
        minimum_delayed_positives=2,
        minimum_delayed_positive_units=2,
        minimum_validation_delayed_positive_units=2,
        minimum_semantic_hard_negatives=2,
    )

    assert result["transition_gate"]["passed"] is True
    assert result["split_audit"]["video_disjoint"] is True
    train = [row for row in result["transition_records"] if row["split"] == "train"]
    assert train and all(row["training_eligible"] for row in train)
    assert any(row["data_slices"]["delayed_positive"] for row in train)
    assert any(
        row["data_slices"]["semantic_neighbor_hard_negative"] for row in train
    )
    assert all(row["hidden_supervision_in_input"] is False for row in train)
    assert all(row["numeric_reward_present"] is False for row in train)
    assert all(
        row["input"]["target_node_safe_view"]["embedding_ref"]["status"]
        == "available"
        for row in train
    )
    assert all(
        row["audit"]["model_corrected_belief_after_not_supervised"]
        for row in train
    )
    assert result["transition_counts"]["train"]["delayed_positive_unit_count"] == 4
    assert (
        result["transition_counts"]["validation"]["delayed_positive_unit_count"]
        == 4
    )


def test_grounded_runtime_data_does_not_count_hypothesis_expansions_as_delayed_units() -> None:
    artifacts, dataset, hidden = _fixture()
    result = build_grounded_runtime_data(
        artifacts,
        dataset,
        hidden,
        minimum_train_videos=2,
        minimum_train_records=4,
        minimum_validation_videos=2,
        minimum_validation_records=4,
        minimum_delayed_positives=2,
        minimum_delayed_positive_units=2,
        minimum_validation_delayed_positive_units=5,
        minimum_semantic_hard_negatives=2,
    )

    assert result["transition_gate"]["passed"] is False
    assert (
        "validation_delayed_positive_unit_count"
        in result["transition_gate"]["blockers"]
    )

    validation = [
        row for row in result["transition_records"] if row["split"] == "validation"
    ]
    delayed = next(row for row in validation if row["data_slices"]["delayed_positive"])
    hypothesis_expansion = deepcopy(delayed)
    hypothesis_expansion["record_id"] = "runtime-transition:another-hypothesis"
    hypothesis_expansion["input"]["hypothesis"] = "another answer hypothesis"
    original_counts = _transition_counts(validation)["validation"]
    expanded_counts = _transition_counts([*validation, hypothesis_expansion])["validation"]
    assert expanded_counts["delayed_positive"] == original_counts["delayed_positive"] + 1
    assert (
        expanded_counts["delayed_positive_unit_count"]
        == original_counts["delayed_positive_unit_count"]
    )


def test_targeted_delayed_path_is_legal_grounded_and_counts_as_one_unit() -> None:
    nodes = tuple(
        MemoryNode(
            node_id=f"node:{index}",
            video_id="validation-video",
            time_span=TimeSpan(float(index), float(index + 1)),
            provenance={"uses_hidden_supervision": False},
            node_type="observation",
            text=f"event {index}",
            embedding_ref=EmbeddingRef(path="embeddings.npy", row_index=index),
        )
        for index in range(2)
    )
    graph = RetainedEvidenceGraph(
        graph_id="graph:fixture",
        nodes=nodes,
        temporal_edges=(
            TemporalNavigationEdge(
                edge_id="edge:temporal",
                src="node:0",
                dst="node:1",
                relation="temporal_next",
            ),
        ),
        correlation_edges=(),
        capacity=2,
    )
    public_case = {
        "case_id": "validation-case",
        "video_id": "validation-video",
        "split": "validation",
        "planner_input": {
            "question": "What happens next?",
            "choices": ["answer-a", "answer-b"],
        },
    }
    hidden_case = {
        "case_id": "validation-case",
        "video_id": "validation-video",
        "clue_intervals": [{"start_s": 1.0, "end_s": 2.0}],
    }
    candidates = _case_candidates(
        "validation-case",
        public_case,
        hidden_case,
        graph,
        anchors=("node:0",),
        required_roles=("next_event",),
    )

    assert len(candidates) == 1
    run = _selected_path_to_run(candidates[0])
    records = _transition_records(
        [run],
        {"validation-case": public_case},
        {"validation-case": hidden_case},
    )
    counts = _transition_counts(records)["validation"]
    assert counts["delayed_positive"] == 2
    assert counts["delayed_positive_unit_count"] == 1
    assert counts["embedding:available"] == 4
    predictions = [
        {
            "record_id": row["record_id"],
            "prediction": {
                field: row["target"]["categorical_audit"][field]["value"]
                for field in (
                    "observation_outcome",
                    "progress",
                    "required_clue_coverage_after",
                    "answerability_after",
                )
            },
        }
        for row in records
    ]
    delayed_id = next(
        row["record_id"] for row in records if row["data_slices"]["delayed_positive"]
    )
    next(
        row for row in predictions if row["record_id"] == delayed_id
    )["prediction"]["observation_outcome"] = "inconclusive"
    report = evaluate_transition_predictions(
        records, predictions, minimum_video_count=1
    )
    assert report["slice_metrics"]["delayed_positive_recall"] == 0.5
    assert report["slice_metrics"]["delayed_positive_unit_count"] == 1
    assert report["slice_metrics"]["delayed_positive_unit_recall"] == 0.0
    assert "delayed_positive_unit_recall" in report["blockers"]


def test_legacy_direct_step_observations_restore_public_hypothesis_paths() -> None:
    public_case = {
        "case_id": "legacy-case",
        "video_id": "legacy-video",
        "split": "validation",
        "planner_input": {
            "question": "What is shown?",
            "choices": ["answer-a", "answer-b"],
        },
    }
    hidden_case = {
        "case_id": "legacy-case",
        "video_id": "legacy-video",
        "clue_intervals": [{"start_s": 1.0, "end_s": 2.0}],
    }
    first = _observation("legacy:first", 0.0)
    second = _observation("legacy:second", 1.0)
    before = _belief("", [])
    middle = _belief("", ["legacy:first"])
    after = _belief("", ["legacy:first", "legacy:second"])
    run = {
        "case_id": "legacy-case",
        "arm": "world_model_guided",
        "graph_fingerprint": "legacy-fingerprint",
        "steps": [
            {
                "belief_before": before,
                "selected_action": {
                    "action_id": "legacy:first-action",
                    "kind": "start_at",
                    "target_id": "legacy:first",
                },
                "real_observation": first,
                "realized_label_evaluator_only": {
                    "newly_covered_clue_indices": []
                },
                "belief_after_real_read": middle,
            },
            {
                "belief_before": middle,
                "selected_action": {
                    "action_id": "legacy:second-action",
                    "kind": "temporal_forward",
                    "source_id": "legacy:first",
                    "target_id": "legacy:second",
                    "relation": "temporal_next",
                },
                "real_observation": second,
                "realized_label_evaluator_only": {
                    "newly_covered_clue_indices": [0]
                },
                "belief_after_real_read": after,
            },
        ],
    }

    records = _transition_records(
        [run], {"legacy-case": public_case}, {"legacy-case": hidden_case}
    )
    assert len(records) == 4
    assert {row["input"]["hypothesis"] for row in records} == {
        "answer-a",
        "answer-b",
    }
    assert sum(row["data_slices"]["delayed_positive"] for row in records) == 2


def test_grounded_runtime_data_blocks_training_without_heldout_reads() -> None:
    artifacts, dataset, hidden = _fixture()
    artifacts[0]["runs"] = [
        row for row in artifacts[0]["runs"] if row["case_id"].startswith("train")
    ]
    result = build_grounded_runtime_data(
        artifacts,
        dataset,
        hidden,
        minimum_train_videos=2,
        minimum_train_records=4,
        minimum_validation_videos=1,
        minimum_validation_records=1,
        minimum_delayed_positives=1,
        minimum_semantic_hard_negatives=1,
    )

    assert result["transition_gate"]["passed"] is False
    assert "validation_video_count" in result["transition_gate"]["blockers"]
    assert not any(row["training_eligible"] for row in result["transition_records"])
    assert not any(
        row["training_eligible"] for row in result["planner_preference_records"]
    )


def test_calibration_gate_uses_categorical_outputs_and_reports_slices() -> None:
    artifacts, dataset, hidden = _fixture()
    result = build_grounded_runtime_data(
        artifacts,
        dataset,
        hidden,
        minimum_train_videos=2,
        minimum_train_records=4,
        minimum_validation_videos=2,
        minimum_validation_records=4,
        minimum_delayed_positives=2,
        minimum_delayed_positive_units=2,
        minimum_validation_delayed_positive_units=2,
        minimum_semantic_hard_negatives=2,
    )
    validation = [
        row for row in result["transition_records"] if row["split"] == "validation"
    ]
    predictions = []
    for row in validation:
        audit = row["target"]["categorical_audit"]
        predictions.append(
            {
                "record_id": row["record_id"],
                "prediction": {
                    name: audit[name]["value"]
                    for name in (
                        "observation_outcome",
                        "progress",
                        "required_clue_coverage_after",
                        "answerability_after",
                    )
                },
            }
        )
    report = evaluate_transition_predictions(
        result["transition_records"],
        predictions,
        minimum_video_count=2,
    )

    assert report["passed"] is True
    assert report["planner_training_authorized"] is True
    assert report["model_output_contract"] == "categorical_labels_only"
    assert report["numeric_metrics_are_evaluator_only"] is True
    assert report["slice_metrics"]["delayed_positive_recall"] == 1.0
    assert report["slice_metrics"]["delayed_positive_unit_recall"] == 1.0
    assert report["slice_metrics"]["semantic_hard_negative_support_rate"] == 0.0


def test_validation_prediction_parser_accepts_fenced_categorical_json() -> None:
    value = _extract_json(
        """```json
        {
          "categorical_audit": {
            "observation_outcome": "support",
            "progress": "advanced",
            "required_clue_coverage_after": "partial",
            "answerability_after": "not_ready"
          }
        }
        ```"""
    )

    assert _categorical_prediction(value) == {
        "observation_outcome": "support",
        "progress": "advanced",
        "required_clue_coverage_after": "partial",
        "answerability_after": "not_ready",
    }
