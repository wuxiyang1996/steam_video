from __future__ import annotations

import json

from steam_video_new.implicit_world_model.iwm_9b.grounded_runtime_data import (
    _transition_records,
)
from steam_video_new.implicit_world_model.iwm_9b.grounded_runtime_v4 import (
    build_v4_records,
)
from steam_video_new.implicit_world_model.iwm_9b.observation_calibration import (
    evaluate_observation_predictions,
)
from steam_video_new.implicit_world_model.iwm_9b.observation_baseline import (
    build_predictions,
)
from steam_video_new.implicit_world_model.iwm_9b.predict_observation_validation import (
    _observation_prediction,
)
from steam_video_new.implicit_world_model.iwm_9b.schemas import OBSERVATION_TASK
from steam_video_new.implicit_world_model.iwm_9b.serialization import (
    render_sft_example,
)
from steam_video_new.implicit_world_model.iwm_9b.train_sft import (
    load_training_records,
)


def _observation(node_id: str, event: str) -> dict:
    return {
        "node_id": node_id,
        "node_type": "observation",
        "text": event,
        "time_span": {"start_s": 0.0, "end_s": 1.0},
        "embedding_ref": {
            "model": "Qwen/Qwen3-VL-Embedding-2B",
            "dimension": 2048,
            "row_index": 0,
            "checksum": "fixture",
        },
        "metadata": {
            "predicate": event,
            "participants": [
                {"role": "actor", "entity_type": "person", "surface": "hand"}
            ],
            "states": [],
            "state_change": None,
            "modality": "visual",
            "action_kind": "action",
        },
    }


def _belief(question: str, acquired: list[str]) -> dict:
    return {
        "question": question,
        "required_roles": ["anchor", "answer"],
        "missing_roles": ["anchor", "answer"],
        "grounded_role_evidence": [],
        "contradictions": [],
        "answerability": "not_ready",
        "acquired_evidence": acquired,
    }


def _legacy_run(case_id: str, split: str) -> tuple[dict, dict, dict]:
    video_id = f"{split}-video-{case_id}"
    first_id = f"node:{case_id}:first"
    second_id = f"node:{case_id}:second"
    first = _observation(first_id, "person approaches object")
    second = _observation(second_id, "person opens object")
    before = _belief("What happens next?", [])
    middle = _belief("What happens next?", [first_id])
    after = _belief("What happens next?", [first_id, second_id])
    run = {
        "case_id": case_id,
        "arm": "targeted_grounded_delayed_collection",
        "graph_fingerprint": f"fingerprint:{case_id}",
        "steps": [
            {
                "belief_before": before,
                "selected_action": {
                    "action_id": f"action:{case_id}:first",
                    "kind": "start_at",
                    "target_id": first_id,
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
                    "action_id": f"action:{case_id}:second",
                    "kind": "follow_correlation",
                    "source_id": first_id,
                    "target_id": second_id,
                    "edge_id": f"edge:{case_id}",
                    "relation": "semantic",
                },
                "real_observation": second,
                "realized_label_evaluator_only": {
                    "newly_covered_clue_indices": [0]
                },
                "belief_after_real_read": after,
            },
        ],
    }
    public = {
        "case_id": case_id,
        "video_id": video_id,
        "split": split,
        "planner_input": {
            "question": "What happens next?",
            "choices": ["opens it", "leaves it"],
        },
    }
    hidden = {
        "case_id": case_id,
        "video_id": video_id,
        "clue_intervals": [{}],
    }
    return run, public, hidden


def _legacy_records() -> list[dict]:
    runs = []
    public = {}
    hidden = {}
    for split in ("train", "validation"):
        for index in range(2):
            case_id = f"{split}-{index}"
            run, public_case, hidden_case = _legacy_run(case_id, split)
            runs.append(run)
            public[case_id] = public_case
            hidden[case_id] = hidden_case
    return _transition_records(runs, public, hidden)


def test_v4_deduplicates_hypotheses_and_keeps_clue_gt_evaluator_only() -> None:
    legacy = _legacy_records()
    assert len(legacy) == 16

    result = build_v4_records(
        legacy,
        minimum_train_units=4,
        minimum_validation_units=4,
        minimum_train_videos=2,
        minimum_validation_videos=2,
    )

    observations = result["observation_records"]
    assert len(observations) == 8
    assert len(result["hypothesis_effect_candidates"]) == 16
    assert result["representation_gate"]["passed"] is True
    assert result["observation_training_ready"] is True
    assert result["hypothesis_effect_training_ready"] is False
    assert result["planner_training_ready"] is False
    assert all(row["representation_eligible"] for row in observations)
    assert all("hypothesis" not in row["input"] for row in observations)
    assert all(
        "clue_acquisition" not in row["target"] for row in observations
    )
    assert all(
        row["evaluation_labels"]["clue_acquisition"]["model_target"] is False
        for row in observations
    )
    assert {
        row["evaluation_labels"]["clue_acquisition"]["value"]
        for row in observations
    } == {"new_clue", "no_new_clue"}
    assert all(
        row["target"]["hypothesis_effect"]["supervised"] is False
        for row in result["hypothesis_effect_candidates"]
    )


def test_v4_text_bridge_exposes_acquired_source_but_not_unread_target() -> None:
    result = build_v4_records(
        _legacy_records(),
        minimum_train_units=4,
        minimum_validation_units=4,
        minimum_train_videos=2,
        minimum_validation_videos=2,
    )
    second = next(
        row
        for row in result["observation_records"]
        if row["input"]["action"]["kind"] == "follow_correlation"
    )

    assert second["input"]["source_evidence"]["descriptor"]["event"] == (
        "person approaches object"
    )
    assert second["input"]["persistent_read_state"]["acquired_evidence"]
    assert second["input"]["target_node_key"]["semantic_key"] == (
        "person opens object"
    )
    assert second["input"]["edge_context"]["channel"] == "correlation"
    assert second["input"]["representation_contract"][
        "unread_target_full_evidence_visible"
    ] is False
    assert "observation_descriptor" not in second["input"]
    assert second["audit"]["embedding_values_consumed"] is False


def test_v4_observation_task_serializes_and_loads_for_sft(tmp_path) -> None:
    result = build_v4_records(
        _legacy_records(),
        minimum_train_units=4,
        minimum_validation_units=4,
        minimum_train_videos=2,
        minimum_validation_videos=2,
    )
    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in result["observation_records"]),
        encoding="utf-8",
    )

    train = load_training_records(path, OBSERVATION_TASK)
    assert len(train) == 4
    rendered = render_sft_example(train[0])
    assert "dataset clue" in rendered["messages"][0]["content"]
    assert "clue_acquisition" not in rendered["messages"][2]["content"]


def test_v4_observation_parser_and_calibration_are_separate_from_clue_gt() -> None:
    result = build_v4_records(
        _legacy_records(),
        minimum_train_units=4,
        minimum_validation_units=4,
        minimum_train_videos=2,
        minimum_validation_videos=2,
    )
    validation = [
        row
        for row in result["observation_records"]
        if row["split"] == "validation" and row["representation_eligible"]
    ]
    predictions = []
    for row in validation:
        target = row["target"]["observation_descriptor"]["value"]
        parsed = _observation_prediction({"observation_descriptor": target})
        predictions.append(
            {
                "record_id": row["record_id"],
                "prediction": {"observation_descriptor": parsed},
                "model_output_contains_numeric_scalar": False,
            }
        )
    report = evaluate_observation_predictions(
        result["observation_records"],
        predictions,
        minimum_video_count=2,
    )

    assert report["passed"] is True
    assert report["metrics"]["full_descriptor_accuracy"] == 1.0
    assert report["clue_acquisition_scored_as_model_target"] is False
    assert report["hypothesis_effect_training_authorized"] is False
    assert report["planner_training_authorized"] is False

    baseline = build_predictions(result["observation_records"])
    assert len(baseline) == len(validation)
    assert all(
        row["baseline"] == "copy_target_semantic_address" for row in baseline
    )
