from __future__ import annotations

import json

import pytest

from steam_video_new.implicit_world_model.iwm_9b.adapters import (
    adapt_grounded_transition_corpus,
    adapt_local_choice_packet,
    build_readiness_report,
)
from steam_video_new.implicit_world_model.iwm_9b.schemas import (
    PLANNER_TASK,
    supervised_target,
)
from steam_video_new.implicit_world_model.iwm_9b.serialization import (
    render_sft_example,
)
from steam_video_new.implicit_world_model.iwm_9b.train_sft import (
    load_training_records,
)


def test_transition_adapter_masks_unavailable_runtime_labels() -> None:
    corpus = {
        "records": [
            {
                "record_id": "transition:1",
                "case_id": "case:1",
                "video_id": "video:1",
                "split": "train",
                "training_context": {
                    "question": "What happened?",
                    "checkpoint": {"answerability": "unknown"},
                    "action": {"action_id": "read:1"},
                },
                "real_transition_target": {
                    "observation_descriptor": {"summary": "A door opens."},
                    "observed_modalities": ["sampled_video_frames"],
                    "categorical_belief_delta": {
                        "evidence_progress": "advances_required_clue_coverage",
                        "required_clue_coverage_after": "partial",
                        "answerability_after": "unknown",
                    },
                },
                "target_is_real_not_imagined": True,
            }
        ]
    }

    record = adapt_grounded_transition_corpus(corpus)[0]

    assert record["training_eligible"] is True
    assert record["target"]["observation_patch"]["descriptor"]["supervised"]
    assert record["target"]["categorical_audit"]["progress"]["value"] == "advanced"
    assert record["target"]["categorical_audit"]["answerability_after"] == {
        "value": None,
        "supervised": False,
        "provenance": "not_available_in_source_artifact",
    }
    target = supervised_target(record["target"])
    assert "answerability_after" not in target["categorical_audit"]
    assert "hypothesis_support" not in json.dumps(target)
    assert render_sft_example(record)["training_eligible"] is True


def test_test_only_planner_packet_fails_readiness_and_training(tmp_path) -> None:
    public, hidden = _planner_packet(split="test")
    records = adapt_local_choice_packet(public, hidden)
    report = build_readiness_report(records)

    assert len(records) == 3
    assert {row["task"] for row in records} == {PLANNER_TASK}
    assert report["planner_sft_pipeline_ready"] is False
    assert "no train-split grounded Planner preference records" in report["blockers"]

    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no eligible train records"):
        load_training_records(path, PLANNER_TASK)


def test_train_planner_packet_with_two_classes_passes_readiness(tmp_path) -> None:
    public, hidden = _planner_packet(split="train")
    records = adapt_local_choice_packet(public, hidden)
    report = build_readiness_report(records)

    assert report["planner_sft_pipeline_ready"] is True
    assert report["eligible_preference_label_counts"] == {
        "incomparable": 1,
        "prefer_left": 1,
        "prefer_right": 1,
    }

    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    assert len(load_training_records(path, PLANNER_TASK)) == 3


def _planner_packet(split: str) -> tuple[dict, dict]:
    candidates = [
        {
            "candidate": alias,
            "predicted_transition": {
                "observation": {"outcome": "inconclusive", "descriptor": []},
                "belief_delta": {"progress": "unchanged"},
            },
        }
        for alias in ("candidate_a", "candidate_b", "candidate_c")
    ]
    public = {
        "local_choice_records": [
            {
                "record_id": "choice:1",
                "case_id": "case:1",
                "split": split,
                "question": "What happened?",
                "belief": {"missing_roles": ["event"]},
                "candidates": candidates,
            }
        ]
    }
    hidden = {
        "local_choice_records": [
            {
                "record_id": "choice:1",
                "case_id": "case:1",
                "split": split,
                "candidate_grounding": {
                    alias: {"target_id": f"visual_l1:video1:{index:04d}"}
                    for index, alias in enumerate(
                        ("candidate_a", "candidate_b", "candidate_c")
                    )
                },
                "pairwise_labels": [
                    {
                        "left": "candidate_a",
                        "right": "candidate_b",
                        "label": "prefer_right",
                    },
                    {
                        "left": "candidate_a",
                        "right": "candidate_c",
                        "label": "incomparable",
                    },
                    {
                        "left": "candidate_b",
                        "right": "candidate_c",
                        "label": "prefer_left",
                    },
                ],
                "label_source": "dataset_gt_after_graph_freeze",
            }
        ]
    }
    return public, hidden
