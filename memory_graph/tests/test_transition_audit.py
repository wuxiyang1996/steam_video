from __future__ import annotations

from copy import deepcopy

import pytest

from steam_video_new.implicit_world_model.full_graph_iwm.transition_audit import (
    build_transition_audit_packet,
    validate_transition_audit_packet,
)


def _candidates() -> dict:
    return {
        "schema_version": "steam-executed-iwm-trace-candidates/v0.2",
        "training_allowed": False,
        "transition_over_crediting_candidates": [
            {
                "record_id": "transition_audit:test",
                "case_id": "case:test",
                "step_index": 0,
                "trajectory_id": "trajectory:test",
                "executed_real_observation_id": "l1:event",
                "predicted_belief_delta": {
                    "progress": "advanced",
                    "answerability_after": "ready",
                    "resolved_roles": ["event"],
                },
                "corrected_belief_before": {
                    "missing_roles": ["event"],
                    "answerability": "not_ready",
                },
                "corrected_belief_after": {
                    "missing_roles": ["event"],
                    "answerability": "not_ready",
                },
                "provisional_label": "over_crediting",
                "unsupported_predicted_resolved_roles": ["event"],
                "hidden_evaluator_labels_included": False,
                "training_allowed": False,
            }
        ],
    }


def _dataset() -> dict:
    return {
        "cases": [
            {"case_id": "case:test", "video_id": "video:test", "split": "test"}
        ]
    }


def _evidence() -> dict:
    return {
        "l1:event": {
            "node_id": "l1:event",
            "video_id": "video:test",
            "time_span": {"start_s": 1.0, "end_s": 2.0},
            "text": "person opens door",
            "provenance": {"uses_hidden_supervision": False},
            "metadata": {
                "predicate": "person opens door",
                "action_kind": "manipulation",
                "participants": [{"role": "actor", "entity_type": "person"}],
                "states": [],
                "state_change": None,
            },
        }
    }


def test_transition_audit_packet_stays_blinded_and_training_blocked() -> None:
    packet, report = build_transition_audit_packet(
        _candidates(),
        _dataset(),
        packet_id="packet:test",
        evidence_by_id=_evidence(),
    )

    assert packet["annotation_status"] == "unreviewed"
    assert packet["training_allowed"] is False
    assert packet["contract"]["hidden_clues_or_answers_included"] is False
    assert packet["records"][0]["automatic_failure_slice"] == "over_crediting"
    assert packet["records"][0]["executed_real_observation"]["predicate"] == (
        "person opens door"
    )
    assert packet["records"][0]["review"] == {
        "decision": None,
        "rationale": None,
        "annotator": None,
    }
    assert report["counts"]["case_count"] == 1
    assert report["counts"]["video_count"] == 1
    assert report["training_ready"] is False
    assert report["counts"]["split_counts"] == {"test": 1}
    assert report["readiness_dimensions"]["train_split_present"] is False
    assert report["readiness_dimensions"]["at_least_thirty_train_cases"] is False


def test_transition_audit_rejects_hidden_evaluator_labels() -> None:
    candidates = _candidates()
    candidates["transition_over_crediting_candidates"][0][
        "hidden_evaluator_labels_included"
    ] = True

    with pytest.raises(ValueError, match="hidden evaluator label leaked"):
        build_transition_audit_packet(
            candidates,
            _dataset(),
            packet_id="packet:test",
            evidence_by_id=_evidence(),
        )


def test_unreviewed_transition_cannot_carry_a_review_decision() -> None:
    packet, _ = build_transition_audit_packet(
        _candidates(),
        _dataset(),
        packet_id="packet:test",
        evidence_by_id=_evidence(),
    )
    invalid = deepcopy(packet)
    invalid["records"][0]["review"]["decision"] = "over_crediting"

    with pytest.raises(ValueError, match="contains review labels"):
        validate_transition_audit_packet(invalid)
