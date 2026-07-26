from __future__ import annotations

from steam_video_new.implicit_world_model.full_graph_iwm.transition_audit_review import (
    review_transition_audit,
)

from steam_video_new.implicit_world_model.full_graph_iwm.transition_audit import (
    build_transition_audit_packet,
)


def _source() -> tuple[dict, dict, dict]:
    candidates = {
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
                    "question": "What happened?",
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
            }
        ],
    }
    dataset = {
        "cases": [
            {"case_id": "case:test", "video_id": "video:test", "split": "test"}
        ]
    }
    evidence = {
        "l1:event": {
            "node_id": "l1:event",
            "video_id": "video:test",
            "time_span": {"start_s": 1.0, "end_s": 2.0},
            "text": "person opens door",
            "provenance": {"uses_hidden_supervision": False},
            "metadata": {"predicate": "person opens door"},
        }
    }
    return candidates, dataset, evidence


class _ReviewClient:
    model = "gpt-review-test"

    def __init__(self) -> None:
        self.payload = None

    def complete_json(self, *, task, payload):
        del task
        self.payload = payload
        return {
            "decisions": {
                alias: {
                    "decision": "over_crediting",
                    "rationale": "the visible event does not justify answer readiness",
                }
                for alias in payload["records"]
            }
        }


def test_model_review_is_blinded_to_automatic_slice_and_hidden_gt() -> None:
    candidates, dataset, evidence = _source()
    packet, _ = build_transition_audit_packet(
        candidates,
        dataset,
        packet_id="packet:test",
        evidence_by_id=evidence,
    )
    client = _ReviewClient()

    review, inspection = review_transition_audit(packet, client, batch_size=1)

    assert review["labels_source"] == "model_provisional"
    assert review["training_allowed"] is False
    assert review["automatic_failure_slice_visible_to_reviewer"] is False
    assert review["decisions"][0]["decision"] == "over_crediting"
    assert inspection["coverage_complete"] is True
    assert inspection["independent_human_review_complete"] is False
    assert "automatic_failure_slice" not in str(client.payload["records"])
    assert "reference_answer" not in str(client.payload)
