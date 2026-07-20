from __future__ import annotations

import importlib.util

import pytest

from factor_graph.categorical import BeliefLabel, project_probability
from memory_graph.types import (
    CausalTemporalOverlay,
    MemoryNode,
    RelationBelief,
    RelationStatus,
    TimeSpan,
)


def test_categorical_projection_has_three_public_states() -> None:
    assert project_probability(0.1) is BeliefLabel.REJECTED
    assert project_probability(0.5) is BeliefLabel.UNCERTAIN
    assert project_probability(0.9) is BeliefLabel.ACCEPTED


@pytest.mark.skipif(importlib.util.find_spec("gtsam") is None, reason="GTSAM optional")
def test_correction_pilot_passes_all_gates() -> None:
    from factor_graph.experiment import run_experiment

    report = run_experiment()
    assert report["all_gates_pass"] is True
    assert report["summaries"]["correction"]["conflict_after"] == "accepted"
    assert report["runs"]["correction"]["boundary"]["query_is_factor"] is False


@pytest.mark.skipif(importlib.util.find_spec("gtsam") is None, reason="GTSAM optional")
def test_factor_table_validation() -> None:
    from factor_graph.gtsam_backend import GTSAMDiscreteBeliefGraph

    graph = GTSAMDiscreteBeliefGraph()
    graph.add_variable("x")
    with pytest.raises(ValueError, match="expected 2"):
        graph.add_factor("bad", ("x",), (1.0,), source="test")


@pytest.mark.skipif(importlib.util.find_spec("gtsam") is None, reason="GTSAM optional")
def test_overlay_adapter_replays_verifier_and_matches_python_bp() -> None:
    from factor_graph.overlay_adapter import GTSAMOverlayAdapter

    observations = [
        MemoryNode(
            node_id=f"l1:{name}",
            video_id="video:test",
            time_span=TimeSpan(index * 2.0, index * 2.0 + 1.0),
            provenance={"producer": "test"},
            node_type="observation",
        )
        for index, name in enumerate(("a", "b"))
    ]
    events = [
        MemoryNode(
            node_id=f"event:{name}",
            video_id="video:test",
            time_span=observation.time_span,
            provenance={"producer": "test"},
            node_type="atomic_event",
            source_segments=[observation.node_id],
        )
        for name, observation in zip(("a", "b"), observations, strict=True)
    ]
    overlay = CausalTemporalOverlay(
        overlay_id="overlay:test",
        example_id="example:test",
        video_id="video:test",
        l1_observations=observations,
        atomic_events=events,
        relations=[
            RelationBelief(
                edge_id="edge:identity",
                src="event:a",
                dst="event:b",
                relation_probabilities={"same_entity": 0.55},
                status=RelationStatus.UNCALIBRATED_PRIOR,
                direction_confidence=0.5,
                provenance={
                    "hard_verifier": {
                        "same_entity": {"passed": True, "reasons": []}
                    }
                },
            ),
            RelationBelief(
                edge_id="edge:state",
                src="event:a",
                dst="event:b",
                relation_probabilities={"state_transition": 0.55},
                status=RelationStatus.UNCALIBRATED_PRIOR,
                direction_confidence=0.5,
            ),
        ],
    )
    adapter = GTSAMOverlayAdapter()
    before = adapter.infer(overlay, activate_verified_measurements=False)
    after = adapter.infer(overlay, activate_verified_measurements=True)
    parity = adapter.parity(overlay)

    identity = "relation::edge:identity::same_entity"
    state = "relation::edge:state::state_transition"
    assert after.posteriors[identity] > before.posteriors[identity]
    assert after.posteriors[state] > before.posteriors[state]
    assert after.max_component_variables == 2
    assert parity.max_absolute_error <= 1e-3
    assert parity.categorical_mismatches == ()
