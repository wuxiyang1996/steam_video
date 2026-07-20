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
from memory_graph.navigation import GraphReadAction, NavigationActionType
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    GraphReadExecution,
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


@pytest.mark.skipif(importlib.util.find_spec("gtsam") is None, reason="GTSAM optional")
def test_executed_measurement_session_is_grounded_idempotent_and_propagates() -> None:
    from factor_graph.measurement import (
        MeasurementCalibrationRegistry,
        MeasurementOutcome,
        VerifierDecision,
    )
    from factor_graph.session import GTSAMBeliefSession

    l1_a = MemoryNode(
        node_id="l1:a",
        video_id="video:session",
        time_span=TimeSpan(0.0, 1.0),
        provenance={"producer": "test"},
        node_type="observation",
    )
    l1_b = MemoryNode(
        node_id="l1:b",
        video_id="video:session",
        time_span=TimeSpan(2.0, 3.0),
        provenance={"producer": "test"},
        node_type="observation",
    )
    event_a = MemoryNode(
        node_id="event:a",
        video_id="video:session",
        time_span=l1_a.time_span,
        provenance={"producer": "test"},
        node_type="atomic_event",
        source_segments=[l1_a.node_id],
    )
    event_b = MemoryNode(
        node_id="event:b",
        video_id="video:session",
        time_span=l1_b.time_span,
        provenance={"producer": "test"},
        node_type="atomic_event",
        source_segments=[l1_b.node_id],
    )
    overlay = CausalTemporalOverlay(
        overlay_id="overlay:session",
        example_id="example:session",
        video_id="video:session",
        l1_observations=[l1_a, l1_b],
        atomic_events=[event_a, event_b],
        relations=[
            RelationBelief(
                edge_id="edge:identity",
                src="event:a",
                dst="event:b",
                relation_probabilities={"same_entity": 0.2},
                status=RelationStatus.UNCALIBRATED_PRIOR,
                direction_confidence=0.5,
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
    action = GraphReadAction(
        NavigationActionType.TRACK_ENTITY,
        source_id="event:a",
        target_ids=("event:b",),
        relation="same_entity",
    )
    execution = GraphReadExecution(
        observations=(event_b,),
        skill_invocation={
            "node_id": "skill:track:a-b",
            "status": "executed",
            "outputs": {"real_observation_ids": ["event:b"]},
            "evidence_refs": ["l1:b"],
        },
    )
    decision = VerifierDecision(
        edge_id="edge:identity",
        relation="same_entity",
        outcome=MeasurementOutcome.SUPPORTS,
        verifier_name="identity_test_verifier",
        verifier_version="v1",
        evidence_refs=("l1:b",),
    )
    session = GTSAMBeliefSession(
        overlay,
        calibration=MeasurementCalibrationRegistry.pilot(),
        activate_persisted_verified_measurements=False,
    )
    update = session.update(action=action, execution=execution, decision=decision)
    repeated = session.update(action=action, execution=execution, decision=decision)

    identity = "relation::edge:identity::same_entity"
    state = "relation::edge:state::state_transition"
    assert update.appended is True
    assert update.direct_changed_variables == (identity,)
    assert state in update.propagated_changed_variables
    assert repeated.appended is False
    assert repeated.changed_variables == ()
    assert len(session.journal.measurements) == 1
    record = session.journal.to_records()[0]
    assert record["outcome"] == "supports"
    assert "confidence" not in record


def test_measurement_rejects_unexecuted_or_ungrounded_evidence() -> None:
    from factor_graph.measurement import (
        MeasurementOutcome,
        VerifierDecision,
        measurement_from_execution,
    )

    node_a = MemoryNode(
        node_id="event:a",
        video_id="video:guard",
        time_span=TimeSpan(0.0, 1.0),
        provenance={"producer": "test"},
        node_type="atomic_event",
    )
    node_b = MemoryNode(
        node_id="event:b",
        video_id="video:guard",
        time_span=TimeSpan(2.0, 3.0),
        provenance={"producer": "test"},
        node_type="atomic_event",
    )
    overlay = CausalTemporalOverlay(
        overlay_id="overlay:guard",
        example_id="example:guard",
        video_id="video:guard",
        l1_observations=[],
        atomic_events=[node_a, node_b],
        relations=[
            RelationBelief(
                edge_id="edge:guard",
                src="event:a",
                dst="event:b",
                relation_probabilities={"same_entity": 0.5},
                status=RelationStatus.UNCALIBRATED_PRIOR,
                direction_confidence=0.5,
            )
        ],
    )
    action = GraphReadAction(
        NavigationActionType.TRACK_ENTITY,
        source_id="event:a",
        target_ids=("event:b",),
        relation="same_entity",
    )
    decision = VerifierDecision(
        edge_id="edge:guard",
        relation="same_entity",
        outcome=MeasurementOutcome.SUPPORTS,
        verifier_name="guard",
        verifier_version="v1",
        evidence_refs=("event:b",),
    )
    execution = GraphReadExecution(
        observations=(node_b,),
        skill_invocation={
            "node_id": "skill:guard",
            "status": "insufficient",
            "outputs": {"real_observation_ids": ["event:b"]},
        },
    )
    with pytest.raises(ValueError, match="successfully executed"):
        measurement_from_execution(
            action=action,
            execution=execution,
            overlay=overlay,
            decision=decision,
            calibration_version="pilot",
        )
