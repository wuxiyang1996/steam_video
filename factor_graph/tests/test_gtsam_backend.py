from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

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


class _SequenceVerifier:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def verify(self, *, edge_id, action, **kwargs):
        from factor_graph.measurement import VerifierDecision

        return VerifierDecision(
            edge_id=edge_id,
            relation=action.relation,
            outcome=next(self.outcomes),
            verifier_name="test_sequence_verifier",
            verifier_version="v1",
            evidence_refs=("l1:a", "l1:b"),
            reasons=("controlled categorical test decision",),
        )


def _persistent_navigation_overlay() -> CausalTemporalOverlay:
    observations = [
        MemoryNode(
            node_id=f"l1:{name}",
            video_id="video:persistent",
            time_span=TimeSpan(index * 2.0, index * 2.0 + 1.0),
            provenance={"producer": "test"},
            node_type="observation",
        )
        for index, name in enumerate(("a", "b"))
    ]
    events = [
        MemoryNode(
            node_id=f"event:{name}",
            video_id="video:persistent",
            time_span=observation.time_span,
            provenance={"producer": "test"},
            node_type="atomic_event",
            source_segments=[observation.node_id],
        )
        for name, observation in zip(("a", "b"), observations, strict=True)
    ]
    return CausalTemporalOverlay(
        overlay_id="overlay:persistent",
        example_id="example:persistent",
        video_id="video:persistent",
        l1_observations=observations,
        atomic_events=events,
        relations=[
            RelationBelief(
                edge_id="edge:identity",
                src="event:a",
                dst="event:b",
                relation_probabilities={"same_entity": 0.2},
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


@pytest.mark.skipif(importlib.util.find_spec("gtsam") is None, reason="GTSAM optional")
def test_navigation_backend_corrects_after_execution_while_frozen_does_not() -> None:
    from factor_graph.navigation_backend import GTSAMExecutedReadBeliefBackend
    from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
        load_overlay_artifact,
    )
    from steam_video_new.implicit_world_model.l15_graph_navigator.planner import (
        PersistedGraphReadExecutor,
    )

    overlay = load_overlay_artifact(
        Path(__file__).resolve().parents[1] / "fixtures" / "phase_d_coupled_overlay.json"
    ).overlay
    action = GraphReadAction(
        NavigationActionType.FOLLOW_DEPENDENCY,
        source_id="event:atomic:video_skills_l1:00140",
        target_ids=("event:atomic:video_skills_l1:00258",),
        relation="transition_support",
    )
    results = {}
    for mode in ("correct", "frozen"):
        backend = GTSAMExecutedReadBeliefBackend(mode=mode)
        belief = backend.initialize(
            "What does the state change support?",
            overlay,
            seed_evidence=("event:atomic:video_skills_l1:00140",),
            missing_roles=("dependency",),
            graph_read_budget=2,
        )
        execution = PersistedGraphReadExecutor().execute(belief, action, overlay)
        results[mode] = backend.update_from_execution(
            belief, action, execution, overlay
        )

    assert results["correct"].audit_record["measurement_status"] == "appended"
    assert results["correct"].belief.missing_roles == ()
    assert results["correct"].belief.answerability.value == "ready"
    assert results["frozen"].audit_record["measurement_status"] == "frozen"
    assert results["frozen"].belief.missing_roles == ("dependency",)


@pytest.mark.skipif(importlib.util.find_spec("gtsam") is None, reason="GTSAM optional")
def test_navigation_belief_excludes_persisted_verifier_and_persists_correction() -> None:
    from factor_graph.measurement import MeasurementOutcome
    from factor_graph.navigation_backend import GTSAMExecutedReadBeliefBackend

    overlay = _persistent_navigation_overlay()
    backend = GTSAMExecutedReadBeliefBackend(
        verifier=_SequenceVerifier((MeasurementOutcome.SUPPORTS,))
    )
    initial = backend.initialize(
        "Who is the same person?",
        overlay,
        seed_evidence=("event:a",),
        missing_roles=("identity",),
        graph_read_budget=3,
    )
    assert all(not state.verified_relations for state in initial.relation_states)
    assert all("hard_verified_relation" not in state.factor_sources for state in initial.relation_states)

    action = GraphReadAction(
        NavigationActionType.TRACK_ENTITY,
        source_id="event:a",
        target_ids=("event:b",),
        relation="same_entity",
    )
    execution = GraphReadExecution(
        observations=(overlay.atomic_events[1],),
        skill_invocation={
            "node_id": "skill:identity-support",
            "status": "executed",
            "outputs": {"real_observation_ids": ["event:b"]},
            "evidence_refs": ["l1:b"],
        },
    )
    corrected = backend.update_from_execution(initial, action, execution, overlay)
    assert corrected.belief.missing_roles == ()
    assert corrected.delta.resolved_roles == ("identity",)
    assert corrected.delta.answerability_after is corrected.belief.answerability

    semantic = GraphReadAction(
        NavigationActionType.SEMANTIC,
        source_id="event:b",
        target_ids=("l1:b",),
    )
    second_execution = GraphReadExecution(
        observations=(overlay.l1_observations[1],),
        skill_invocation={
            "node_id": "skill:semantic-after-correction",
            "status": "executed",
            "outputs": {"real_observation_ids": ["l1:b"]},
            "evidence_refs": ["l1:b"],
        },
    )
    persisted = backend.update_from_execution(
        corrected.belief, semantic, second_execution, overlay
    )
    identity = next(
        state for state in persisted.belief.relation_states if state.edge_id == "edge:identity"
    )
    assert identity.verified_relations == ("same_entity",)
    assert persisted.belief.missing_roles == ()


@pytest.mark.skipif(importlib.util.find_spec("gtsam") is None, reason="GTSAM optional")
def test_navigation_reject_inconclusive_and_conflict_are_persistent() -> None:
    from factor_graph.measurement import MeasurementOutcome
    from factor_graph.navigation_backend import GTSAMExecutedReadBeliefBackend

    overlay = _persistent_navigation_overlay()
    action = GraphReadAction(
        NavigationActionType.TRACK_ENTITY,
        source_id="event:a",
        target_ids=("event:b",),
        relation="same_entity",
    )

    inconclusive_backend = GTSAMExecutedReadBeliefBackend(
        verifier=_SequenceVerifier((MeasurementOutcome.INCONCLUSIVE,))
    )
    inconclusive_initial = inconclusive_backend.initialize(
        "Who is the same person?",
        overlay,
        seed_evidence=("event:a",),
        missing_roles=("identity",),
        graph_read_budget=2,
    )
    inconclusive_execution = GraphReadExecution(
        observations=(overlay.atomic_events[1],),
        skill_invocation={
            "node_id": "skill:identity-inconclusive",
            "status": "executed",
            "outputs": {"real_observation_ids": ["event:b"]},
            "evidence_refs": ["l1:b"],
        },
    )
    inconclusive = inconclusive_backend.update_from_execution(
        inconclusive_initial, action, inconclusive_execution, overlay
    )
    assert inconclusive.audit_record["factor_activated"] is False
    assert inconclusive.audit_record["changed_variables"] == []
    assert inconclusive.belief.missing_roles == ("identity",)

    conflict_backend = GTSAMExecutedReadBeliefBackend(
        verifier=_SequenceVerifier(
            (MeasurementOutcome.SUPPORTS, MeasurementOutcome.REJECTS)
        )
    )
    conflict_initial = conflict_backend.initialize(
        "Who is the same person?",
        overlay,
        seed_evidence=("event:a",),
        missing_roles=("identity",),
        graph_read_budget=3,
    )
    support_execution = replace(
        inconclusive_execution,
        skill_invocation={
            **inconclusive_execution.skill_invocation,
            "node_id": "skill:identity-support-then-reject",
        },
    )
    supported = conflict_backend.update_from_execution(
        conflict_initial, action, support_execution, overlay
    )
    reject_execution = replace(
        inconclusive_execution,
        skill_invocation={
            **inconclusive_execution.skill_invocation,
            "node_id": "skill:identity-reject-after-support",
        },
    )
    conflicted = conflict_backend.update_from_execution(
        supported.belief, action, reject_execution, overlay
    )
    assert conflicted.belief.missing_roles == ("identity",)
    assert "edge:identity" in conflicted.belief.contradictions
    assert "edge:identity" in conflicted.belief.blocked_edge_ids
    assert conflicted.delta.answerability_after is conflicted.belief.answerability
    assert len(conflict_backend.session.journal.measurements) == 2


@pytest.mark.skipif(importlib.util.find_spec("gtsam") is None, reason="GTSAM optional")
def test_correction_sensitive_navigation_separates_propagation_from_verifier(
    tmp_path: Path,
) -> None:
    from factor_graph.correction_sensitive_experiment import run_experiment

    report = run_experiment(
        Path(__file__).resolve().parents[1] / "fixtures" / "phase_d_coupled_overlay.json",
        tmp_path,
    )

    assert report["summary"] == {
        "case_count": 3,
        "support_propagation_changes_next_action": True,
        "reject_propagation_changes_next_action": True,
        "inconclusive_is_negative_control": True,
    }
    support, reject, inconclusive = report["cases"]
    assert support["arms"]["correct"]["probe_outcome"] == "supports"
    assert reject["arms"]["correct"]["probe_outcome"] == "rejects"
    assert inconclusive["arms"]["correct"]["probe_outcome"] == "inconclusive"
    assert (
        support["arms"]["correct"]["next_action"]
        != support["arms"]["verifier_only"]["next_action"]
    )
    assert (
        inconclusive["arms"]["correct"]["next_action"]
        == inconclusive["arms"]["verifier_only"]["next_action"]
    )


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


def test_post_read_verifier_uses_acquired_structure_not_persisted_decision() -> None:
    from factor_graph.measurement import MeasurementOutcome
    from factor_graph.post_read_verifier import PostReadCategoricalVerifier

    track = "l1-track:person"
    l1_a = MemoryNode(
        "l1:a",
        "video:post-read",
        TimeSpan(0.0, 1.0),
        {"producer": "test"},
        node_type="observation",
    )
    l1_b = MemoryNode(
        "l1:b",
        "video:post-read",
        TimeSpan(2.0, 3.0),
        {"producer": "test"},
        node_type="observation",
    )
    event_a = MemoryNode(
        "event:a",
        "video:post-read",
        l1_a.time_span,
        {"producer": "test"},
        node_type="atomic_event",
        source_segments=[l1_a.node_id],
        metadata={
            "participants": [
                {"mention_id": track, "entity_type": "person", "surface": "man"}
            ],
            "states": [
                {
                    "mention_id": track,
                    "attribute": "expression",
                    "value": "neutral",
                    "polarity": "positive",
                }
            ],
        },
    )
    event_b = MemoryNode(
        "event:b",
        "video:post-read",
        l1_b.time_span,
        {"producer": "test"},
        node_type="atomic_event",
        source_segments=[l1_b.node_id],
        metadata={
            "participants": [
                {"mention_id": track, "entity_type": "person", "surface": "man"}
            ],
            "states": [
                {
                    "mention_id": track,
                    "attribute": "expression",
                    "value": "focused",
                    "polarity": "positive",
                }
            ],
        },
    )
    edge = RelationBelief(
        "edge:state",
        event_a.node_id,
        event_b.node_id,
        {"state_transition": 0.2},
        RelationStatus.UNCALIBRATED_PRIOR,
        0.5,
        provenance={
            "participant_alignment": {"src": track, "dst": track},
            "hard_verifier": {
                "state_transition": {
                    "passed": False,
                    "reasons": ["deliberately stale persisted result"],
                }
            },
        },
    )
    overlay = CausalTemporalOverlay(
        "overlay:post-read",
        "example:post-read",
        "video:post-read",
        [l1_a, l1_b],
        [event_a, event_b],
        [edge],
    )
    action = GraphReadAction(
        NavigationActionType.INSPECT_STATE_CHANGE,
        event_a.node_id,
        (event_b.node_id,),
        "state_transition",
    )
    execution = GraphReadExecution(
        (event_b,),
        {
            "node_id": "skill:state:a-b",
            "status": "executed",
            "outputs": {"real_observation_ids": [event_b.node_id]},
            "evidence_refs": [l1_b.node_id],
        },
    )

    decision = PostReadCategoricalVerifier().verify(
        action=action,
        execution=execution,
        overlay=overlay,
        edge_id=edge.edge_id,
        acquired_observation_ids=(event_a.node_id,),
    )

    assert decision.outcome is MeasurementOutcome.SUPPORTS
    assert decision.evidence_refs == (l1_a.node_id, l1_b.node_id)
    assert all("stale" not in reason for reason in decision.reasons)


def test_post_read_verifier_requires_both_endpoints_to_be_observed() -> None:
    from factor_graph.post_read_verifier import PostReadCategoricalVerifier

    first = MemoryNode(
        "event:a",
        "video:guard",
        TimeSpan(0.0, 1.0),
        {"producer": "test"},
        node_type="atomic_event",
    )
    second = MemoryNode(
        "event:b",
        "video:guard",
        TimeSpan(2.0, 3.0),
        {"producer": "test"},
        node_type="atomic_event",
    )
    edge = RelationBelief(
        "edge:guard",
        first.node_id,
        second.node_id,
        {"same_entity": 0.5},
        RelationStatus.UNCALIBRATED_PRIOR,
        0.5,
    )
    overlay = CausalTemporalOverlay(
        "overlay:guard-post-read",
        "example:guard",
        "video:guard",
        [],
        [first, second],
        [edge],
    )
    action = GraphReadAction(
        NavigationActionType.TRACK_ENTITY,
        first.node_id,
        (second.node_id,),
        "same_entity",
    )
    execution = GraphReadExecution(
        (second,),
        {
            "node_id": "skill:guard-post-read",
            "status": "executed",
            "outputs": {"real_observation_ids": [second.node_id]},
        },
    )

    with pytest.raises(ValueError, match="both verifier endpoints"):
        PostReadCategoricalVerifier().verify(
            action=action,
            execution=execution,
            overlay=overlay,
            edge_id=edge.edge_id,
            acquired_observation_ids=(),
        )


def test_phase_d_fixture_has_verified_coupling_and_qwen_embedding_refs() -> None:
    from memory_graph.verifiers import verify_relation
    from steam_video_new.implicit_world_model.l15_graph_navigator import (
        load_overlay_artifact,
    )

    fixture = Path(__file__).parents[1] / "fixtures" / "phase_d_coupled_overlay.json"
    overlay = load_overlay_artifact(
        fixture,
        require_embedding_files=True,
        verify_embedding_checksums=True,
    ).overlay
    events = {node.node_id: node for node in overlay.atomic_events}
    decisions = {
        relation: verify_relation(edge, events[edge.src], events[edge.dst], relation)
        for edge in overlay.relations
        for relation in edge.relation_probabilities
    }

    assert set(decisions) == {
        "same_entity",
        "state_transition",
        "transition_support",
        "contradicts",
    }
    assert all(decision.passed for decision in decisions.values())
    assert "expression" in decisions["state_transition"].reasons[0]
    assert all(
        node.embedding_ref is not None
        and Path(node.embedding_ref.path).is_absolute()
        and node.embedding_ref.model == "Qwen/Qwen3-VL-Embedding-2B"
        for node in overlay.atomic_events
    )


@pytest.mark.skipif(importlib.util.find_spec("gtsam") is None, reason="GTSAM optional")
def test_phase_d_grounded_experiment_passes_all_gates() -> None:
    from factor_graph.phase_d_experiment import run_experiment

    fixture = Path(__file__).parents[1] / "fixtures" / "phase_d_coupled_overlay.json"
    report = run_experiment(fixture)

    assert report["all_gates_pass"] is True
    assert report["arms"]["normal"]["decision"]["outcome"] == "supports"
    assert "confidence" not in report["arms"]["normal"]["decision"]
