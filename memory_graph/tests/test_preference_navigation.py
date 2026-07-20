from __future__ import annotations

from dataclasses import fields

from memory_graph.navigation import NavigationActionType, propose_navigation_actions
from memory_graph.types import (
    CausalTemporalOverlay,
    MemoryNode,
    RelationBelief,
    RelationStatus,
    TimeSpan,
)
from steam_video_new.implicit_world_model.l15_graph_navigator import (
    Answerability,
    ClosedLoopNavigator,
    FactorizedBeliefBackend,
    PreferenceOnlyPlanner,
    RelationGrounding,
    RuleBasedObservationBeliefModel,
    RuleBasedTrajectoryPreferenceModel,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    PlanDecision,
    PredictedTransition,
    TrajectoryPrediction,
)


def _node(
    node_id: str,
    text: str,
    start_s: float,
    *,
    node_type: str,
    source_segments: list[str] | None = None,
) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        video_id="video:preference",
        time_span=TimeSpan(start_s, start_s + 1.0),
        provenance={"created_by": "test"},
        node_type=node_type,
        text=text,
        source_segments=source_segments or [],
    )


def _overlay() -> CausalTemporalOverlay:
    l1_push = _node(
        "l1:push",
        "A person visibly pushes the door.",
        0.0,
        node_type="observation",
    )
    l1_open = _node(
        "l1:open",
        "The door visibly opens.",
        2.0,
        node_type="observation",
    )
    push = _node(
        "event:push",
        "A person pushes the door.",
        0.0,
        node_type="atomic_event",
        source_segments=[l1_push.node_id],
    )
    opened = _node(
        "event:open",
        "The door opens.",
        2.0,
        node_type="atomic_event",
        source_segments=[l1_open.node_id],
    )
    relation = RelationBelief(
        edge_id="edge:push-open",
        src=push.node_id,
        dst=opened.node_id,
        relation_probabilities={"enables": 0.82, "before": 1.0},
        status=RelationStatus.UNCALIBRATED_PRIOR,
        direction_confidence=0.9,
        evidence_refs=[l1_push.node_id, l1_open.node_id],
        features={"correlation": 0.63},
        provenance={
            "hard_verifier": {
                "enables": {"passed": True, "reasons": []},
            }
        },
    )
    return CausalTemporalOverlay(
        overlay_id="overlay:preference",
        example_id="example:preference",
        video_id="video:preference",
        l1_observations=[l1_push, l1_open],
        atomic_events=[push, opened],
        relations=[relation],
    )


def _planner(*, horizon: int = 1) -> PreferenceOnlyPlanner:
    return PreferenceOnlyPlanner(
        RuleBasedObservationBeliefModel(),
        RuleBasedTrajectoryPreferenceModel(),
        horizon=horizon,
    )


def test_factorized_backend_keeps_probabilities_as_relation_metadata() -> None:
    overlay = _overlay()
    backend = FactorizedBeliefBackend()
    belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )

    relation = belief.relation_states[0]
    assert relation.grounding is RelationGrounding.PARTIAL
    assert dict(relation.relation_probabilities)["enables"] == 0.82
    assert dict(relation.correlation_features)["correlation"] == 0.63
    assert belief.answerability is Answerability.NOT_READY


def test_candidate_causal_edges_propose_directional_actions() -> None:
    overlay = _overlay()
    backend = FactorizedBeliefBackend()
    effect_belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    cause_belief = backend.initialize(
        "What happened after the push?",
        overlay,
        seed_evidence=("event:push",),
        missing_roles=("dependency",),
    )

    effect_actions = propose_navigation_actions(
        effect_belief.navigation_view(), overlay
    )
    cause_actions = propose_navigation_actions(
        cause_belief.navigation_view(), overlay
    )

    candidate = next(
        action
        for action in effect_actions
        if action.action_type is NavigationActionType.CANDIDATE_CAUSE
    )
    effect = next(
        action
        for action in cause_actions
        if action.action_type is NavigationActionType.EFFECT
    )
    assert candidate.target_ids == ("event:push",)
    assert effect.target_ids == ("event:open",)


def test_planner_uses_pairwise_preference_without_numeric_value_output() -> None:
    overlay = _overlay()
    belief = FactorizedBeliefBackend().initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )

    decision = _planner().plan(belief, overlay)

    assert decision.selected_action.action_type is NavigationActionType.CANDIDATE_CAUSE
    assert decision.selected_action.target_ids == ("event:push",)
    assert decision.comparisons
    banned = {"reward", "utility", "q_value", "score", "delayed_utility"}
    for contract in (PredictedTransition, TrajectoryPrediction, PlanDecision):
        assert banned.isdisjoint(field.name for field in fields(contract))


def test_closed_loop_executes_real_read_then_replans_and_exports_l2_trace() -> None:
    overlay = _overlay()
    backend = FactorizedBeliefBackend()
    initial = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
        graph_read_budget=3,
    )
    navigator = ClosedLoopNavigator(backend, _planner(horizon=2))

    first_decision = navigator.planner.plan(initial, overlay)
    assert "event:push" not in initial.acquired_evidence
    assert first_decision.selected_action.target_ids == ("event:push",)

    run = navigator.run(initial, overlay, max_steps=3)

    assert "event:push" in run.final_belief.acquired_evidence
    assert run.final_belief.relation_states[0].grounding is RelationGrounding.VERIFIED
    assert run.final_belief.answerability is Answerability.READY
    assert len(run.steps) == 2
    assert run.steps[0].observation_ids == ("event:push",)
    assert run.steps[1].decision.selected_action.action_type is NavigationActionType.STOP
    rollout = run.to_l2_rollout()
    assert rollout[0]["real_observation_ids"] == ["event:push"]
    assert rollout[0]["preference_output"] == "ordinal_only"
    assert rollout[0]["realized_belief_delta"]["predicted_only"] is False
    assert rollout[1]["real_observation_ids"] == []


def test_unverified_candidate_edge_does_not_close_dependency_role() -> None:
    overlay = _overlay()
    overlay.relations[0].provenance = {}
    backend = FactorizedBeliefBackend()
    initial = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    candidate = next(
        action
        for action in propose_navigation_actions(initial.navigation_view(), overlay)
        if action.action_type is NavigationActionType.CANDIDATE_CAUSE
    )

    prediction = RuleBasedObservationBeliefModel().predict(
        initial, candidate, overlay
    )
    observation = [node for node in overlay.atomic_events if node.node_id == "event:push"]
    update = backend.update(initial, candidate, observation, overlay)

    assert prediction.belief_delta.resolved_roles == ()
    assert update.delta.resolved_roles == ()
    assert update.belief.answerability is Answerability.NOT_READY
    assert (
        update.belief.relation_states[0].grounding
        is RelationGrounding.ENDPOINTS_OBSERVED
    )
