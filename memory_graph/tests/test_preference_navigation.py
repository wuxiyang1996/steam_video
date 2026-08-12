from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_graph.navigation import (
    GraphReadAction,
    NavigationActionType,
    propose_navigation_actions,
)
from memory_graph.categorical_confusion import (
    admitted_candidate_predictions,
    categorical_confusion_report,
)
from memory_graph.types import (
    CausalTemporalOverlay,
    EmbeddingRef,
    MemoryNode,
    RelationBelief,
    RelationStatus,
    TimeSpan,
)
from steam_video_new.implicit_world_model.l15_graph_navigator import (
    Answerability,
    apply_evidence_review,
    BALANCED_CASE_CATEGORIES,
    ClosedLoopNavigator,
    FactorizedBeliefBackend,
    FactorGraphBeliefBackend,
    FrozenBeliefWorldModel,
    PreferenceOnlyPlanner,
    ReasoningContextBudget,
    ReasoningContextBuilder,
    ReasoningHopType,
    RelationGrounding,
    RuleBasedObservationBeliefModel,
    RuleBasedTrajectoryPreferenceModel,
    TransitionIntervention,
    VideoSkillsL2Adapter,
    build_executed_transition_dataset,
    build_transition_review_packet,
    build_visual_review_bundle,
    build_video_skills_l2_rollout,
    build_balanced_evidence_packet,
    derive_realized_belief_delta,
    export_executed_transition_training_records,
    export_reviewed_balanced_case_set,
    generate_sibling_artifact,
    guided_navigation_actions,
    load_overlay_artifact,
    lock_executed_transition_dataset,
    lock_balanced_review_queue,
    lock_balanced_evidence_packet,
    mine_balanced_reasoning_cases,
    import_evidence_annotations,
    inspect_balanced_evidence_packet,
    inspect_transition_gathering,
    inspect_transition_review,
    validate_executed_transition_dataset,
    validate_transition_review_packet,
    validate_visual_review_bundle,
    validate_balanced_review_queue,
    validate_balanced_evidence_packet,
    validate_sibling_artifact,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.gpt_oss import (
    GPTOSSObservationBeliefModel,
    GPTOSSTrajectoryPreferenceModel,
    OpenAICompatibleCategoricalClient,
    _reject_numeric_output,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.matched_ablation import (
    MATCHED_STRATEGIES,
    evaluate_matched_navigation,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.case_miner import (
    mine_navigation_cases,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.preference_data import (
    build_preference_annotation_packet,
    export_training_records,
    lock_annotation_packet,
    lock_navigation_case_set,
    validate_navigation_case_set,
    validate_preference_annotation_packet,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.preference_review import (
    apply_preference_review,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.provisional_review import (
    apply_provisional_case_review,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.train_models import (
    predict_preference_label,
    predict_transition_labels,
    train_baselines,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    PairwisePreference,
    PlanDecision,
    PredictedTransition,
    PreferenceLabel,
    TrajectoryPrediction,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.run import main as run_main


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
        metadata={
            "layer_contract": "l1_observations_plus_l1_5_atomic_overlay",
            "input_mode": "video_only",
            "source_l1_graph_id": "clue_memory:preference",
        },
    )


def _planner(*, horizon: int = 1) -> PreferenceOnlyPlanner:
    return PreferenceOnlyPlanner(
        RuleBasedObservationBeliefModel(),
        RuleBasedTrajectoryPreferenceModel(),
        horizon=horizon,
    )


class _FakeCategoricalClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def complete_json(self, *, task: str, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append({"task": task, "payload": payload})
        return self.responses.pop(0)


def test_reasoning_context_bounds_graph_candidates_and_comparisons() -> None:
    overlay = _overlay()
    for node in overlay.atomic_events:
        node.embedding_ref = EmbeddingRef(
            path="event_embeddings.npy",
            model="Qwen/Qwen3-VL-Embedding-2B",
            row_index=0,
        )
    backend = FactorGraphBeliefBackend()
    belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    planner = PreferenceOnlyPlanner(
        RuleBasedObservationBeliefModel(),
        RuleBasedTrajectoryPreferenceModel(),
        horizon=1,
        context_builder=ReasoningContextBuilder(
            ReasoningContextBudget(
                max_nodes=2,
                max_edges=1,
                max_candidate_hops=2,
                max_comparisons=1,
                recent_hop_window=1,
            )
        ),
    )

    decision = planner.plan(belief, overlay, recent_hops=("read_event", "verify_relation"))

    context = decision.reasoning_context
    assert context is not None
    assert len(context.local_node_ids) <= 2
    assert len(context.local_edge_ids) <= 1
    assert len(context.candidate_hops) <= 2
    assert len(decision.comparisons) <= 1
    assert context.recent_hops == ("verify_relation",)
    assert context.audit.embedding_models_available == (
        "Qwen/Qwen3-VL-Embedding-2B",
    )
    assert decision.selected_hop.hop_type in set(ReasoningHopType)
    assert "embedding_ref" not in json.dumps(
        {
            "nodes": context.local_node_ids,
            "audit": context.audit.embedding_models_available,
        }
    )


def test_gpt_oss_models_accept_only_categorical_descriptors() -> None:
    overlay = _overlay()
    backend = FactorGraphBeliefBackend()
    belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    action = next(
        action
        for action in guided_navigation_actions(belief, overlay)
        if action.action_type is NavigationActionType.CANDIDATE_CAUSE
    )
    client = _FakeCategoricalClient(
        [
                {
                    "node_kind": "atomic_event",
                "resolved_roles": ["dependency"],
                "relation_updates": ["edge:push-open"],
                "hypothesis_updates": [
                    {
                        "edge_id": "edge:push-open",
                        "disposition": "accepted",
                    }
                ],
                "contradiction_updates": [],
                "frontier_change": "opened",
                "contradiction_change": "unchanged",
                "path_change": "opened",
                "recovery_status": "recovered",
                "uncertainty_change": "decrease",
                "answerability_after": "ready",
            },
            {"label": "prefer_left", "rationale": "left completes the missing role"},
        ]
    )
    world_model = GPTOSSObservationBeliefModel(client)  # type: ignore[arg-type]
    transition = world_model.predict(belief, action, overlay)
    trajectory = TrajectoryPrediction("left", (transition,))
    stop = GraphReadAction(NavigationActionType.STOP)
    stop_transition = PredictedTransition(
        stop,
        RuleBasedObservationBeliefModel().predict(belief, stop, overlay).observation,
        RuleBasedObservationBeliefModel().predict(belief, stop, overlay).belief_delta,
    )
    preference = GPTOSSTrajectoryPreferenceModel(client).compare(  # type: ignore[arg-type]
        trajectory,
        TrajectoryPrediction("right", (stop_transition,)),
        belief,
    )

    assert transition.belief_delta.resolved_roles == ("dependency",)
    assert transition.belief_delta.hypothesis_updates[0].edge_id == "edge:push-open"
    assert preference.label.value == "prefer_left"
    assert len(client.requests) == 2
    world_payload = client.requests[0]["payload"]
    assert isinstance(world_payload, dict)
    assert "posterior_probabilities" not in json.dumps(world_payload)
    assert "relation_probabilities" not in json.dumps(world_payload)
    preference_payload = client.requests[1]["payload"]
    assert isinstance(preference_payload, dict)
    serialized_preference = json.dumps(
        {"belief": preference_payload["belief"],
         "left": preference_payload["left"],
         "right": preference_payload["right"]}
    )
    for forbidden in (
        "operator",
        "target_ids",
        "trajectory_id",
        "relation_updates",
        "edge:push-open",
        "rationale",
    ):
        assert forbidden not in serialized_preference


def test_context_candidate_order_is_permutation_invariant() -> None:
    overlay = _overlay()
    belief = FactorGraphBeliefBackend().initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    actions = guided_navigation_actions(belief, overlay)
    builder = ReasoningContextBuilder(
        ReasoningContextBudget(max_candidate_hops=2, max_comparisons=1)
    )

    forward = builder.build(belief, overlay, actions)
    reversed_order = builder.build(belief, overlay, list(reversed(actions)))

    assert forward.context.candidate_hops == reversed_order.context.candidate_hops
    assert forward.actions == reversed_order.actions


class _TiePreferenceModel:
    def compare(self, left, right, belief) -> PairwisePreference:
        return PairwisePreference(
            left_id=left.trajectory_id,
            right_id=right.trajectory_id,
            label=PreferenceLabel.TIE,
            rationale="indistinguishable imagined outcomes",
        )


def test_non_unique_undominated_actions_cause_explicit_abstention() -> None:
    overlay = _overlay()
    belief = FactorizedBeliefBackend().initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    planner = PreferenceOnlyPlanner(
        RuleBasedObservationBeliefModel(),
        _TiePreferenceModel(),  # type: ignore[arg-type]
        horizon=1,
    )

    decision = planner.plan(belief, overlay)

    assert decision.selected_action.action_type is NavigationActionType.STOP
    assert decision.planning_status == "abstain"
    assert decision.ambiguity_reason == "multiple_undominated_trajectories"


def test_null_wm_intervention_erases_predicted_progress_and_abstains() -> None:
    overlay = _overlay()
    belief = FactorizedBeliefBackend().initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    planner = PreferenceOnlyPlanner(
        RuleBasedObservationBeliefModel(),
        _TiePreferenceModel(),  # type: ignore[arg-type]
        horizon=1,
        transition_intervention=TransitionIntervention.NULL,
    )

    decision = planner.plan(belief, overlay)

    assert decision.planning_status == "abstain"
    assert all(
        transition.observation.role.value == "none"
        and transition.belief_delta.resolved_roles == ()
        for trajectory in decision.trajectories
        for transition in trajectory.transitions
    )


def test_frozen_wm_ignores_later_real_belief_change() -> None:
    overlay = _overlay()
    backend = FactorizedBeliefBackend()
    initial = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    action = next(
        item
        for item in guided_navigation_actions(initial, overlay)
        if item.action_type is NavigationActionType.CANDIDATE_CAUSE
    )
    model = FrozenBeliefWorldModel(RuleBasedObservationBeliefModel())
    first = model.predict(initial, action, overlay)
    observed = next(node for node in overlay.atomic_events if node.node_id == "event:push")
    updated = backend.update(initial, action, [observed], overlay).belief

    frozen = model.predict(updated, action, overlay)
    normal = RuleBasedObservationBeliefModel().predict(updated, action, overlay)

    assert first.belief_delta.uncertainty_change.value == "decrease"
    assert frozen.belief_delta.uncertainty_change.value == "decrease"
    assert normal.belief_delta.uncertainty_change.value == "unchanged"


def test_gpt_oss_numeric_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden numeric value"):
        _reject_numeric_output({"confidence": 0.9})


def test_gpt_oss_stop_is_deterministic_and_never_calls_model() -> None:
    overlay = _overlay()
    belief = FactorGraphBeliefBackend().initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    client = _FakeCategoricalClient([])

    transition = GPTOSSObservationBeliefModel(client).predict(  # type: ignore[arg-type]
        belief,
        GraphReadAction(NavigationActionType.STOP),
        overlay,
    )

    assert transition.observation.role.value == "none"
    assert transition.belief_delta.resolved_roles == ()
    assert transition.belief_delta.relation_updates == ()
    assert transition.belief_delta.answerability_after is belief.answerability
    assert client.requests == []


def test_gpt_oss_openrouter_keys_file_loader(tmp_path: Path) -> None:
    keys_path = tmp_path / "keys.py"
    keys_path.write_text("OPENROUTER_API_KEY = 'test-secret'\n", encoding="utf-8")

    client = OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
        keys_path
    )

    assert client.endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert client.api_key == "test-secret"


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


def test_factor_graph_updates_posteriors_and_prioritizes_unresolved_role() -> None:
    overlay = _overlay()
    backend = FactorGraphBeliefBackend(inference_iterations=6)
    belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )

    relation = belief.relation_states[0]
    posterior = dict(relation.posterior_probabilities)
    assert belief.backend_name == "hybrid_factor_graph/v0.1"
    assert belief.backend_ref is not None
    assert "sum-product" in belief.backend_ref
    assert "continuous=observed_intervals/v0.1:observed:1" in belief.backend_ref
    assert posterior["enables"] > dict(relation.relation_probabilities)["enables"]
    assert "hard_verified_relation" in relation.factor_sources
    assert relation.edge_id in belief.priority_edge_ids

    actions = guided_navigation_actions(belief, overlay)
    assert any(
        action.action_type is NavigationActionType.CANDIDATE_CAUSE
        for action in actions
    )


def test_factor_graph_does_not_promote_unverified_candidate_relation() -> None:
    overlay = _overlay()
    overlay.relations[0].provenance = {}
    backend = FactorGraphBeliefBackend()
    belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    candidate = next(
        action
        for action in guided_navigation_actions(belief, overlay)
        if action.action_type is NavigationActionType.CANDIDATE_CAUSE
    )
    observation = [node for node in overlay.atomic_events if node.node_id == "event:push"]

    update = backend.update(belief, candidate, observation, overlay)

    assert update.delta.resolved_roles == ()
    assert update.belief.answerability is Answerability.NOT_READY
    assert update.belief.relation_states[0].verified_relations == ()
    assert (
        update.belief.relation_states[0].grounding
        is RelationGrounding.ENDPOINTS_OBSERVED
    )


def test_temporal_direction_defines_legal_candidates_without_ranking() -> None:
    overlay = _overlay()
    backend = FactorGraphBeliefBackend()

    after = backend.initialize(
        "What happened after the person pushed the door?",
        overlay,
        seed_evidence=("event:push",),
        missing_roles=("temporal",),
    )
    before = backend.initialize(
        "What happened before the door opened?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("temporal",),
    )

    assert any(
        action.action_type is NavigationActionType.TEMPORAL_FORWARD
        for action in guided_navigation_actions(after, overlay)
    )
    assert any(
        action.action_type is NavigationActionType.TEMPORAL_BACK
        for action in guided_navigation_actions(before, overlay)
    )


def test_factor_graph_propagates_identity_component_contradiction() -> None:
    l1_nodes = [
        _node(f"l1:{name}", f"Observation {name}.", index * 2.0, node_type="observation")
        for index, name in enumerate(("a", "b", "c"))
    ]
    events = [
        _node(
            f"event:{name}",
            f"The same tracked person at {name}.",
            index * 2.0,
            node_type="atomic_event",
            source_segments=[f"l1:{name}"],
        )
        for index, name in enumerate(("a", "b", "c"))
    ]

    def relation(
        edge_id: str,
        src: str,
        dst: str,
        name: str,
        *,
        verified: bool = False,
    ) -> RelationBelief:
        return RelationBelief(
            edge_id=edge_id,
            src=src,
            dst=dst,
            relation_probabilities={name: 0.9},
            status=RelationStatus.UNCALIBRATED_PRIOR,
            direction_confidence=0.9,
            provenance=(
                {"hard_verifier": {name: {"passed": True, "reasons": []}}}
                if verified
                else {}
            ),
        )

    overlay = CausalTemporalOverlay(
        overlay_id="overlay:identity-conflict",
        example_id="example:identity-conflict",
        video_id="video:preference",
        l1_observations=l1_nodes,
        atomic_events=events,
        relations=[
            relation("identity:ab", "event:a", "event:b", "same_entity", verified=True),
            relation("identity:bc", "event:b", "event:c", "same_entity", verified=True),
            relation(
                "conflict:ac",
                "event:a",
                "event:c",
                "contradicts",
                verified=True,
            ),
        ],
        metadata={"layer_contract": "l1_observations_plus_l1_5_atomic_overlay"},
    )
    backend = FactorGraphBeliefBackend()
    belief = backend.initialize(
        "Is this the same person?",
        overlay,
        seed_evidence=("event:a", "event:b"),
        missing_roles=(),
    )

    assert belief.answerability is Answerability.READY
    assert belief.contradictions == ()
    assert next(
        state for state in belief.relation_states if state.edge_id == "identity:ab"
    ).grounding is RelationGrounding.VERIFIED

    update = backend.update(
        belief,
        GraphReadAction(
            action_type=NavigationActionType.SEMANTIC,
            target_ids=("event:c",),
        ),
        [events[2]],
        overlay,
    )
    belief = update.belief

    assert belief.answerability is Answerability.NOT_READY
    assert set(belief.contradictions) == {
        "identity:ab",
        "identity:bc",
        "conflict:ac",
    }
    assert set(belief.blocked_edge_ids) == set(belief.contradictions)
    assert all(
        state.grounding is RelationGrounding.CONTRADICTED
        for state in belief.relation_states
    )
    assert set(update.delta.contradiction_updates) == set(belief.contradictions)
    actions = guided_navigation_actions(belief, overlay)
    assert not any(
        action.action_type is NavigationActionType.TRACK_ENTITY
        for action in actions
    )
    verify = [
        action
        for action in actions
        if action.action_type is NavigationActionType.VERIFY
    ]
    assert verify
    assert "event:b" in verify[0].target_ids
    assert set(verify[0].target_ids) <= {"event:b", "l1:a", "l1:b", "l1:c"}


def test_factor_graph_ignores_unverified_contradiction_candidate() -> None:
    overlay = _overlay()
    overlay.relations[0] = RelationBelief(
        edge_id="candidate:contradiction",
        src="event:push",
        dst="event:open",
        relation_probabilities={"contradicts": 0.9},
        status=RelationStatus.UNCALIBRATED_PRIOR,
        direction_confidence=0.9,
    )

    belief = FactorGraphBeliefBackend().initialize(
        "Did these observations conflict?",
        overlay,
        seed_evidence=("event:push", "event:open"),
        missing_roles=(),
    )

    assert belief.contradictions == ()
    assert belief.blocked_edge_ids == ()
    assert belief.relation_states[0].grounding is RelationGrounding.ENDPOINTS_OBSERVED


def test_factor_graph_blocks_only_edges_on_verified_temporal_cycle() -> None:
    overlay = _overlay()
    extra = _node(
        "event:later",
        "A later event.",
        4.0,
        node_type="atomic_event",
    )
    overlay.atomic_events.append(extra)

    def verified_before(edge_id: str, src: str, dst: str) -> RelationBelief:
        return RelationBelief(
            edge_id=edge_id,
            src=src,
            dst=dst,
            relation_probabilities={"before": 0.9},
            status=RelationStatus.UNCALIBRATED_PRIOR,
            direction_confidence=0.9,
            provenance={
                "hard_verifier": {"before": {"passed": True, "reasons": []}}
            },
        )

    overlay.relations = [
        verified_before("upstream", "event:push", "event:open"),
        verified_before("cycle:forward", "event:open", "event:later"),
        verified_before("cycle:back", "event:later", "event:open"),
    ]
    belief = FactorGraphBeliefBackend().initialize(
        "What happened next?",
        overlay,
        seed_evidence=("event:push", "event:open", "event:later"),
        missing_roles=(),
    )

    assert set(belief.blocked_edge_ids) == {"cycle:forward", "cycle:back"}
    assert "upstream" not in belief.contradictions


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


def test_disabled_hard_verifier_does_not_count_as_verified() -> None:
    overlay = _overlay()
    overlay.relations[0].provenance = {
        "hard_verifier": {
            "enables": {
                "passed": True,
                "reasons": ["hard verification disabled"],
            }
        }
    }
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
    observation = [node for node in overlay.atomic_events if node.node_id == "event:push"]

    update = backend.update(initial, candidate, observation, overlay)

    assert update.delta.resolved_roles == ()
    assert (
        update.belief.relation_states[0].grounding
        is RelationGrounding.ENDPOINTS_OBSERVED
    )


def test_overlay_loader_preserves_qwen_embedding_and_provenance(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    embedding_path = tmp_path / "event_embeddings.npy"
    embedding_path.write_bytes(b"embedding-matrix-placeholder")
    checksum = hashlib.sha256(embedding_path.read_bytes()).hexdigest()
    overlay.atomic_events[0].embedding_ref = EmbeddingRef(
        path=str(embedding_path),
        model="Qwen/Qwen3-VL-Embedding-2B",
        dimension=2048,
        row_index=0,
        checksum=checksum,
    )
    artifact_path = tmp_path / "causal_temporal_overlay.json"
    artifact_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")

    loaded = load_overlay_artifact(
        artifact_path,
        require_embedding_files=True,
        verify_embedding_checksums=True,
    )

    event = loaded.overlay.atomic_events[0]
    assert loaded.embedding_problems == ()
    assert event.embedding_ref is not None
    assert event.embedding_ref.model == "Qwen/Qwen3-VL-Embedding-2B"
    assert event.embedding_ref.row_index == 0
    assert event.provenance["created_by"] == "test"


def test_video_skills_adapter_and_siblings_emit_only_real_grounded_reads() -> None:
    overlay = _overlay()
    backend = FactorizedBeliefBackend()
    belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    adapter = VideoSkillsL2Adapter(use_video_skills_runtime=False)
    planner = _planner(horizon=1)
    run = ClosedLoopNavigator(backend, planner, adapter).run(
        belief,
        overlay,
        max_steps=2,
    )
    siblings = generate_sibling_artifact(
        belief,
        overlay,
        backend=backend,
        executor=adapter,
        world_model=RuleBasedObservationBeliefModel(),
        preference_model=RuleBasedTrajectoryPreferenceModel(),
    )
    l2 = build_video_skills_l2_rollout(
        run,
        overlay,
        question=belief.question,
    )

    first_node = l2["nodes"][0]
    assert first_node["skill_id"] == "retrieve_by_relation"
    assert first_node["outputs"]["real_observation_ids"] == ["event:push"]
    assert first_node["evidence_refs"] == ["l1:push"]
    verifier = first_node["verifier_result"]
    assert verifier["categorical_outcome"] == "supports"
    assert verifier["source"] == "persisted_relation_verifier"
    assert verifier["post_read"] is False
    assert verifier["numeric_output_exposed"] is False
    assert l2["preference_navigation"]["output_contract"] == "ordinal_only"
    assert siblings["annotation_status"] == "requires_independent_annotation"
    assert siblings["pairwise_preferences"]
    assert validate_sibling_artifact(siblings) == []
    assert {
        row["belief_before_id"] for row in siblings["branches"]
    } == {belief.belief_id}
    serialized = json.dumps({"l2": l2, "siblings": siblings}).lower()
    for forbidden in ('"reward"', '"utility"', '"q_value"', '"score"'):
        assert forbidden not in serialized


def test_video_skills_runtime_verify_exposes_only_categorical_measurement() -> None:
    overlay = _overlay()
    belief = FactorizedBeliefBackend().initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:push",),
        missing_roles=("verification",),
    )
    action = GraphReadAction(
        NavigationActionType.VERIFY,
        source_id="event:push",
        target_ids=("event:open",),
        relation="enables",
    )

    execution = VideoSkillsL2Adapter().execute(belief, action, overlay)
    verifier = execution.skill_invocation["verifier_result"]

    assert verifier["post_read"] is True
    assert verifier["source"] == "video_skills_post_read_claim_verifier"
    assert verifier["categorical_outcome"] in {
        "supports", "rejects", "inconclusive"
    }
    assert verifier["numeric_output_exposed"] is False
    serialized = json.dumps(verifier).lower()
    for forbidden in ("score", "confidence", "probability", "utility"):
        assert forbidden not in serialized


def test_video_skills_empty_counterevidence_search_is_grounded_observation() -> None:
    overlay = _overlay()
    belief = FactorizedBeliefBackend().initialize(
        "Could the proposed explanation be contradicted?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("counterevidence",),
    )
    action = GraphReadAction(
        NavigationActionType.SEARCH_COUNTEREVIDENCE,
        source_id="event:open",
        relation="contradicts",
    )
    adapter = VideoSkillsL2Adapter()
    adapter._execute_video_skills = lambda *_: SimpleNamespace(  # type: ignore[method-assign]
        evidence_refs=[],
        skill_id="search_counterevidence",
        ok=False,
        failure_code="no_counterevidence",
        outputs={"evidence_refs": []},
    )

    execution = adapter.execute(belief, action, overlay)

    assert execution.observations == ()
    assert execution.skill_invocation["status"] == "executed"
    assert execution.skill_invocation["search_completed"] is True
    assert execution.skill_invocation["grounded_empty_result"] is True
    assert execution.skill_invocation["evidence_refs"] == ["l1:open"]


def test_cli_writes_navigation_l2_belief_and_sibling_artifacts(
    tmp_path: Path,
) -> None:
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(_overlay().to_dict()), encoding="utf-8")
    output_dir = tmp_path / "run"

    status = run_main(
        [
            "--overlay",
            str(overlay_path),
            "--question",
            "Why did the door open?",
            "--seed-event",
            "event:open",
            "--missing-role",
            "dependency",
            "--horizon",
            "1",
            "--max-steps",
            "2",
            "--output-dir",
            str(output_dir),
            "--no-video-skills-runtime",
        ]
    )

    assert status == 0
    expected = {
        "navigation_run.json",
        "l2_rollout.json",
        "belief_snapshots.jsonl",
        "sibling_checkpoint.json",
        "run_summary.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    summary = json.loads((output_dir / "run_summary.json").read_text())
    assert summary["final_answerability"] == "ready"
    assert summary["preference_output_contract"] == "ordinal_only"
    assert summary["belief_backend"] == "hybrid_factor_graph/v0.1"
    assert summary["sibling_branch_count"] > 1


def _navigation_case_set(overlay_path: Path, overlay: CausalTemporalOverlay) -> dict:
    return {
        "schema_version": "steam-navigation-gold-cases/v0.1",
        "case_set_id": "case-set:test",
        "annotation_status": "draft",
        "annotator": None,
        "locked_sha256": None,
        "cases": [
            {
                "case_id": "door-dependency",
                "overlay_path": str(overlay_path),
                "overlay_id": overlay.overlay_id,
                "question": "Why did the door open?",
                "seed_event_ids": ["event:open"],
                "missing_roles": ["dependency"],
                "graph_read_budget": 2,
                "gold_event_ids": ["event:push", "event:open"],
                "acceptable_first_actions": [
                    {
                        "action_type": "candidate_cause",
                        "source_id": "event:open",
                        "target_ids": ["event:push"],
                        "relation": "enables",
                    }
                ],
                "required_relation_types": ["enables"],
                "tags": ["dependency"],
            }
        ],
    }


def test_case_lock_annotation_packet_and_training_export_enforce_trust_boundary(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    case_set = _navigation_case_set(overlay_path, overlay)
    assert validate_navigation_case_set(case_set) == []
    locked_cases = lock_navigation_case_set(
        case_set,
        annotation_status="human_locked",
        annotator="independent-reviewer",
    )
    assert validate_navigation_case_set(locked_cases) == []
    tampered_cases = json.loads(json.dumps(locked_cases))
    tampered_cases["cases"][0]["question"] = "Tampered question"
    assert any(
        "locked_sha256" in error
        for error in validate_navigation_case_set(tampered_cases)
    )

    backend = FactorGraphBeliefBackend()
    belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    artifact = generate_sibling_artifact(
        belief,
        overlay,
        backend=backend,
        executor=VideoSkillsL2Adapter(use_video_skills_runtime=False),
        world_model=RuleBasedObservationBeliefModel(),
        preference_model=RuleBasedTrajectoryPreferenceModel(),
    )
    packet = build_preference_annotation_packet(
        [("door-dependency", artifact)],
        packet_id="packet:test",
    )
    assert validate_preference_annotation_packet(packet) == []
    serialized = json.dumps(packet)
    assert "rule_based_provisional" not in serialized
    assert "posterior_probabilities" not in serialized
    sampled_packet = build_preference_annotation_packet(
        [("door-dependency", artifact)],
        packet_id="packet:sampled",
        max_comparisons_per_case=2,
    )
    assert len(sampled_packet["comparisons"]) == 2
    for comparison in packet["comparisons"]:
        comparison["label"] = "prefer_left"
        comparison["rationale"] = "Left resolves the required evidence role."
    provisional = lock_annotation_packet(
        packet,
        annotation_status="ai_provisional",
        annotator="GPT-5.6 provisional",
    )
    assert validate_preference_annotation_packet(provisional) == []
    tampered_packet = json.loads(json.dumps(provisional))
    tampered_packet["comparisons"][0]["rationale"] = "Tampered rationale."
    assert any(
        "locked_sha256" in error
        for error in validate_preference_annotation_packet(tampered_packet)
    )
    with pytest.raises(ValueError, match="human_locked"):
        export_training_records(provisional)
    records = export_training_records(provisional, allow_ai_provisional=True)
    assert {row["task"] for row in records} == {
        "observation_belief_transition",
        "trajectory_pairwise_preference",
    }
    lowered = json.dumps(records).lower()
    for forbidden in ('"reward"', '"utility"', '"q_value"', '"score"'):
        assert forbidden not in lowered


def test_executed_transition_dataset_is_real_immutable_and_review_gated(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    overlay.atomic_events[0].embedding_ref = EmbeddingRef(
        path="event_embeddings.npy",
        model="Qwen/Qwen3-VL-Embedding-2B",
        dimension=2048,
        row_index=0,
    )
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    case_set = lock_navigation_case_set(
        _navigation_case_set(overlay_path, overlay),
        annotation_status="human_locked",
        annotator="independent-reviewer",
    )

    dataset = build_executed_transition_dataset(
        case_set,
        case_root=tmp_path,
        dataset_id="transitions:test",
        belief_backend="factor_graph",
        executor_factory=lambda: VideoSkillsL2Adapter(
            use_video_skills_runtime=False
        ),
    )
    repeated = build_executed_transition_dataset(
        case_set,
        case_root=tmp_path,
        dataset_id="transitions:test",
        belief_backend="factor_graph",
        executor_factory=lambda: VideoSkillsL2Adapter(
            use_video_skills_runtime=False
        ),
    )

    assert validate_executed_transition_dataset(dataset) == []
    assert dataset["checkpoints"] == repeated["checkpoints"]
    assert dataset["records"] == repeated["records"]
    assert dataset["annotation_status"] == "unreviewed"
    assert dataset["schema_version"] == "steam-executed-transition-dataset/v0.2"
    assert dataset["formal_eligible"] is False
    assert dataset["records"]
    assert dataset["summary"]["post_read_verifier_count"] == 0
    assert dataset["summary"]["action_origin_counts"]
    assert all(
        row["action_provenance"]["graph_mutated"] is False
        for row in dataset["records"]
    )
    assert len(dataset["checkpoints"]) == 1
    assert len({row["checkpoint_ref"] for row in dataset["records"]}) == 1
    assert all(
        row["target"]["observation_descriptor"]["predicted_only"] is False
        and row["target"]["belief_delta"]["predicted_only"] is False
        for row in dataset["records"]
    )
    assert any(
        node.get("embedding_ref", {}).get("model")
        == "Qwen/Qwen3-VL-Embedding-2B"
        for row in dataset["records"]
        for node in row["local_context"]["nodes"]
        if node.get("embedding_ref") is not None
    )
    gathering = inspect_transition_gathering(dataset, case_set)
    assert gathering["dataset_valid"] is True
    assert gathering["training_ready"] is False
    assert gathering["training_performed"] is False
    assert gathering["unreviewed_target_count"] == len(dataset["records"])
    with pytest.raises(ValueError, match="review_decision"):
        lock_executed_transition_dataset(
            dataset,
            annotation_status="human_locked",
            annotator="transition-reviewer",
        )
    reviewed = json.loads(json.dumps(dataset))
    for row in reviewed["records"]:
        row["review_decision"] = "accept"
        row["review_rationale"] = "Real read and categorical delta verified."
    locked = lock_executed_transition_dataset(
        reviewed,
        annotation_status="human_locked",
        annotator="transition-reviewer",
    )
    assert locked["formal_eligible"] is True
    training = export_executed_transition_training_records(locked)
    assert len(training) == len(locked["records"])
    assert all(row["action_provenance"] for row in training)
    assert all(row["execution_provenance"]["skill_id"] for row in training)
    serialized = json.dumps(training).lower()
    for forbidden in ('"reward"', '"utility"', '"q_value"', '"score"'):
        assert forbidden not in serialized
    tampered = json.loads(json.dumps(locked))
    tampered["records"][0]["target"]["belief_delta"][
        "answerability_after"
    ] = "abstain"
    assert any(
        "locked_sha256" in error
        for error in validate_executed_transition_dataset(tampered)
    )


def test_transition_generation_rejects_unapproved_provisional_case_set(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    provisional = lock_navigation_case_set(
        _navigation_case_set(overlay_path, overlay),
        annotation_status="ai_provisional",
        annotator="GPT-5.6 provisional",
    )

    with pytest.raises(ValueError, match="human_locked"):
        build_executed_transition_dataset(
            provisional,
            case_root=tmp_path,
            dataset_id="transitions:rejected",
        )


def test_reviewed_action_restores_endpoint_read_without_mutating_graph(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    case_set = _navigation_case_set(overlay_path, overlay)
    case = case_set["cases"][0]
    case["missing_roles"] = ["state_transition"]
    case["acceptable_first_actions"] = [
        {
            "action_type": "inspect_state_change",
            "source_id": "event:open",
            "target_ids": ["event:push"],
            "relation": "state_transition",
        }
    ]
    case["required_relation_types"] = ["state_transition"]
    locked = lock_navigation_case_set(
        case_set,
        annotation_status="human_locked",
        annotator="independent-reviewer",
    )

    dataset = build_executed_transition_dataset(
        locked,
        case_root=tmp_path,
        dataset_id="transitions:review-restored",
        executor_factory=lambda: VideoSkillsL2Adapter(
            use_video_skills_runtime=False
        ),
    )
    record = next(
        row
        for row in dataset["records"]
        if row["action"]["action_type"] == "inspect_state_change"
    )

    assert record["action_provenance"]["origin"] == "review_anchored"
    assert record["action_provenance"]["legality"] == "review_restored"
    assert record["action_provenance"]["edge_admitted"] is False
    assert record["action_provenance"]["graph_mutated"] is False
    assert record["execution"]["grounded"] is True
    assert record["execution"]["skill_id"] == "review_anchored_endpoint_read"
    assert record["target"]["belief_delta"]["resolved_roles"] == []


def test_transition_review_packet_is_blinded_categorical_and_duplicate_audited(
    tmp_path: Path,
) -> None:
    from steam_video_new.implicit_world_model.l15_graph_navigator import (
        apply_transition_review,
    )

    overlay = _overlay()
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    locked_cases = lock_navigation_case_set(
        _navigation_case_set(overlay_path, overlay),
        annotation_status="human_locked",
        annotator="independent-reviewer",
    )
    dataset = build_executed_transition_dataset(
        locked_cases,
        case_root=tmp_path,
        dataset_id="transitions:review-packet",
        executor_factory=lambda: VideoSkillsL2Adapter(
            use_video_skills_runtime=False
        ),
    )
    packet, hidden = build_transition_review_packet(
        dataset,
        locked_cases,
        packet_id="transition-review:test",
        native_controls_per_action=1,
        consistency_duplicates=1,
    )

    assert validate_transition_review_packet(packet) == []
    serialized = json.dumps(packet).lower()
    for forbidden in (
        "action_provenance", "legality", "verifier_measurement",
        "belief_update_audit", "target_source", "stored_target",
    ):
        assert forbidden not in serialized
    decisions = []
    for item in packet["items"]:
        refs = item["executed_result"]["evidence_refs"]
        if not refs:
            refs = [item["visible_context"]["nodes"][0]["node_id"]]
        decisions.append(
            {
                "item_id": item["item_id"],
                "review_decision": "accept",
                "observation_validity": "valid",
                "action_execution_validity": "valid",
                "belief_delta_validity": "valid",
                "relation_outcome": "supports",
                "evidence_refs": [refs[0]],
                "rationale": "Visible grounded execution supports the categorical update.",
            }
        )
    review = {
        "schema_version": "steam-transition-review-response/v0.1",
        "packet_id": packet["packet_id"],
        "labels_source": "model_provisional",
        "annotator": "GPT-5.6 provisional",
        "protocol": "outcome_blinded_categorical_review",
        "decisions": decisions,
    }
    locked = apply_transition_review(packet, review)
    report = inspect_transition_review(locked, hidden)

    assert validate_transition_review_packet(locked) == []
    assert locked["annotation_status"] == "ai_provisional"
    assert report["valid"] is True
    assert report["duplicate_consistency"]["all_groups_exact"] is True
    assert report["training_ready"] is False
    assert report["training_performed"] is False

    human_review = json.loads(json.dumps(review))
    human_review["labels_source"] = "independent_human"
    human_review["annotator"] = "independent-reviewer"
    human_review["protocol"] = "outcome_blinded_categorical_human_review"
    human_locked = apply_transition_review(packet, human_review)
    assert human_locked["annotation_status"] == "human_locked"
    assert human_locked["labels_source"] == "independent_human"
    assert validate_transition_review_packet(human_locked) == []

    selected_id = dataset["records"][0]["record_id"]
    targeted, targeted_key = build_transition_review_packet(
        dataset,
        locked_cases,
        packet_id="transition-review:targeted",
        native_controls_per_action=0,
        consistency_duplicates=1,
        selected_record_strata={selected_id: "control"},
    )
    assert len(targeted["items"]) == 2
    assert validate_transition_review_packet(targeted) == []
    assert {
        row["sampling_stratum"] for row in targeted_key["items"]
    } == {"targeted:control", "consistency_duplicate"}

    numeric = json.loads(json.dumps(review))
    numeric["decisions"][0]["reward"] = 1
    with pytest.raises(ValueError, match="invalid transition review decision"):
        apply_transition_review(packet, numeric)


def test_visual_review_bundle_binds_video_without_public_path_or_outcome_leak(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    locked_cases = lock_navigation_case_set(
        _navigation_case_set(overlay_path, overlay),
        annotation_status="human_locked",
        annotator="independent-reviewer",
    )
    dataset = build_executed_transition_dataset(
        locked_cases,
        case_root=tmp_path,
        dataset_id="transitions:visual-review",
        executor_factory=lambda: VideoSkillsL2Adapter(use_video_skills_runtime=False),
    )
    packet, key = build_transition_review_packet(
        dataset,
        locked_cases,
        packet_id="transition-review:visual",
        native_controls_per_action=1,
        consistency_duplicates=1,
    )
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video_path = video_root / "video:preference.mp4"
    video_path.write_bytes(b"synthetic-container-for-binding-test")

    public, private, report = build_visual_review_bundle(
        packet, key, video_root=video_root
    )

    assert validate_visual_review_bundle(packet, public, private) == []
    assert report["fully_covered_item_count"] == len(packet["items"])
    assert report["missing_node_count"] == 0
    assert report["training_performed"] is False
    public_text = json.dumps(public).lower()
    assert str(tmp_path).lower() not in public_text
    for forbidden in (
        "video_path", "overlay_path", "sampling_stratum", "stored_target",
        "verifier_measurement", "reward", "probability",
    ):
        assert forbidden not in public_text
    assert all(
        asset["media_url"].startswith("/api/media/visual:")
        for item in public["items"]
        for asset in item["visual_evidence"]
    )
    assert {row["video_path"] for row in private["assets"]} == {str(video_path)}


def test_visual_review_bundle_reports_truncated_and_out_of_range_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from steam_video_new.implicit_world_model.l15_graph_navigator import visual_review

    overlay = _overlay()
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    locked_cases = lock_navigation_case_set(
        _navigation_case_set(overlay_path, overlay),
        annotation_status="human_locked",
        annotator="independent-reviewer",
    )
    dataset = build_executed_transition_dataset(
        locked_cases,
        case_root=tmp_path,
        dataset_id="transitions:duration-check",
        executor_factory=lambda: VideoSkillsL2Adapter(use_video_skills_runtime=False),
    )
    packet, key = build_transition_review_packet(
        dataset, locked_cases, packet_id="transition-review:duration-check",
        native_controls_per_action=1, consistency_duplicates=1,
    )
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "video:preference.mp4").write_bytes(b"container")
    monkeypatch.setattr(visual_review, "_probe_video_duration", lambda _: 1.5)

    public, private, report = build_visual_review_bundle(packet, key, video_root=video_root)

    availabilities = {
        asset["availability"]
        for item in public["items"]
        for asset in item["visual_evidence"]
    }
    assert "partial" in availabilities or "missing" in availabilities
    assert report["fully_covered_item_count"] < len(packet["items"])
    assert report["partially_available_node_count"] + report["missing_node_count"] > 0
    assert validate_visual_review_bundle(packet, public, private) == []


def test_human_review_media_range_parser() -> None:
    from steam_video_new.implicit_world_model.l15_graph_navigator.human_review_server import (
        _parse_range_header,
    )

    assert _parse_range_header(None, 100) is None
    assert _parse_range_header("bytes=0-9", 100) == (0, 9)
    assert _parse_range_header("bytes=90-", 100) == (90, 99)
    assert _parse_range_header("bytes=-10", 100) == (90, 99)
    assert _parse_range_header("bytes=95-200", 100) == (95, 99)
    with pytest.raises(ValueError):
        _parse_range_header("bytes=100-101", 100)
    with pytest.raises(ValueError):
        _parse_range_header("bytes=0-1,4-5", 100)


def test_realized_delta_recomputes_full_categorical_state_change() -> None:
    overlay = _overlay()
    backend = FactorizedBeliefBackend()
    before = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    action = next(
        value
        for value in guided_navigation_actions(before, overlay)
        if value.action_type is NavigationActionType.CANDIDATE_CAUSE
    )
    observed = next(
        node for node in overlay.atomic_events if node.node_id == "event:push"
    )
    after = backend.update(before, action, [observed], overlay).belief

    delta = derive_realized_belief_delta(before, after)

    assert delta.predicted_only is False
    assert delta.resolved_roles == ("dependency",)
    assert delta.hypothesis_updates[0].edge_id == "edge:push-open"
    assert delta.hypothesis_updates[0].disposition.value == "accepted"
    assert delta.frontier_change.value == "opened"
    assert delta.recovery_status.value == "recovered"


def test_balanced_miner_reports_deficits_without_cross_category_backfill(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    overlay.relations[0].provenance["hard_verifier"]["state_transition"] = {
        "passed": False,
        "reasons": ["visible state delta is not established"],
    }
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    quotas = {category: 0 for category in BALANCED_CASE_CATEGORIES}
    quotas.update({"state_reject": 1, "identity_reject": 1, "counterevidence_empty": 1})

    cases, report, queue = mine_balanced_reasoning_cases(
        [overlay_path],
        case_set_id="balanced:test",
        quotas=quotas,
    )

    assert report["cross_category_backfill"] is False
    assert report["selected_categories"]["state_reject"] == 1
    assert report["selected_categories"]["counterevidence_empty"] == 1
    assert report["quota_deficits"]["identity_reject"] == 1
    assert len(cases["cases"]) == 2
    assert validate_balanced_review_queue(queue) == []
    assert "quota_deficits" not in queue
    with pytest.raises(ValueError, match="incomplete"):
        lock_balanced_review_queue(
            queue,
            annotation_status="human_locked",
            annotator="reviewer",
        )

    reviewed = json.loads(json.dumps(queue))
    for row in reviewed["reviews"]:
        is_offline_reject = row["category"] == "state"
        row["review_decision"] = "reject" if is_offline_reject else "accept"
        row["verifier_outcome"] = "rejects" if is_offline_reject else "not_applicable"
        row["evidence_chain_valid"] = not is_offline_reject
        row["first_action_valid"] = not is_offline_reject
        row["delayed_effect"] = "not_applicable"
        row["rationale"] = "Visible evidence and online action legality reviewed."
    locked = lock_balanced_review_queue(
        reviewed,
        annotation_status="human_locked",
        annotator="reviewer",
    )
    exported = export_reviewed_balanced_case_set(
        locked,
        case_set_id="balanced:accepted",
    )
    assert exported["annotation_status"] == "human_locked"
    assert len(exported["cases"]) == 1
    serialized_reviews = json.dumps(queue["reviews"])
    assert "state_reject" not in serialized_reviews
    assert "counterevidence_empty" not in serialized_reviews


def test_balanced_evidence_packet_is_blinded_grounded_and_importable(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    overlay.relations[0].provenance["hard_verifier"]["state_transition"] = {
        "passed": False,
        "reasons": ["visible state delta is not established"],
    }
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    quotas = {category: 0 for category in BALANCED_CASE_CATEGORIES}
    quotas.update({"state_reject": 1, "counterevidence_empty": 1})
    _, report, queue = mine_balanced_reasoning_cases(
        [overlay_path], case_set_id="balanced:evidence", quotas=quotas
    )

    packet, key = build_balanced_evidence_packet(
        queue, packet_id="evidence:test", context_limit=2
    )
    assert validate_balanced_evidence_packet(packet) == []
    assert inspect_balanced_evidence_packet(packet)["training_performed"] is False
    serialized_items = json.dumps(packet["items"]).lower()
    for forbidden in (
        "hard_verifier", "state_reject", "counterevidence_empty",
        "relation_probabilities", "direction_confidence",
    ):
        assert forbidden not in serialized_items
    assert key["items"]
    assert key["items"][0]["overlay_path"] == str(overlay_path.resolve())

    annotated = json.loads(json.dumps(packet))
    for item in annotated["items"]:
        reference = item["evidence"]["proposed_endpoints"][0]["node_id"]
        is_state = item["stratum"] == "state"
        item["annotation"] = {
            "review_decision": "reject" if is_state else "accept",
            "verifier_outcome": "rejects" if is_state else "not_applicable",
            "evidence_chain_valid": not is_state,
            "first_action_valid": not is_state,
            "delayed_effect": "not_applicable",
            "evidence_refs": [reference],
            "rationale": "Visible evidence and proposed action were reviewed.",
        }
    locked_packet = lock_balanced_evidence_packet(
        annotated,
        annotation_status="ai_provisional",
        annotator="gpt-5.6-test",
    )
    external_review = {
        "schema_version": "steam-balanced-evidence-review/v0.1",
        "packet_id": packet["packet_id"],
        "labels_source": "model_provisional",
        "annotator": "gpt-5.6-test",
        "protocol": "categorical-only",
        "decisions": [
            {"item_id": item["item_id"], **item["annotation"]}
            for item in annotated["items"]
        ],
    }
    leaked_review = json.loads(json.dumps(external_review))
    leaked_review["decisions"][0]["score"] = 0.5
    with pytest.raises(ValueError, match="unexpected annotation fields"):
        apply_evidence_review(packet, leaked_review)
    assert apply_evidence_review(packet, external_review) == locked_packet
    locked_queue, inspection = import_evidence_annotations(queue, locked_packet)
    assert locked_queue["annotation_status"] == "ai_provisional"
    assert inspection["formal_gate_eligible"] is False
    assert inspection["training_performed"] is False
    assert report["cross_category_backfill"] is False


def test_blinded_packet_randomizes_sides_without_reading_labels() -> None:
    overlay = _overlay()
    backend = FactorGraphBeliefBackend()
    belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    artifact = generate_sibling_artifact(
        belief,
        overlay,
        backend=backend,
        executor=VideoSkillsL2Adapter(use_video_skills_runtime=False),
        world_model=RuleBasedObservationBeliefModel(),
        preference_model=RuleBasedTrajectoryPreferenceModel(),
    )
    first = build_preference_annotation_packet(
        [("door-dependency", artifact)],
        packet_id="packet:a",
        max_comparisons_per_case=1,
    )
    repeated = build_preference_annotation_packet(
        [("door-dependency", artifact)],
        packet_id="packet:a",
        max_comparisons_per_case=1,
    )
    reversed_packet = build_preference_annotation_packet(
        [("door-dependency", artifact)],
        packet_id="packet:d",
        max_comparisons_per_case=1,
    )

    assert first["comparisons"] == repeated["comparisons"]
    assert first["comparisons"][0]["left"] == reversed_packet["comparisons"][0]["right"]
    assert "randomized" in " ".join(first["instructions"])


def test_ai_reviews_lock_cases_and_preferences_without_numeric_reward(tmp_path: Path) -> None:
    overlay = _overlay()
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    draft = _navigation_case_set(overlay_path, overlay)
    case_id = draft["cases"][0]["case_id"]
    locked_cases, case_report = apply_provisional_case_review(
        draft,
        {
            "labels_source": "model_provisional",
            "annotator": "gpt-test",
            "decisions": [
                {
                    "case_id": case_id,
                    "judgment": "supported",
                    "evidence_refs": ["event:push", "event:open"],
                    "rationale": "Both grounded events are required.",
                }
            ],
        },
    )
    assert locked_cases["annotation_status"] == "ai_provisional"
    assert case_report["formal_gate_eligible"] is False

    backend = FactorGraphBeliefBackend()
    belief = backend.initialize(
        "Why did the door open?",
        overlay,
        seed_evidence=("event:open",),
        missing_roles=("dependency",),
    )
    artifact = generate_sibling_artifact(
        belief,
        overlay,
        backend=backend,
        executor=VideoSkillsL2Adapter(use_video_skills_runtime=False),
        world_model=RuleBasedObservationBeliefModel(),
        preference_model=RuleBasedTrajectoryPreferenceModel(),
    )
    packet = build_preference_annotation_packet(
        [(case_id, artifact)], packet_id="packet:review", max_comparisons_per_case=1
    )
    comparison_id = packet["comparisons"][0]["comparison_id"]
    locked_packet, preference_report = apply_preference_review(
        packet,
        {
            "labels_source": "model_provisional",
            "annotator": "gpt-test",
            "decisions": [
                {
                    "comparison_id": comparison_id,
                    "label": "prefer_left",
                    "rationale": "Left gives more grounded progress.",
                }
            ],
        },
    )
    serialized = json.dumps(locked_packet).lower()
    assert preference_report["numeric_reward_present"] is False
    for forbidden in ('"reward"', '"utility"', '"q_value"', '"score"'):
        assert forbidden not in serialized


def test_categorical_confusion_keeps_inconclusive_as_a_third_class() -> None:
    packet = {
        "labels_source": "model_provisional",
        "items": [
            {
                "item_id": "one",
                "relation": "same_entity",
                "annotation": {"judgment": "supported"},
            },
            {
                "item_id": "two",
                "relation": "same_entity",
                "annotation": {"judgment": "contradicted"},
            },
            {
                "item_id": "three",
                "relation": "state_transition",
                "annotation": {"judgment": "unclear"},
            },
        ],
    }
    report = categorical_confusion_report(
        packet, admitted_candidate_predictions(packet)
    )

    matrix = report["groups"]["all"]["reference_by_prediction"]
    assert matrix["supports"]["supports"] == 1
    assert matrix["rejects"]["supports"] == 1
    assert matrix["inconclusive"]["supports"] == 1
    assert report["formal_gate_eligible"] is False


def test_matched_ablation_runs_all_policies_from_locked_checkpoint(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    locked = lock_navigation_case_set(
        _navigation_case_set(overlay_path, overlay),
        annotation_status="human_locked",
        annotator="independent-reviewer",
    )

    report = evaluate_matched_navigation(locked, case_root=tmp_path)

    assert tuple(report["strategies"]) == MATCHED_STRATEGIES
    assert report["case_set_annotation_status"] == "human_locked"
    assert report["strategies"]["factor_graph_direct_preference"][
        "first_action_accuracy"
    ] == 1.0
    assert report["strategies"]["factor_graph_rule_world_model_lookahead"][
        "answer_accuracy"
    ] == 1.0
    assert "factor_graph_frozen_posterior_lookahead" in report["strategies"]
    assert "factor_graph_shuffled_relations" in report["strategies"]
    assert report["diagnostic_gates"]["engineering_status"] == "descriptive_only"
    assert report["diagnostic_gates"]["formal_result"] is False
    assert "comparisons" in report["diagnostic_gates"]


def test_case_miner_produces_video_disjoint_draft_and_reports_missing_strata(
    tmp_path: Path,
) -> None:
    paths = []
    for index in range(3):
        overlay = _overlay()
        overlay.video_id = f"video:{index}"
        overlay.overlay_id = f"overlay:mine:{index}"
        overlay.example_id = f"example:mine:{index}"
        overlay.relations.append(
            RelationBelief(
                edge_id=f"temporal:{index}",
                src="event:push",
                dst="event:open",
                relation_probabilities={"temporal_next": 1.0},
                status=RelationStatus.DETERMINISTIC,
                direction_confidence=1.0,
            )
        )
        path = tmp_path / f"overlay-{index}.json"
        path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
        paths.append(path)

    cases, report = mine_navigation_cases(
        paths,
        case_set_id="mined:test",
        desired_count=6,
        per_video_limit=2,
        quotas={
            "temporal": 3,
            "identity": 0,
            "state_transition": 1,
            "verified_dependency": 2,
            "delayed_bridge": 0,
            "counterevidence": 0,
        },
    )

    assert validate_navigation_case_set(cases) == []
    assert cases["annotation_status"] == "draft"
    assert len(cases["cases"]) == 6
    video_splits: dict[str, set[str]] = {}
    for case in cases["cases"]:
        video_splits.setdefault(case["overlay_id"], set()).add(case["split"])
    assert all(len(splits) == 1 for splits in video_splits.values())
    assert report["formal_ready"] is False
    assert report["quota_deficits"]["state_transition"] == 1


def test_lightweight_models_emit_categories_and_ordinal_labels_only(
    tmp_path: Path,
) -> None:
    records = []
    for index, (video_id, label) in enumerate(
        (("video:a", "prefer_left"), ("video:b", "prefer_right"), ("video:c", "prefer_left"), ("video:d", "prefer_right"))
    ):
        checkpoint = {
            "missing_roles": ["dependency"],
            "answerability": "not_ready",
        }
        action = {"action_type": "candidate_cause", "target_ids": [f"event:{index}"]}
        records.append(
            {
                "task": "observation_belief_transition",
                "case_id": f"case:{index}",
                "video_id": video_id,
                "question": "Why did this happen?",
                "checkpoint": checkpoint,
                "action": action,
                "target": {
                    "observation_descriptor": {"role": "dependency"},
                    "belief_delta": {
                        "resolved_roles": ["dependency"],
                        "uncertainty_change": "decrease",
                        "answerability_after": "ready",
                        "contradiction_updates": [],
                    },
                },
                "label_source": "human_locked",
            }
        )
        records.append(
            {
                "task": "trajectory_pairwise_preference",
                "comparison_id": f"comparison:{index}",
                "case_id": f"case:{index}",
                "video_id": video_id,
                "question": "Why did this happen?",
                "checkpoint": checkpoint,
                "left": {"action": action},
                "right": {"action": {"action_type": "semantic"}},
                "target": {"label": label},
                "rationale": "Categorical preference.",
                "label_source": "human_locked",
            }
        )

    provisional_records = json.loads(json.dumps(records))
    for row in provisional_records:
        row["label_source"] = "ai_provisional"
    with pytest.raises(ValueError, match="not human_locked"):
        train_baselines(provisional_records, output_dir=tmp_path / "rejected")

    report = train_baselines(records, output_dir=tmp_path / "models")
    import joblib

    transition = joblib.load(tmp_path / "models" / "transition_model.joblib")
    preference = joblib.load(tmp_path / "models" / "preference_model.joblib")
    transition_row = next(row for row in records if row["task"].startswith("observation"))
    preference_row = next(row for row in records if row["task"].startswith("trajectory"))
    assert predict_transition_labels(transition, transition_row)["observation_role"] == "dependency"
    assert predict_preference_label(preference, preference_row) in {
        "prefer_left",
        "prefer_right",
        "tie",
        "incomparable",
    }
    assert report["reward_model"] is False
