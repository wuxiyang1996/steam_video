from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

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
    ClosedLoopNavigator,
    FactorizedBeliefBackend,
    FactorGraphBeliefBackend,
    PreferenceOnlyPlanner,
    RelationGrounding,
    RuleBasedObservationBeliefModel,
    RuleBasedTrajectoryPreferenceModel,
    VideoSkillsL2Adapter,
    build_video_skills_l2_rollout,
    generate_sibling_artifact,
    guided_navigation_actions,
    load_overlay_artifact,
    validate_sibling_artifact,
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
    PlanDecision,
    PredictedTransition,
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
    assert actions[0].action_type is NavigationActionType.CANDIDATE_CAUSE


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


def test_factor_guidance_uses_explicit_temporal_direction_as_tie_breaker() -> None:
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

    assert guided_navigation_actions(after, overlay)[0].action_type is (
        NavigationActionType.TEMPORAL_FORWARD
    )
    assert guided_navigation_actions(before, overlay)[0].action_type is (
        NavigationActionType.TEMPORAL_BACK
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
