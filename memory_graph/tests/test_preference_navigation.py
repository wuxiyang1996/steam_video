from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

from memory_graph.navigation import NavigationActionType, propose_navigation_actions
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
    PreferenceOnlyPlanner,
    RelationGrounding,
    RuleBasedObservationBeliefModel,
    RuleBasedTrajectoryPreferenceModel,
    VideoSkillsL2Adapter,
    build_video_skills_l2_rollout,
    generate_sibling_artifact,
    load_overlay_artifact,
    validate_sibling_artifact,
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
    assert summary["sibling_branch_count"] > 1
