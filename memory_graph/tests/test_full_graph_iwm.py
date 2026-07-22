from __future__ import annotations

from dataclasses import replace
import json

from memory_graph.adaptive_windowing import (
    SurpriseWindowConfig,
    SurpriseWindowWriter,
    VisualFeatureSample,
)
from memory_graph.consolidation import materialize_bounded_memory
from memory_graph.correlation_overlay import (
    CategoricalCorrelationJudgment,
    CorrelationEdge,
    CorrelationPair,
    CorrelationStatus,
    CorrelationType,
    build_categorical_correlation_overlay,
)
from memory_graph.selectstream_policy import plan_bounded_memory
from memory_graph.soft_correlation import SoftNavigationCorrelation
from memory_graph.types import (
    CausalTemporalOverlay,
    EmbeddingRef,
    MemoryNode,
    RelationBelief,
    RelationStatus,
    TimeSpan,
)
from steam_video_new.implicit_world_model.full_graph_iwm import (
    ActionKind,
    AnswerabilityState,
    CategoricalBeliefDelta,
    CursorBeliefState,
    EvidenceOutcome,
    FullGraphIWMPlanner,
    GraphActionCompiler,
    GPTOSSCategoricalCorrelationEvaluator,
    GPTOSSFullGraphPreferenceModel,
    GPTOSSFullGraphWorldModel,
    GPTOSSRealEvidenceBeliefUpdater,
    GPTOSSReactiveGraphPlanner,
    IWMRequest,
    ImaginedTransition,
    PreferenceLabel,
    RetainedEvidenceGraph,
    TemporalNavigationEdge,
    TrajectoryPrediction,
    ClueInterval,
    FrozenWorldModel,
    ShuffledWorldModel,
    execute_graph_action,
    graph_fingerprint,
    run_oracle_clue_ceiling,
    run_real_read_closed_loop,
    build_l1_l15_navigation_graph,
    compile_l1_l15_navigation_graph,
)
from steam_video_new.implicit_world_model.full_graph_iwm.cgbench_pilot import (
    compile_cgbench_gate,
)
from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
    FrontierChange,
    PredictedObservation,
    ProgressChange,
    TrajectoryPair,
    TrajectoryPreference,
)
from steam_video_new.implicit_world_model.full_graph_iwm.model_input import (
    build_iwm_graph_input,
    graph_input_to_categorical_payload,
)
from steam_video_new.implicit_world_model.full_graph_iwm.survivor_preference_data import (
    build_grounded_survivor_preference_packet,
)
from steam_video_new.implicit_world_model.full_graph_iwm.survivor_preference_eval import (
    evaluate_blinded_preferences,
)


def _node(
    node_id: str,
    start: float,
    text: str,
    *,
    hidden: bool = False,
    source_segments: list[str] | None = None,
) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        video_id="video:test",
        time_span=TimeSpan(start, start + 1.0),
        provenance={"producer": "test"},
        node_type="observation",
        text=text,
        source_segments=source_segments or [],
        embedding_ref=EmbeddingRef(
            path="embeddings.npy",
            dimension=4,
            row_index=int(start),
        ),
        metadata={"hidden_supervision": hidden, "modality": "visual"},
    )


def _temporal(left: MemoryNode, right: MemoryNode) -> RelationBelief:
    return RelationBelief(
        edge_id=f"temporal:{left.node_id}:{right.node_id}",
        src=left.node_id,
        dst=right.node_id,
        relation_probabilities={"temporal_next": 1.0, "before": 1.0},
        status=RelationStatus.DETERMINISTIC,
        direction_confidence=1.0,
        evidence_refs=[left.node_id, right.node_id],
        provenance={"producer": "test"},
    )


def test_surprise_writer_expands_stable_content_and_splits_on_change() -> None:
    samples = [
        VisualFeatureSample(index * 0.5, (1.0, 0.0) if index < 6 else (0.0, 1.0))
        for index in range(13)
    ]
    windows = SurpriseWindowWriter(
        SurpriseWindowConfig(
            sample_period_s=0.5,
            min_window_s=1.0,
            max_window_s=4.0,
            calibration_history=3,
            high_surprise_quantile=0.75,
        )
    ).segment(samples, duration_s=6.0)

    assert windows[0].boundary_reason == "representation_surprise"
    assert windows[0].end_s == 3.0
    assert windows[-1].end_s == 6.0
    assert all(window.end_s - window.start_s <= 4.0 for window in windows)


def test_consolidation_materializes_merge_and_rebuilds_temporal_chain() -> None:
    first = _node("l1:first", 0.0, "stable view one", source_segments=["clip:1"])
    second = _node("l1:second", 1.0, "stable view two", source_segments=["clip:1"])
    third = _node("l1:third", 2.0, "new event", source_segments=["clip:2"])
    relations = [_temporal(first, second), _temporal(second, third)]
    decision = plan_bounded_memory(
        [first, second, third],
        relations,
        capacity=2,
        redundancy={first.node_id: 0.95, second.node_id: 0.95},
    )
    result = materialize_bounded_memory(
        [first, second, third],
        relations,
        decision,
    )

    assert len(result.nodes) == 2
    assert result.id_map[first.node_id] == result.id_map[second.node_id]
    merged = next(
        node for node in result.nodes if node.node_id.startswith("memory:consolidated:")
    )
    assert merged.embedding_ref is None
    assert merged.metadata["consolidation"]["embedding_status"] == "refresh_required"
    assert len(result.relations) == 1
    assert set(result.relations[0].relation_probabilities) == {
        "before",
        "temporal_next",
    }


def test_consolidation_preserves_non_temporal_part_of_a_mixed_relation() -> None:
    first = _node("l1:first", 0.0, "first")
    second = _node("l1:second", 1.0, "second")
    mixed = RelationBelief(
        edge_id="relation:mixed",
        src=first.node_id,
        dst=second.node_id,
        relation_probabilities={"before": 1.0, "transition_support": 0.7},
        status=RelationStatus.UNCALIBRATED_PRIOR,
        direction_confidence=0.8,
        evidence_refs=[first.node_id, second.node_id],
        provenance={"producer": "test"},
    )
    decision = plan_bounded_memory([first, second], [mixed], capacity=2)
    result = materialize_bounded_memory([first, second], [mixed], decision)

    assert any(
        set(edge.relation_probabilities) == {"transition_support"}
        for edge in result.relations
    )
    assert any(
        "temporal_next" in edge.relation_probabilities for edge in result.relations
    )


def test_pairwise_embedding_redundancy_coalesces_an_adjacent_semantic_run() -> None:
    nodes = [
        _node(
            f"l1:{index}",
            float(index),
            "same continuing action",
            source_segments=[f"clip:{index}"],
        )
        for index in range(3)
    ]
    decision = plan_bounded_memory(
        nodes,
        [_temporal(nodes[0], nodes[1]), _temporal(nodes[1], nodes[2])],
        capacity=3,
        pairwise_redundancy={
            (nodes[0].node_id, nodes[1].node_id): 1.0,
            (nodes[1].node_id, nodes[2].node_id): 1.0,
        },
        merge_redundancy_threshold=1.0 - 1e-6,
    )
    result = materialize_bounded_memory(nodes, [], decision)

    assert decision.merge_groups == (("l1:0", "l1:1", "l1:2"),)
    assert len(result.nodes) == 1
    assert next(iter(result.merged_lineage.values())) == (
        "l1:0",
        "l1:1",
        "l1:2",
    )


class _RecordingEvaluator:
    evaluator_name = "recording-categorical-evaluator"

    def __init__(self) -> None:
        self.pairs: tuple[CorrelationPair, ...] = ()

    def evaluate(self, pairs: tuple[CorrelationPair, ...]):
        self.pairs = tuple(pairs)
        first = self.pairs[0]
        return (
            CategoricalCorrelationJudgment(
                src=first.src.node_id,
                dst=first.dst.node_id,
                relation=CorrelationType.SEMANTIC_RECURRENCE,
                status=CorrelationStatus.CANDIDATE,
                evidence_refs=(first.src.node_id, first.dst.node_id),
                candidate_sources=(self.evaluator_name,),
            ),
        )


def test_correlation_builder_evaluates_every_retained_pair_without_top_k() -> None:
    nodes = [
        _node(f"l1:{index}", float(index), f"evidence {index}") for index in range(3)
    ]
    evaluator = _RecordingEvaluator()
    overlay = build_categorical_correlation_overlay(nodes, evaluator=evaluator)

    assert len(evaluator.pairs) == 3
    assert overlay.evaluated_pair_count == 3
    assert (
        overlay.to_dict()["build_audit"]["pair_generation"]
        == "all_retained_node_pairs_no_top_k"
    )
    assert overlay.edges[0].status is CorrelationStatus.CANDIDATE


class _RejectingEvaluator:
    evaluator_name = "rejecting-categorical-evaluator"

    def evaluate(self, pairs):
        pair = pairs[0]
        return (
            CategoricalCorrelationJudgment(
                src=pair.src.node_id,
                dst=pair.dst.node_id,
                relation=CorrelationType.SEMANTIC_RECURRENCE,
                status=CorrelationStatus.REJECTED,
                evidence_refs=(pair.src.node_id, pair.dst.node_id),
                candidate_sources=(self.evaluator_name,),
            ),
        )


def test_categorical_rejection_overrides_an_unverified_seed_candidate() -> None:
    nodes = [_node("l1:first", 0.0, "first"), _node("l1:second", 1.0, "second")]
    seed = CorrelationEdge(
        edge_id="correlation:seed",
        src=nodes[0].node_id,
        dst=nodes[1].node_id,
        relation=CorrelationType.SEMANTIC_RECURRENCE,
        status=CorrelationStatus.CANDIDATE,
        evidence_refs=(nodes[0].node_id, nodes[1].node_id),
        candidate_sources=("seed",),
    )
    overlay = build_categorical_correlation_overlay(
        nodes,
        evaluator=_RejectingEvaluator(),
        seed_edges=(seed,),
    )
    assert overlay.edges[0].status is CorrelationStatus.REJECTED


def test_legacy_overlay_adapter_separates_navigation_from_strict_relations() -> None:
    first = _node("l1:first", 0.0, "first evidence")
    second = _node("l1:second", 1.0, "second evidence")
    first_event = replace(
        _node("event:first", 0.0, "first event", source_segments=[first.node_id]),
        node_type="atomic_event",
    )
    second_event = replace(
        _node("event:second", 1.0, "second event", source_segments=[second.node_id]),
        node_type="atomic_event",
    )
    dependency = RelationBelief(
        edge_id="relation:event-dependency",
        src=first_event.node_id,
        dst=second_event.node_id,
        relation_probabilities={"transition_support": 0.7},
        status=RelationStatus.UNCALIBRATED_PRIOR,
        direction_confidence=0.6,
        evidence_refs=[first.node_id, second.node_id],
        provenance={"producer": "legacy-test"},
    )
    legacy = CausalTemporalOverlay(
        overlay_id="legacy:test",
        example_id="example:test",
        video_id="video:test",
        l1_observations=[first, second],
        atomic_events=[first_event, second_event],
        relations=[dependency],
        l1_structural_relations=[_temporal(first, second)],
        metadata={"observation_horizon_s": 42.0},
    )

    graph = build_l1_l15_navigation_graph(
        legacy,
        capacity=2,
        embedding_vectors={
            first.node_id: (1.0, 0.0),
            second.node_id: (0.0, 1.0),
        },
    )

    assert len(graph.nodes) == 2
    assert (
        graph.metadata["navigation_contract"]
        == "single_cursor_temporal_plus_soft_correlation_no_top_k"
    )
    assert graph.metadata["observation_end_s"] == 42.0
    assert graph.correlation_edges == ()
    assert graph.verified_relations == ()
    assert (
        graph.metadata["l1_contract"]
        == "grounded_semantic_nodes_plus_deterministic_temporal_backbone"
    )
    assert (
        graph.metadata["l1_5_contract"]
        == "soft_semantic_navigation_plus_non_admitting_grounded_pair_descriptors"
    )
    serialized = graph.to_dict()
    assert serialized["schema_version"] == "steam-l1-l1.5-navigation-graph/v0.1"
    assert len(serialized["nodes"]) == 2
    assert serialized["verified_relations"] == []

    compiled = compile_l1_l15_navigation_graph(
        legacy,
        capacity=2,
        embedding_vectors={
            first.node_id: (1.0, 0.0),
            second.node_id: (0.0, 1.0),
        },
    )
    assert compiled.source_l1_fingerprint == graph.metadata["source_l1_fingerprint"]
    assert compiled.graph.metadata["correlation_builder_mutates_source_l1"] is False
    assert compiled.correlation_pair_audit["contains_question_or_answer"] is False
    assert compiled.correlation_pair_audit["source_l1_fingerprint"] == (
        compiled.source_l1_fingerprint
    )
    assert compiled.multichannel_pair_audit["contains_question_or_answer"] is False
    assert compiled.multichannel_pair_audit["grounded_descriptors_admit_edges"] is False
    assert compiled.multichannel_pair_audit["learned_edge_selector_present"] is False


def _retained_graph(*, include_hidden: bool = False) -> RetainedEvidenceGraph:
    first = _node("l1:bridge", 0.0, "bridge evidence")
    second = _node("l1:answer", 1.0, "answer evidence")
    nodes = [first, second]
    if include_hidden:
        nodes.append(_node("l1:hidden", 2.0, "hidden answer", hidden=True))
    correlation = SoftNavigationCorrelation(
        edge_id="corr:bridge-answer",
        src=first.node_id,
        dst=second.node_id,
        semantic_similarity=0.8,
        src_to_dst_affinity=1.0,
        dst_to_src_affinity=1.0,
        score_source="test-embedding",
        evidence_refs=(first.node_id, second.node_id),
        provenance={"producer": "test"},
    )
    return RetainedEvidenceGraph(
        graph_id="retained:test",
        nodes=tuple(nodes),
        temporal_edges=(
            TemporalNavigationEdge(
                edge_id="time:bridge-answer",
                src=first.node_id,
                dst=second.node_id,
                relation="temporal_next",
            ),
        ),
        correlation_edges=(correlation,),
        capacity=3,
    )


def test_action_compiler_uses_one_cursor_and_all_visible_nodes() -> None:
    graph = _retained_graph(include_hidden=True)
    compiler = GraphActionCompiler()
    initial = CursorBeliefState("belief:0", "question")
    initial_actions = compiler.compile(initial, graph)
    starts = {
        action.target_id
        for action in initial_actions
        if action.kind is ActionKind.START_AT
    }
    assert starts == {"l1:bridge", "l1:answer"}
    assert all(action.target_id != "l1:hidden" for action in initial_actions)

    start = next(
        action for action in initial_actions if action.target_id == "l1:bridge"
    )
    execution = execute_graph_action(start, initial, graph)
    assert execution.observation is not None
    actions = compiler.compile(execution.updated_belief, graph)
    assert any(
        action.kind is ActionKind.FOLLOW_CORRELATION
        and action.source_id == "l1:bridge"
        and action.target_id == "l1:answer"
        for action in actions
    )
    changed_question = replace(execution.updated_belief, question="unrelated words")
    assert {action.action_id for action in actions} == {
        action.action_id for action in compiler.compile(changed_question, graph)
    }

    one_way = replace(
        graph,
        correlation_edges=(
            replace(graph.correlation_edges[0], dst_to_src_affinity=0.0),
        ),
    )
    start_at_answer = next(
        action
        for action in compiler.compile(initial, one_way)
        if action.target_id == "l1:answer"
    )
    from_answer = compiler.compile(
        execute_graph_action(start_at_answer, initial, one_way).updated_belief,
        one_way,
    )
    assert all(
        action.kind is not ActionKind.FOLLOW_CORRELATION for action in from_answer
    )

    model_input = build_iwm_graph_input(execution.updated_belief, graph, actions)
    assert {view.key.node_id for view in model_input.nodes} == {
        "l1:bridge",
        "l1:answer",
    }
    unread = next(view for view in model_input.nodes if view.key.node_id == "l1:answer")
    assert unread.evidence_value is None
    assert unread.key.embedding_ref is not None
    assert "path" not in unread.key.embedding_ref
    payload = graph_input_to_categorical_payload(model_input)
    assert all("provenance" not in edge for edge in payload["correlation_edges"])


class _DelayedWorldModel:
    model_name = "scripted-delayed-world-model"

    def __init__(self) -> None:
        self.batches = []

    def predict_batch(self, requests):
        self.batches.append(tuple(requests))
        predictions = []
        for request in requests:
            delayed_success = (
                request.action.target_id == "l1:answer"
                and request.imagined_history
                and request.imagined_history[0].action.target_id == "l1:bridge"
            )
            predictions.append(
                ImaginedTransition(
                    action=request.action,
                    observation=PredictedObservation(
                        target_id=request.action.target_id,
                        outcome=(
                            EvidenceOutcome.SUPPORT
                            if delayed_success
                            else EvidenceOutcome.INCONCLUSIVE
                        ),
                    ),
                    belief_delta=CategoricalBeliefDelta(
                        progress=(
                            ProgressChange.ADVANCED
                            if delayed_success
                            else ProgressChange.UNCHANGED
                        ),
                        answerability_after=(
                            AnswerabilityState.READY
                            if delayed_success
                            else AnswerabilityState.NOT_READY
                        ),
                        frontier_change=FrontierChange.OPENED,
                    ),
                )
            )
        return tuple(predictions)


class _AnswerabilityPreference:
    model_name = "scripted-answerability-preference"

    def compare_batch(self, pairs, belief):
        del belief
        comparisons = []
        for pair in pairs:
            left_ready = (
                pair.left.transitions[-1].belief_delta.answerability_after
                is AnswerabilityState.READY
            )
            right_ready = (
                pair.right.transitions[-1].belief_delta.answerability_after
                is AnswerabilityState.READY
            )
            label = (
                PreferenceLabel.PREFER_LEFT
                if left_ready and not right_ready
                else (
                    PreferenceLabel.PREFER_RIGHT
                    if right_ready and not left_ready
                    else PreferenceLabel.TIE
                )
            )
            comparisons.append(
                TrajectoryPreference(
                    pair.left.trajectory_id,
                    pair.right.trajectory_id,
                    label,
                )
            )
        return tuple(comparisons)


def test_planner_selects_delayed_first_hop_from_world_model_predictions() -> None:
    graph = _retained_graph()
    world_model = _DelayedWorldModel()
    planner = FullGraphIWMPlanner(
        world_model,
        _AnswerabilityPreference(),
        horizon=2,
    )
    decision = planner.plan(CursorBeliefState("belief:0", "question"), graph)

    assert decision.selected_action.kind is ActionKind.START_AT
    assert decision.selected_action.target_id == "l1:bridge"
    assert decision.top_k_applied is False
    assert len(world_model.batches) == 2
    imagined_second_batch = world_model.batches[1]
    target_view = next(
        view
        for request in imagined_second_batch
        if request.imagined_history
        and request.imagined_history[0].action.target_id == "l1:bridge"
        for view in request.graph_input.nodes
        if view.key.node_id == "l1:bridge"
    )
    assert target_view.acquired is True
    assert target_view.evidence_value is None


class _TiePreference:
    model_name = "tie-preference"

    def compare_batch(self, pairs, belief):
        del belief
        return tuple(
            TrajectoryPreference(
                pair.left.trajectory_id,
                pair.right.trajectory_id,
                PreferenceLabel.TIE,
            )
            for pair in pairs
        )


def test_planner_abstains_instead_of_using_order_as_tie_break() -> None:
    decision = FullGraphIWMPlanner(
        _DelayedWorldModel(),
        _TiePreference(),
        horizon=1,
    ).plan(CursorBeliefState("belief:0", "question"), _retained_graph())
    assert decision.selected_action.kind is ActionKind.ABSTAIN
    assert decision.planning_status == "abstain_non_unique_partial_order"


class _FakeCategoricalClient:
    model = "openai/gpt-oss-120b"

    def __init__(self) -> None:
        self.calls = []

    def complete_json(self, *, task, payload):
        self.calls.append((task, payload))
        if "retained-node pair" in task:
            return {
                "pair_judgments": [
                    {"pair_id": row["pair_id"], "relations": []}
                    for row in payload["pairs"]
                ]
            }
        return {
            "predictions": [
                {
                    "choice": row["choice"],
                    "outcome": "inconclusive",
                    "progress": "unchanged",
                    "answerability_after": "not_ready",
                    "frontier_change": "unchanged",
                    "contradiction_change": "unchanged",
                    "resolved_roles": [],
                    "opened_roles": [],
                    "relation_updates": [],
                }
                for row in payload["requests"]
            ]
        }


def test_gpt_oss_batch_adapter_deduplicates_shared_full_graph_context() -> None:
    graph = _retained_graph()
    belief = CursorBeliefState("belief:0", "question")
    actions = GraphActionCompiler().compile(belief, graph)
    graph_input = build_iwm_graph_input(belief, graph, actions)
    client = _FakeCategoricalClient()
    predictions = GPTOSSFullGraphWorldModel(client).predict_batch(
        tuple(
            IWMRequest(belief=belief, graph_input=graph_input, action=action)
            for action in actions
        )
    )

    assert len(predictions) == len(actions)
    request_payload = client.calls[0][1]
    assert len(request_payload["contexts"]) == 1
    assert all("graph" not in row for row in request_payload["requests"])


def test_world_descriptors_are_target_bound_across_order_and_batch_size() -> None:
    graph = _retained_graph()
    belief = CursorBeliefState("belief:0", "question")
    actions = GraphActionCompiler().compile(belief, graph)
    graph_input = build_iwm_graph_input(belief, graph, actions)
    requests = tuple(
        IWMRequest(belief=belief, graph_input=graph_input, action=action)
        for action in actions
    )

    def predict(batch_size, values):
        predictions = GPTOSSFullGraphWorldModel(
            _FakeCategoricalClient(), batch_size=batch_size
        ).predict_batch(values)
        return {
            row.action.action_id: row.observation.descriptor for row in predictions
        }

    expected = {
        request.action.action_id: (
            next(
                view.key.semantic_key
                for view in graph_input.nodes
                if view.key.node_id == request.action.target_id
            ),
        )
        if request.action.reads_evidence
        else ()
        for request in requests
    }
    assert predict(1, requests) == expected
    assert predict(48, tuple(reversed(requests))) == expected


class _RealBeliefUpdateClient:
    model = "test-real-belief-updater"

    def __init__(self, responses):
        self.responses = list(responses)

    def complete_json(self, *, task, payload):
        del task, payload
        return self.responses.pop(0)


def test_real_belief_requires_evidence_lineage_and_two_observations() -> None:
    client = _RealBeliefUpdateClient(
        [
            {
                "resolved_roles": ["anchor", "outcome", "temporal_relation"],
                "opened_roles": [],
                "contradiction_change": "unchanged",
                "rationale": "first observation",
            },
            {
                "resolved_roles": [],
                "opened_roles": [],
                "contradiction_change": "unchanged",
                "rationale": "second observation",
            },
        ]
    )
    updater = GPTOSSRealEvidenceBeliefUpdater(client)
    first = _node("l1:anchor", 0.0, "puppy is drenched")
    belief = CursorBeliefState(
        "belief:0",
        "what happens after the puppy is drenched?",
        current_node_id=first.node_id,
        acquired_evidence=(first.node_id,),
        required_roles=("anchor", "outcome", "temporal_relation"),
        missing_roles=("anchor", "outcome", "temporal_relation"),
    )
    after_first = updater.update(belief, first)
    assert after_first.answerability is AnswerabilityState.NOT_READY
    assert dict(after_first.grounded_role_evidence) == {
        "anchor": first.node_id,
        "outcome": first.node_id,
        "temporal_relation": first.node_id,
    }

    second = _node("l1:outcome", 1.0, "puppy shakes off water")
    before_second = replace(
        after_first,
        current_node_id=second.node_id,
        acquired_evidence=(first.node_id, second.node_id),
    )
    after_second = updater.update(before_second, second)
    assert after_second.answerability is AnswerabilityState.READY


def test_stable_tie_flag_cannot_restore_order_based_execution() -> None:
    decision = FullGraphIWMPlanner(
        _DelayedWorldModel(),
        _TiePreference(),
        horizon=1,
        execute_stable_ties=True,
    ).plan(CursorBeliefState("belief:0", "question"), _retained_graph())
    assert decision.selected_action.kind is ActionKind.ABSTAIN
    assert decision.planning_status == "abstain_non_unique_partial_order"


def test_grounded_survivor_packet_separates_blinded_pair_and_gt_label(
    tmp_path,
) -> None:
    graph_dir = tmp_path / "video:test"
    graph_dir.mkdir()
    graph = {
        "nodes": [
            {
                "node_id": "l1:grounded",
                "time_span": {"start_s": 1.0, "end_s": 2.0},
                "text": "grounded event",
                "provenance": {"producer": "test"},
            },
            {
                "node_id": "l1:negative",
                "time_span": {"start_s": 8.0, "end_s": 9.0},
                "text": "unrelated event",
                "provenance": {"producer": "test"},
            },
        ]
    }
    (graph_dir / "l1_l15_navigation_graph.json").write_text(json.dumps(graph))

    def trajectory(trajectory_id, action_id, target_id):
        return {
            "trajectory_id": trajectory_id,
            "transitions": [
                {
                    "action": {
                        "action_id": action_id,
                        "target_id": target_id,
                        "reads_evidence": True,
                    },
                    "observation": {
                        "target_id": target_id,
                        "outcome": "support",
                        "descriptor": ["same predicted descriptor"],
                    },
                    "belief_delta": {
                        "progress": "advanced",
                        "answerability_after": "not_ready",
                        "resolved_roles": ["event"],
                    },
                }
            ],
        }

    left = trajectory("trajectory:left", "action:left", "l1:grounded")
    right = trajectory("trajectory:right", "action:right", "l1:negative")
    runs = [
        {
            "case_id": "case:test",
            "arm": "world_model_guided",
            "steps": [
                {
                    "belief_before": {
                        "required_roles": ["event"],
                        "missing_roles": ["event"],
                        "answerability": "not_ready",
                    },
                    "decision": {
                        "undominated_trajectory_ids": [
                            "trajectory:left",
                            "trajectory:right",
                        ],
                        "trajectories": [left, right],
                    },
                }
            ],
        }
    ]
    public, hidden = build_grounded_survivor_preference_packet(
        runs=runs,
        dataset={
            "cases": [
                {
                    "case_id": "case:test",
                    "planner_input": {"question": "what happened?"},
                }
            ]
        },
        hidden_key={
            "cases": [
                {
                    "case_id": "case:test",
                    "video_id": "video:test",
                    "answer_text": "must stay hidden",
                    "clue_intervals": [{"start_s": 1.25, "end_s": 1.75}],
                }
            ]
        },
        graph_root=tmp_path,
    )

    assert public["record_count"] == 1
    assert public["records"][0]["label"] is None
    serialized_public = json.dumps(public)
    assert "must stay hidden" not in serialized_public
    assert "l1:grounded" not in serialized_public
    assert "start_s" not in serialized_public
    assert hidden["records"][0]["label"] == "prefer_left"
    assert hidden["records"][0]["outcome_belief_delta_collision"] is True

    class _PreferenceClient:
        model = "test-preference-proxy"

        def complete_json(self, *, task, payload):
            del task
            return {
                "decisions": {
                    alias: {
                        "label": "prefer_left",
                        "rationale": "left better grounds the missing event",
                    }
                    for alias in payload["independent_pairs"]
                }
            }

    evaluation = evaluate_blinded_preferences(
        packet=public,
        hidden_key=hidden,
        client=_PreferenceClient(),
    )
    assert evaluation["accuracy"] == 1.0
    assert evaluation["confusion_matrix"] == {
        "prefer_left": {"prefer_left": 1}
    }
    swapped = evaluate_blinded_preferences(
        packet=public,
        hidden_key=hidden,
        client=_PreferenceClient(),
        swap_sides=True,
    )
    assert swapped["accuracy"] == 0.0
    assert swapped["confusion_matrix"] == {
        "prefer_right": {"prefer_left": 1}
    }


class _OneBadWorldResponseClient(_FakeCategoricalClient):
    def complete_json(self, *, task, payload):
        if not self.calls:
            self.calls.append((task, payload))
            return {
                "predictions": [
                    {
                        "choice": row["choice"],
                        "outcome": "inconclusive",
                        "progress": "unchanged",
                        "answerability_after": "not_ready",
                        "frontier_change": "unchanged",
                        "contradiction_change": "unchanged",
                        "resolved_roles": "not_an_array",
                        "opened_roles": [],
                        "relation_updates": [],
                    }
                    for row in payload["requests"]
                ]
            }
        return super().complete_json(task=task, payload=payload)


def test_gpt_oss_world_model_drops_out_of_allowlist_singleton_role() -> None:
    graph = _retained_graph()
    belief = CursorBeliefState("belief:0", "question")
    actions = GraphActionCompiler().compile(belief, graph)
    graph_input = build_iwm_graph_input(belief, graph, actions)
    client = _OneBadWorldResponseClient()
    world_model = GPTOSSFullGraphWorldModel(client)
    predictions = world_model.predict_batch(
        tuple(
            IWMRequest(belief=belief, graph_input=graph_input, action=action)
            for action in actions
        )
    )
    assert len(predictions) == len(actions)
    assert len(client.calls) == 1
    assert world_model.normalization_audits
    assert all(not row.belief_delta.resolved_roles for row in predictions)


class _OneBadPreferenceResponseClient:
    model = "openai/gpt-oss-120b"

    def __init__(self) -> None:
        self.calls = []

    def complete_json(self, *, task, payload):
        self.calls.append((task, payload))
        if len(self.calls) == 1:
            raise ValueError("GPT-OSS categorical output must be a JSON object")
        return {
            "comparisons": [
                {
                    "comparison": row["comparison"],
                    "label": "tie",
                    "rationale": "same categorical outcome",
                }
                for row in payload["pairs"]
            ]
        }


def test_gpt_oss_preference_model_repairs_one_non_object_response() -> None:
    graph = _retained_graph()
    belief = CursorBeliefState("belief:0", "question")
    actions = GraphActionCompiler().compile(belief, graph)
    transitions = tuple(
        ImaginedTransition(
            action=action,
            observation=PredictedObservation(
                target_id=action.target_id,
                outcome=EvidenceOutcome.INCONCLUSIVE,
            ),
            belief_delta=CategoricalBeliefDelta(
                progress=ProgressChange.UNCHANGED,
                answerability_after=AnswerabilityState.NOT_READY,
            ),
        )
        for action in actions[:2]
    )
    pair = TrajectoryPair(
        TrajectoryPrediction("trajectory:left", (transitions[0],)),
        TrajectoryPrediction("trajectory:right", (transitions[1],)),
    )
    client = _OneBadPreferenceResponseClient()
    result = GPTOSSFullGraphPreferenceModel(client).compare_batch((pair,), belief)
    assert result[0].label is PreferenceLabel.TIE
    assert len(client.calls) == 2
    trajectories = client.calls[-1][1]["trajectories"]
    serialized = json.dumps(trajectories)
    assert "action_id" not in serialized
    assert "target_id" not in serialized
    assert "edge_id" not in serialized


class _TiePreferenceResponseClient:
    model = "openai/gpt-oss-120b"

    def __init__(self) -> None:
        self.calls = []

    def complete_json(self, *, task, payload):
        self.calls.append((task, payload))
        return {
            "comparisons": [
                {
                    "comparison": row["comparison"],
                    "label": "tie",
                    "rationale": "same categorical consequence",
                }
                for row in payload["pairs"]
            ]
        }


def test_gpt_oss_preference_batches_all_pairs_without_sampling() -> None:
    graph = _retained_graph()
    belief = CursorBeliefState("belief:0", "question")
    actions = GraphActionCompiler().compile(belief, graph)
    trajectories = tuple(
        TrajectoryPrediction(
            f"trajectory:{index}",
            (
                ImaginedTransition(
                    action=action,
                    observation=PredictedObservation(
                        target_id=action.target_id,
                        outcome=EvidenceOutcome.INCONCLUSIVE,
                    ),
                    belief_delta=CategoricalBeliefDelta(
                        progress=ProgressChange.UNCHANGED,
                        answerability_after=AnswerabilityState.NOT_READY,
                    ),
                ),
            ),
        )
        for index, action in enumerate(actions)
    )
    pairs = tuple(
        TrajectoryPair(left, right)
        for left_index, left in enumerate(trajectories)
        for right in trajectories[left_index + 1 :]
    )
    client = _TiePreferenceResponseClient()

    results = GPTOSSFullGraphPreferenceModel(client, batch_size=2).compare_batch(
        pairs,
        belief,
    )

    assert len(results) == len(pairs)
    assert len(client.calls) == (len(pairs) + 1) // 2
    assert all(len(payload["pairs"]) <= 2 for _, payload in client.calls)


def test_gpt_oss_correlation_evaluator_covers_all_pairs_across_batches() -> None:
    nodes = [
        _node(f"l1:{index}", float(index), f"evidence {index}") for index in range(3)
    ]
    pairs = tuple(
        CorrelationPair(nodes[left], nodes[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    client = _FakeCategoricalClient()
    judgments = GPTOSSCategoricalCorrelationEvaluator(
        client,
        batch_size=2,
    ).evaluate(pairs)

    assert judgments == ()
    assert len(client.calls) == 2
    assert sum(len(payload["pairs"]) for _, payload in client.calls) == 3


class _FirstUnreadPlanner:
    def plan(self, belief, graph):
        actions = GraphActionCompiler().compile(belief, graph)
        selected = next(
            (
                action
                for action in actions
                if action.reads_evidence
                and action.target_id not in belief.acquired_evidence
                and (
                    action.target_id == "l1:bridge"
                    or "l1:bridge" in belief.acquired_evidence
                )
            ),
            next(action for action in actions if action.kind is ActionKind.STOP),
        )
        from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
            FullGraphPlanDecision,
        )

        return FullGraphPlanDecision(
            selected_action=selected,
            planning_status="test_first_unread",
            trajectories=(),
            preferences=(),
            undominated_trajectory_ids=(),
            legal_action_count=len(actions),
        )


def test_real_closed_loop_never_feeds_hidden_clue_label_back_to_planner() -> None:
    graph = _retained_graph()
    before = graph_fingerprint(graph)
    run = run_real_read_closed_loop(
        case_id="case:test",
        question="question",
        graph=graph,
        clue_intervals=(ClueInterval(0.0, 0.5),),
        planner=_FirstUnreadPlanner(),
        read_budget=1,
        arm="test",
    )

    assert run["metrics"]["clue_coverage_complete"] is True
    label = run["steps"][0]["realized_label_evaluator_only"]
    assert label["belief_delta"]["answerability_after"] == "ready"
    assert label["fed_back_to_planner"] is False
    assert run["final_visible_belief"]["answerability"] == "not_ready"
    assert run["final_visible_belief"]["missing_roles"] == []
    assert run["hidden_evaluator_feedback_to_planner"] is False
    assert graph_fingerprint(graph) == before == run["graph_fingerprint"]


def test_oracle_uses_intermediate_navigation_hops_to_reach_later_clue() -> None:
    nodes = tuple(
        _node(f"l1:{index}", float(index), f"evidence {index}")
        for index in range(3)
    )
    graph = RetainedEvidenceGraph(
        graph_id="retained:oracle-path",
        nodes=nodes,
        temporal_edges=(
            TemporalNavigationEdge("time:0:1", "l1:0", "l1:1", "temporal_next"),
            TemporalNavigationEdge("time:1:2", "l1:1", "l1:2", "temporal_next"),
        ),
        correlation_edges=(),
        capacity=3,
    )

    run = run_oracle_clue_ceiling(
        case_id="case:oracle-path",
        question="what connects the first and last evidence?",
        graph=graph,
        clue_intervals=(ClueInterval(0.1, 0.9), ClueInterval(2.1, 2.9)),
        read_budget=3,
    )

    assert run["termination"] == "oracle_coverage_complete"
    assert run["metrics"]["clue_recall"] == 1.0
    assert [
        step["selected_action"]["target_id"] for step in run["steps"]
    ] == ["l1:0", "l1:1", "l1:2"]


class _StepSensitiveWorldModel:
    model_name = "step-sensitive"

    def predict_batch(self, requests):
        return tuple(
            ImaginedTransition(
                action=request.action,
                observation=PredictedObservation(
                    target_id=request.action.target_id,
                    outcome=(
                        EvidenceOutcome.SUPPORT
                        if request.belief.step == 0
                        else EvidenceOutcome.COUNTEREVIDENCE
                    ),
                ),
                belief_delta=CategoricalBeliefDelta(
                    progress=(
                        ProgressChange.ADVANCED
                        if request.belief.step == 0
                        else ProgressChange.REGRESSED
                    ),
                    answerability_after=AnswerabilityState.NOT_READY,
                ),
            )
            for request in requests
        )


def test_shuffle_and_frozen_interventions_preserve_legal_action_identity() -> None:
    graph = _retained_graph()
    belief = CursorBeliefState("belief:0", "question")
    actions = GraphActionCompiler().compile(belief, graph)
    graph_input = build_iwm_graph_input(belief, graph, actions)
    requests = tuple(
        IWMRequest(belief=belief, graph_input=graph_input, action=action)
        for action in actions
    )
    shuffled_model = ShuffledWorldModel(_StepSensitiveWorldModel())
    shuffled = tuple(shuffled_model.predict_batch(requests))
    assert [row.action for row in shuffled] == list(actions)
    assert [row.observation.target_id for row in shuffled] == [
        action.target_id for action in actions
    ]
    assert [row.observation.descriptor for row in shuffled] == [
        (
            next(
                view.key.semantic_key
                for view in graph_input.nodes
                if view.key.node_id == action.target_id
            ),
        )
        if action.reads_evidence
        else ()
        for action in actions
    ]

    frozen_model = FrozenWorldModel(_StepSensitiveWorldModel())
    first = tuple(frozen_model.predict_batch(requests))
    later_belief = replace(belief, belief_id="belief:1", step=1)
    later_input = build_iwm_graph_input(later_belief, graph, actions)
    later_requests = tuple(
        IWMRequest(belief=later_belief, graph_input=later_input, action=action)
        for action in actions
    )
    second = tuple(frozen_model.predict_batch(later_requests))
    assert [row.action for row in second] == list(actions)
    assert [row.observation.outcome for row in second] == [
        row.observation.outcome for row in first
    ]
    assert frozen_model.cache_audits[-1]["reused_prediction_count"] == len(actions)


def test_planner_resource_abstains_without_top_k_or_partial_pair_sampling() -> None:
    decision = FullGraphIWMPlanner(
        _DelayedWorldModel(),
        _TiePreference(),
        horizon=1,
        max_trajectory_pairs=2,
    ).plan(CursorBeliefState("belief:0", "question"), _retained_graph())

    assert decision.selected_action.kind is ActionKind.ABSTAIN
    assert decision.planning_status == "abstain_exhaustive_comparison_budget_exceeded"
    assert decision.preferences == ()
    assert decision.top_k_applied is False


class _ReactivePreferenceClient:
    model = "openai/gpt-oss-120b"

    def __init__(self) -> None:
        self.calls = []

    def complete_json(self, *, task, payload):
        self.calls.append((task, payload))
        preferred = next(
            alias
            for alias, action in payload["actions"].items()
            if action["target_id"] == "l1:bridge"
        )
        comparisons = []
        for pair in payload["pairs"]:
            if pair["left"] == preferred:
                label = "prefer_left"
            elif pair["right"] == preferred:
                label = "prefer_right"
            else:
                label = "tie"
            comparisons.append(
                {
                    "comparison": pair["comparison"],
                    "label": label,
                    "rationale": "current belief favors this direct read",
                }
            )
        return {"comparisons": comparisons}


def test_no_world_model_arm_is_a_reactive_direct_policy_not_null_iwm() -> None:
    client = _ReactivePreferenceClient()
    decision = GPTOSSReactiveGraphPlanner(client, preference_batch_size=2).plan(
        CursorBeliefState("belief:0", "question"),
        _retained_graph(),
    )

    assert decision.selected_action.target_id == "l1:bridge"
    assert decision.planning_status == "reactive_selected_unique_undominated_action"
    assert decision.top_k_applied is False
    assert len(client.calls) == 3
    task, payload = client.calls[0]
    assert "current real belief" in task
    assert payload["required_contract"]["no_imagined_transition"] is True
    assert "predictions" not in payload


def test_cgbench_gate_uses_public_selection_then_hidden_retention_only(
    tmp_path,
) -> None:
    import numpy as np

    matrix_path = tmp_path / "node_embeddings.npy"
    matrix = np.zeros((2, 2048), dtype=np.float32)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    np.save(matrix_path, matrix)
    first = replace(
        _node("l1:first", 0.0, "first evidence"),
        embedding_ref=EmbeddingRef(
            path=str(matrix_path),
            dimension=2048,
            row_index=0,
        ),
    )
    second = replace(
        _node("l1:second", 1.0, "second evidence"),
        embedding_ref=EmbeddingRef(
            path=str(matrix_path),
            dimension=2048,
            row_index=1,
        ),
    )
    overlay = CausalTemporalOverlay(
        overlay_id="overlay:cgbench-test",
        example_id="example:cgbench-test",
        video_id="video:test",
        l1_observations=[first, second],
        atomic_events=[],
        relations=[],
        l1_structural_relations=[_temporal(first, second)],
        metadata={
            "question_independent_contract": True,
            "observation_end_s": 10.0,
            "input_mode": "video_only",
            "layer_contract": "l1_observations_plus_l1_5_atomic_overlay",
        },
    )
    graph_path = tmp_path / "video:test" / "causal_temporal_overlay.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(json.dumps(overlay.to_dict()), encoding="utf-8")
    dataset = {
        "dataset_id": "cgbench-test",
        "cases": [
            {
                "case_id": "case:test",
                "video_id": "video:test",
                "split": "test",
                "planner_input": {"question": "what happened?"},
                "executed_transitions": [
                    {"target": {"answerability_after": "unknown"}}
                ],
            }
        ],
    }
    hidden = {
        "cases": [
            {
                "case_id": "case:test",
                "answer_text": "secret",
                "answer_key": "A",
                "clue_intervals": [{"start_s": 0.1, "end_s": 0.9}],
            }
        ]
    }
    selection = {
        "schema_version": "test-selection",
        "videos": [{"video_id": "video:test"}],
    }

    gate = compile_cgbench_gate(
        dataset=dataset,
        hidden_key=hidden,
        selection=selection,
        graph_root=tmp_path,
        capacity=2,
        read_budget=2,
        video_limit=1,
    )

    assert gate["gate_passed"] is True
    assert gate["runnable_case_ids"] == ["case:test"]
    assert gate["cases"][0]["forbidden_planner_keys"] == []
    assert gate["cases"][0]["unread_evidence_value_leak_count"] == 0
    assert gate["cases"][0]["retained_clue_recall_evaluator_only"] == 1.0
    assert gate["dataset_boundary"] == "public_artifact_contains_no_hidden_fields"
    assert gate["checks"]["all_graphs_have_embedding_correlation_build"] is True
    assert gate["gpt_service_called"] is False
