from __future__ import annotations

from dataclasses import replace

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
    IWMRequest,
    ImaginedTransition,
    PreferenceLabel,
    RetainedEvidenceGraph,
    TemporalNavigationEdge,
    TrajectoryPrediction,
    execute_graph_action,
    build_retained_graph_from_legacy_overlay,
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


def test_legacy_overlay_adapter_builds_fixed_capacity_categorical_graph() -> None:
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
    )

    graph = build_retained_graph_from_legacy_overlay(legacy, capacity=2)

    assert len(graph.nodes) == 2
    assert (
        graph.metadata["navigation_contract"]
        == "single_cursor_all_legal_actions_no_top_k"
    )
    assert any(
        edge.relation is CorrelationType.TRANSITION_SUPPORT
        and edge.status is CorrelationStatus.CANDIDATE
        for edge in graph.correlation_edges
    )


def _retained_graph(*, include_hidden: bool = False) -> RetainedEvidenceGraph:
    first = _node("l1:bridge", 0.0, "bridge evidence")
    second = _node("l1:answer", 1.0, "answer evidence")
    nodes = [first, second]
    if include_hidden:
        nodes.append(_node("l1:hidden", 2.0, "hidden answer", hidden=True))
    correlation = CorrelationEdge(
        edge_id="corr:bridge-answer",
        src=first.node_id,
        dst=second.node_id,
        relation=CorrelationType.TRANSITION_SUPPORT,
        status=CorrelationStatus.CANDIDATE,
        evidence_refs=(first.node_id, second.node_id),
        candidate_sources=("test",),
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
        action.kind is ActionKind.SEMANTIC_PROBE and action.target_id == "l1:answer"
        for action in actions
    )
    assert any(
        action.kind is ActionKind.INSPECT_CORRELATION
        and action.source_id == "l1:bridge"
        for action in actions
    )
    changed_question = replace(execution.updated_belief, question="unrelated words")
    assert {action.action_id for action in actions} == {
        action.action_id for action in compiler.compile(changed_question, graph)
    }

    model_input = build_iwm_graph_input(execution.updated_belief, graph, actions)
    unread = next(view for view in model_input.nodes if view.key.node_id == "l1:answer")
    assert unread.evidence_value is None
    assert unread.key.embedding_ref is not None


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


def test_gpt_oss_world_model_repairs_one_invalid_field_schema() -> None:
    graph = _retained_graph()
    belief = CursorBeliefState("belief:0", "question")
    actions = GraphActionCompiler().compile(belief, graph)
    graph_input = build_iwm_graph_input(belief, graph, actions)
    client = _OneBadWorldResponseClient()
    predictions = GPTOSSFullGraphWorldModel(client).predict_batch(
        tuple(
            IWMRequest(belief=belief, graph_input=graph_input, action=action)
            for action in actions
        )
    )
    assert len(predictions) == len(actions)
    assert len(client.calls) == 2


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
