from __future__ import annotations


import pytest

from steam_video_new.implicit_world_model.reasoning_v2.evaluation import (
    EntryProtocol,
    FrozenReasoningCase,
    build_entry_frontiers,
    evaluate_frozen_cohort,
    shortest_path_distance,
    shortest_path_next_hops,
    with_entry_frontier,
)
from steam_video_new.implicit_world_model.reasoning_v2.evidence import (
    EvidenceAddress,
    EvidenceMemory,
    EvidenceRecord,
    EvidenceValue,
)
from steam_video_new.implicit_world_model.reasoning_v2.navigation import (
    NavigationGraph,
    NavigationProposal,
    ProposalKind,
    evaluate_proposals,
)
from steam_video_new.implicit_world_model.reasoning_v2.planner import (
    PathStatus,
    PersistentMultiPathPlanner,
    PreferenceStatus,
    TrajectoryPreferenceDecision,
    apply_real_read,
    initialize_forest,
)
from steam_video_new.implicit_world_model.reasoning_v2.world_model import (
    AddressOnlyObservationBaseline,
    Answerability,
    BeliefState,
    ConservativeEffectBaseline,
    GroundedBeliefEffect,
    ObservationDescriptor,
    ObservationKind,
    ObservationPrediction,
    validate_observation_predictions,
)


def _memory(*node_ids: str) -> EvidenceMemory:
    return EvidenceMemory(
        memory_id="memory:test",
        records=tuple(
            EvidenceRecord(
                EvidenceAddress(
                    node_id=node_id,
                    video_id="video",
                    start_s=float(index),
                    end_s=float(index + 1),
                    event_family="action",
                    semantic_key=f"safe semantic key {node_id}",
                    source_segments=(f"clip:{node_id}",),
                ),
                EvidenceValue(
                    descriptor=f"grounded private value {node_id}",
                    predicate=f"predicate_{node_id}",
                ),
            )
            for index, node_id in enumerate(node_ids)
        ),
        temporal_links=(),
        metadata={"formal_learned_representation": True},
    )


def _proposal(src: str, dst: str, *, suffix: str) -> NavigationProposal:
    return NavigationProposal(
        proposal_id=f"proposal:{suffix}",
        src=src,
        dst=dst,
        kind=ProposalKind.SEMANTIC_NEIGHBOR,
        bidirectional=True,
        evidence_refs=(src, dst),
    )


def test_evidence_address_does_not_leak_unread_grounded_value() -> None:
    memory = _memory("a", "b")
    views = memory.address_view(("a",))
    assert views[0]["value"].predicate == "predicate_a"
    assert views[1]["value"] is None
    assert views[1]["address"].event_family == "action"
    assert views[1]["address"].semantic_key == "safe semantic key b"
    assert "predicate_b" not in repr(views[1]["address"])


class _FixedEntryLocalizer:
    def __init__(self, selected: tuple[str, ...]) -> None:
        self.selected = selected
        self.audits = []

    def localize(self, *, question, missing_roles, memory):
        del question, missing_roles
        self.audits.append(
            {
                "candidate_address_count": len(memory.records),
                "selected_node_ids": list(self.selected),
                "all_addresses_inspected": True,
                "top_k_applied": False,
            }
        )
        return self.selected


def test_complete_graph_entry_protocols_only_change_the_frontier() -> None:
    memory = _memory("a", "b", "c")
    graph = NavigationGraph(
        "graph",
        ("a", "b", "c"),
        (_proposal("a", "b", suffix="ab"), _proposal("b", "c", suffix="bc")),
    )
    oracle, learned = build_entry_frontiers(
        question="question",
        missing_roles=("answer",),
        memory=memory,
        expected_first_clue_ids=("a",),
        localizer=_FixedEntryLocalizer(("c",)),
    )

    assert oracle.protocol is EntryProtocol.ORACLE
    assert oracle.node_ids == ("a",)
    assert learned.protocol is EntryProtocol.LEARNED
    assert learned.node_ids == ("c",)
    for frontier in (oracle, learned):
        updated = with_entry_frontier(graph, frontier)
        assert updated.entry_node_ids == frontier.node_ids
        assert updated.node_ids == graph.node_ids
        assert updated.proposals == graph.proposals
        assert updated.metadata["complete_graph_preserved"] is True


def test_shortest_path_next_hops_are_evaluator_only_route_targets() -> None:
    graph = NavigationGraph(
        "graph",
        ("a", "b", "c", "d"),
        (
            _proposal("a", "b", suffix="ab"),
            _proposal("b", "c", suffix="bc"),
            _proposal("a", "d", suffix="ad"),
            _proposal("d", "c", suffix="dc"),
        ),
    )

    assert shortest_path_next_hops(
        graph, source_ids=("a",), goal_ids=("c",)
    ) == ("b", "d")


def test_shortest_path_distance_reports_budget_relevant_edge_count() -> None:
    graph = NavigationGraph(
        "graph",
        ("a", "b", "c", "d"),
        (
            _proposal("a", "b", suffix="ab"),
            _proposal("b", "c", suffix="bc"),
        ),
    )

    assert shortest_path_distance(graph, source_ids=("a",), goal_ids=("c",)) == 2
    assert shortest_path_distance(graph, source_ids=("a",), goal_ids=("a",)) == 0
    assert shortest_path_distance(graph, source_ids=("a",), goal_ids=("d",)) is None


def test_proposal_calibration_requires_trusted_hard_negatives() -> None:
    graph = NavigationGraph(
        "graph",
        ("a", "b", "c"),
        (_proposal("a", "b", suffix="ab"),),
    )
    missing_negative = evaluate_proposals(graph, {frozenset(("a", "b")): "positive"})
    assert not missing_negative.passed
    assert "trusted_hard_negatives_missing" in missing_negative.blockers

    calibrated = evaluate_proposals(
        graph,
        {
            frozenset(("a", "b")): "positive",
            frozenset(("a", "c")): "hard_negative",
        },
    )
    assert calibrated.passed
    assert calibrated.precision == 1.0


def test_observation_contract_rejects_hypothesis_dependent_physical_outcome() -> None:
    memory = _memory("a")
    from steam_video_new.implicit_world_model.reasoning_v2.navigation.actions import (
        compile_legal_actions,
    )
    from steam_video_new.implicit_world_model.reasoning_v2.world_model import (
        ObservationContext,
        ObservationRequest,
    )

    graph = NavigationGraph("graph", ("a",), (), entry_node_ids=("a",))
    action = compile_legal_actions(graph, cursor_id=None, acquired_ids=())[0]
    belief = BeliefState("question")
    context = ObservationContext(())
    requests = tuple(
        ObservationRequest(
            request_id=f"request:{index}",
            belief=belief,
            action=action,
            target_address=memory.read("a").address,
            context=context,
        )
        for index in range(2)
    )
    predictions = (
        ObservationPrediction(
            "request:0",
            action.action_id,
            "a",
            ObservationDescriptor(ObservationKind.EVENT, "action"),
        ),
        ObservationPrediction(
            "request:1",
            action.action_id,
            "a",
            ObservationDescriptor(ObservationKind.STATE, "action"),
        ),
    )
    with pytest.raises(ValueError, match="depends on reasoning hypothesis"):
        validate_observation_predictions(requests, predictions)


class _SelectTargetA:
    model_name = "select-target-a"

    def choose(self, forest, candidates):
        del forest
        selected = next(
            (row for row in candidates if row.first_action.target_id == "a"),
            candidates[0],
        )
        return TrajectoryPreferenceDecision(
            PreferenceStatus.SELECT,
            tuple(row.tree_id for row in candidates),
            selected.tree_id,
            "categorically select target a",
        )


class _CountingObservationModel(AddressOnlyObservationBaseline):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict_batch(self, requests):
        self.batch_sizes.append(len(requests))
        return super().predict_batch(requests)


class _RecordingCorrector:
    model_name = "recording-corrector"

    def __init__(self) -> None:
        self.calls = []

    def correct_batch(
        self,
        *,
        path_hypotheses,
        beliefs,
        acquired_ids_before,
        observation,
    ):
        self.calls.append(
            (
                dict(path_hypotheses),
                acquired_ids_before,
                observation.target_id,
            )
        )
        return tuple(
            GroundedBeliefEffect(
                path_id,
                BeliefState(
                    belief.question,
                    required_roles=belief.required_roles,
                    missing_roles=(),
                    answerability=Answerability.READY,
                ),
                direct_same_target=True,
                verified=True,
            )
            for path_id, belief in beliefs.items()
        )


def test_planner_keeps_distinct_unselected_reasoning_paths() -> None:
    memory = _memory("a", "b")
    navigation = NavigationGraph(
        "graph",
        ("a", "b"),
        (),
        entry_node_ids=("a", "b"),
    )
    belief = BeliefState(
        "Which event answers the question?",
        required_roles=("answer_evidence",),
        missing_roles=("answer_evidence",),
        answerability=Answerability.NOT_READY,
    )
    forest = initialize_forest(
        forest_id="forest",
        belief=belief,
        hypotheses=("hypothesis_a", "hypothesis_b"),
        read_budget=2,
    )
    observation_model = _CountingObservationModel()
    planner = PersistentMultiPathPlanner(
        observation_model,
        ConservativeEffectBaseline(),
        _SelectTargetA(),
        horizon=1,
    )
    decision = planner.plan(forest, memory, navigation)

    # Two physical addresses are predicted once each, not once per answer
    # hypothesis. Effects remain hypothesis/path conditioned.
    assert observation_model.batch_sizes == [2]
    assert decision.observation_prediction_count == 2
    assert decision.effect_prediction_count == 4
    assert len(decision.candidates) == 2
    assert all(
        set(tree.covered_hypotheses) == {"hypothesis_a", "hypothesis_b"}
        for tree in decision.candidates
    )

    updated = apply_real_read(forest, decision, memory)
    assert updated.shared_evidence.acquired_ids == ("a",)
    assert len(updated.paths) == 4
    assert {row.status for row in updated.paths} == {
        PathStatus.ACTIVE,
        PathStatus.SUSPENDED,
    }
    active = [row for row in updated.paths if row.status is PathStatus.ACTIVE]
    suspended = [row for row in updated.paths if row.status is PathStatus.SUSPENDED]
    assert all(len(row.action_history) == 1 for row in active)
    assert all(row.action_history == () for row in suspended)
    assert all(row.pending_first_action_id is not None for row in suspended)


def test_second_hop_observation_is_deduplicated_across_hypotheses() -> None:
    memory = _memory("a", "b", "c")
    navigation = NavigationGraph(
        "graph",
        ("a", "b", "c"),
        (
            _proposal("a", "c", suffix="ac"),
            _proposal("b", "c", suffix="bc"),
        ),
        entry_node_ids=("a", "b"),
    )
    forest = initialize_forest(
        forest_id="forest:horizon-two",
        belief=BeliefState("question"),
        hypotheses=("left", "right"),
        read_budget=2,
    )
    observation_model = _CountingObservationModel()
    planner = PersistentMultiPathPlanner(
        observation_model,
        ConservativeEffectBaseline(),
        _SelectTargetA(),
        horizon=2,
    )
    decision = planner.plan(forest, memory, navigation)

    # First-hop a/b are each predicted once. Second-hop c is predicted once
    # after imagined a and once after imagined b, never once per hypothesis.
    assert observation_model.batch_sizes == [2, 2]
    assert decision.observation_prediction_count == 4
    assert all(
        set(tree.covered_hypotheses) == {"left", "right"}
        for tree in decision.candidates
    )

    after_first_read = apply_real_read(forest, decision, memory)
    # A real acquired EvidenceValue contains dict provenance in production. It
    # must remain safe in the next physical-world deduplication key.
    next_decision = planner.plan(after_first_read, memory, navigation)
    assert next_decision.candidates


def test_real_read_correction_is_one_shared_post_execution_batch() -> None:
    memory = _memory("a", "b")
    navigation = NavigationGraph(
        "graph",
        ("a", "b"),
        (),
        entry_node_ids=("a", "b"),
    )
    belief = BeliefState(
        "question",
        required_roles=("answer",),
        missing_roles=("answer",),
    )
    forest = initialize_forest(
        forest_id="forest:correct",
        belief=belief,
        hypotheses=("left", "right"),
        read_budget=2,
    )
    planner = PersistentMultiPathPlanner(
        AddressOnlyObservationBaseline(),
        ConservativeEffectBaseline(),
        _SelectTargetA(),
        horizon=1,
    )
    decision = planner.plan(forest, memory, navigation)
    corrector = _RecordingCorrector()
    updated = apply_real_read(
        forest,
        decision,
        memory,
        corrector=corrector,
    )

    assert len(corrector.calls) == 1
    assert corrector.calls[0][1:] == ((), "a")
    assert len(corrector.calls[0][0]) == len(updated.paths)
    assert all(
        path.belief.answerability is Answerability.READY for path in updated.paths
    )


def test_delayed_gate_requires_a_real_second_graph_hop() -> None:
    memory = _memory("a", "b", "c")
    navigation = NavigationGraph(
        "graph",
        ("a", "b", "c"),
        (
            _proposal("a", "b", suffix="ab"),
            _proposal("b", "c", suffix="bc"),
        ),
        entry_node_ids=("a",),
    )
    report = evaluate_frozen_cohort(
        memory,
        navigation,
        (FrozenReasoningCase("case", ("a",), (("c",),), 3),),
        minimum_case_count=1,
        minimum_delayed_cases=1,
    )
    assert report.passed
    assert report.cases[0].shortest_clue_hops == (2,)
    assert report.cases[0].delayed_candidate is True


def test_cohort_gate_counts_entry_as_a_real_read() -> None:
    memory = _memory("a", "b", "c")
    navigation = NavigationGraph(
        "graph",
        ("a", "b", "c"),
        (
            _proposal("a", "b", suffix="ab"),
            _proposal("b", "c", suffix="bc"),
        ),
        entry_node_ids=("a",),
    )

    report = evaluate_frozen_cohort(
        memory,
        navigation,
        (FrozenReasoningCase("case", ("a",), (("c",),), 2),),
        minimum_case_count=1,
        minimum_delayed_cases=0,
    )

    assert report.cases[0].shortest_clue_hops == (None,)
    assert report.cases[0].all_clues_reachable is False
    assert "legal_navigation_reachability_incomplete" in report.blockers
