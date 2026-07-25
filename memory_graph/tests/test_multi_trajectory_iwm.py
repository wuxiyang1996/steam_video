from __future__ import annotations

from dataclasses import replace

import pytest

from memory_graph.types import MemoryNode, TimeSpan
from steam_video_new.implicit_world_model.full_graph_iwm import (
    ActionKind,
    CursorBeliefState,
    EvidenceEffect,
    GPTOSSMultiTrajectoryIWM,
    GPTOSSRealEvidenceBeliefUpdater,
    GPTOSSRealTrajectoryEvidenceAssessor,
    GraphActionCompiler,
    MultiTrajectoryIWMDecision,
    MultiTrajectoryIWMPlanner,
    MultiTrajectoryPlanDecision,
    PoolPreferenceStatus,
    PredictedLifecycle,
    ReasoningTrajectory,
    RetainedEvidenceGraph,
    TemporalNavigationEdge,
    TrajectoryEvidenceAssessment,
    TrajectoryExpansion,
    TrajectoryPool,
    TrajectoryStatus,
    branch_trajectory_pool,
    consolidate_trajectory_pool,
    execute_shared_trajectory_action,
    initialize_trajectory_pool,
    run_multi_trajectory_closed_loop,
)


def _node(node_id: str, start: float, text: str) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        video_id="video:test",
        time_span=TimeSpan(start, start + 1.0),
        provenance={"producer": "test"},
        node_type="observation",
        text=text,
        metadata={"predicate": text, "layer": "L1"},
    )


def _graph() -> RetainedEvidenceGraph:
    first = _node("l1:first", 0.0, "person enters room")
    second = _node("l1:second", 1.0, "person opens door")
    return RetainedEvidenceGraph(
        graph_id="graph:test",
        nodes=(first, second),
        temporal_edges=(
            TemporalNavigationEdge(
                "time:first-second", first.node_id, second.node_id, "temporal_next"
            ),
        ),
        correlation_edges=(),
        capacity=2,
    )


def _pool() -> TrajectoryPool:
    return initialize_trajectory_pool(
        CursorBeliefState(
            belief_id="belief:root",
            question="Who opened the door after entering?",
            localized_entry_node_ids=("l1:first", "l1:second"),
            required_roles=("identity", "after_event"),
            missing_roles=("identity", "after_event"),
            remaining_reads=2,
        ),
        ("the entrant opened the door", "another person opened the door"),
    )


class _SharedFirstIWM:
    model_name = "shared-first-test-iwm"

    def prefer(self, pool, expansions, graph):
        del graph
        preferred = tuple(
            row.expansion_id for row in expansions if row.action.target_id == "l1:first"
        )
        return MultiTrajectoryIWMDecision(
            status=PoolPreferenceStatus.TIE,
            preferred_expansion_ids=preferred,
            lifecycle_predictions=tuple(
                PredictedLifecycle(row.trajectory_id, TrajectoryStatus.ACTIVE)
                for row in pool.expandable
            ),
        )


class _DistinctActionIWM:
    model_name = "distinct-action-test-iwm"

    def prefer(self, pool, expansions, graph):
        del pool, graph
        first = next(row for row in expansions if row.action.target_id == "l1:first")
        second = next(row for row in expansions if row.action.target_id == "l1:second")
        return MultiTrajectoryIWMDecision(
            status=PoolPreferenceStatus.TIE,
            preferred_expansion_ids=(first.expansion_id, second.expansion_id),
        )


def test_multi_trajectory_planner_executes_one_shared_preferred_action() -> None:
    pool = _pool()
    decision = MultiTrajectoryIWMPlanner(_SharedFirstIWM()).plan(pool, _graph())

    assert decision.selected_action.target_id == "l1:first"
    assert decision.planning_status == "selected_shared_action_across_trajectories"
    assert len(decision.preferred_expansion_ids) == 2
    assert decision.legal_expansion_count == 8
    assert decision.top_k_applied is False


def test_multi_trajectory_planner_abstains_for_distinct_preferred_actions() -> None:
    decision = MultiTrajectoryIWMPlanner(_DistinctActionIWM()).plan(_pool(), _graph())

    assert decision.selected_action.kind is ActionKind.ABSTAIN
    assert decision.planning_status == "abstain_distinct_preferred_actions"


class _HypothesisAssessor:
    def assess_batch(self, trajectories, observation, corrected_beliefs):
        del observation, corrected_beliefs
        return tuple(
            TrajectoryEvidenceAssessment(
                trajectory.trajectory_id,
                (
                    EvidenceEffect.SUPPORT
                    if trajectory.hypothesis.startswith("the entrant")
                    else EvidenceEffect.COUNTEREVIDENCE
                ),
                "categorical grounded assessment",
            )
            for trajectory in trajectories
        )


def test_one_real_read_is_broadcast_to_every_active_trajectory() -> None:
    pool = _pool()
    planner = MultiTrajectoryIWMPlanner(_SharedFirstIWM())
    decision = planner.plan(pool, _graph())
    execution = execute_shared_trajectory_action(
        pool,
        decision,
        _graph(),
        assessor=_HypothesisAssessor(),
    )

    assert execution.observation is not None
    assert execution.observation.node_id == "l1:first"
    assert len(execution.assessments) == 2
    supported, contradicted = execution.pool.trajectories
    assert supported.status is TrajectoryStatus.SUPPORTED
    assert contradicted.status is TrajectoryStatus.CONTRADICTED
    assert supported.belief.current_node_id == "l1:first"
    assert contradicted.belief.current_node_id == "l1:first"
    assert all(
        row.belief.acquired_evidence == ("l1:first",)
        for row in execution.pool.trajectories
    )
    assert all(row.belief.remaining_reads == 1 for row in execution.pool.trajectories)
    assert all(
        row.shared_observation_ids == ("l1:first",)
        for row in execution.pool.trajectories
    )
    assert all(
        row.action_history == (decision.selected_action.action_id,)
        for row in execution.pool.trajectories
    )


class _ActionAwareUpdater:
    def __init__(self):
        self.calls = []

    def update_after_action(
        self,
        trajectory_id,
        previous_belief,
        structurally_updated_belief,
        action,
        observation,
        graph,
    ):
        self.calls.append(
            (
                trajectory_id,
                previous_belief.acquired_evidence,
                structurally_updated_belief.acquired_evidence,
                action.target_id,
                observation.node_id,
                graph.graph_id,
            )
        )
        return structurally_updated_belief


def test_action_aware_backup_receives_only_executed_real_observation() -> None:
    pool = _pool()
    decision = MultiTrajectoryIWMPlanner(_SharedFirstIWM()).plan(pool, _graph())
    updater = _ActionAwareUpdater()

    execute_shared_trajectory_action(
        pool,
        decision,
        _graph(),
        belief_updater=updater,
    )

    assert len(updater.calls) == 2
    assert all(row[1] == () for row in updater.calls)
    assert all(row[2] == ("l1:first",) for row in updater.calls)
    assert all(
        row[3:] == ("l1:first", "l1:first", "graph:test") for row in updater.calls
    )


class _HypothesisAwareUpdater:
    def __init__(self):
        self.hypotheses = []

    def update_for_trajectory(
        self,
        trajectory_id,
        hypothesis,
        previous_belief,
        structurally_updated_belief,
        action,
        observation,
        graph,
    ):
        del (
            trajectory_id,
            previous_belief,
            action,
            observation,
            graph,
        )
        self.hypotheses.append(hypothesis)
        return replace(
            structurally_updated_belief,
            contradictions=(f"hypothesis:{hypothesis}",),
        )


def test_real_belief_correction_is_conditioned_on_each_hypothesis() -> None:
    pool = _pool()
    decision = MultiTrajectoryIWMPlanner(_SharedFirstIWM()).plan(pool, _graph())
    updater = _HypothesisAwareUpdater()

    execution = execute_shared_trajectory_action(
        pool,
        decision,
        _graph(),
        belief_updater=updater,
    )

    assert updater.hypotheses == [row.hypothesis for row in pool.trajectories]
    assert {row.belief.contradictions for row in execution.pool.trajectories} == {
        ("hypothesis:the entrant opened the door",),
        ("hypothesis:another person opened the door",),
    }


class _HypothesisConditionedCorrectionClient:
    model = "hypothesis-conditioned-correction-test"

    def __init__(self):
        self.payloads = []

    def complete_json(self, *, task, payload):
        del task
        self.payloads.append(payload)
        resolves_identity = payload["trajectory_hypothesis"].startswith("the entrant")
        return {
            "resolved_roles": ["identity"] if resolves_identity else [],
            "opened_roles": [],
            "contradiction_change": "unchanged",
            "rationale": "the real observation is assessed under this hypothesis",
        }


def test_gpt_real_updater_receives_hypothesis_and_diverges_beliefs() -> None:
    pool = _pool()
    decision = MultiTrajectoryIWMPlanner(_SharedFirstIWM()).plan(pool, _graph())
    client = _HypothesisConditionedCorrectionClient()

    execution = execute_shared_trajectory_action(
        pool,
        decision,
        _graph(),
        belief_updater=GPTOSSRealEvidenceBeliefUpdater(client),
    )

    assert [row["trajectory_hypothesis"] for row in client.payloads] == [
        row.hypothesis for row in pool.trajectories
    ]
    assert execution.pool.trajectories[0].belief.missing_roles == ("after_event",)
    assert execution.pool.trajectories[1].belief.missing_roles == (
        "identity",
        "after_event",
    )


def test_structural_consolidation_merges_only_exact_equivalents() -> None:
    belief = CursorBeliefState("belief:0", "question")
    first = ReasoningTrajectory("trajectory:a", "same hypothesis", belief)
    duplicate = replace(first, trajectory_id="trajectory:b")
    alternative = ReasoningTrajectory("trajectory:c", "different hypothesis", belief)
    consolidated = consolidate_trajectory_pool(
        TrajectoryPool("pool:test", (first, duplicate, alternative))
    )

    assert consolidated.trajectories[0].status is TrajectoryStatus.ACTIVE
    assert consolidated.trajectories[1].status is TrajectoryStatus.MERGED
    assert consolidated.trajectories[1].merged_into == "trajectory:a"
    assert consolidated.trajectories[2].status is TrajectoryStatus.ACTIVE
    assert len(consolidated.trajectories) == 3
    assert consolidated.top_k_applied is False


def test_grounded_branching_preserves_every_new_interpretation_without_top_k() -> None:
    pool = _pool()
    parent = pool.trajectories[0]
    branched = branch_trajectory_pool(
        pool,
        parent.trajectory_id,
        ("the event happened before entry", "the event happened after entry"),
    )

    assert len(branched.trajectories) == 4
    assert branched.top_k_applied is False
    children = branched.trajectories[2:]
    assert all(row.parent_trajectory_id == parent.trajectory_id for row in children)
    assert all(row.belief.acquired_evidence == () for row in children)


class _RecordingMultiTrajectoryClient:
    model = "test-direct-multi-trajectory-iwm"

    def __init__(self):
        self.payload = None

    def complete_json(self, *, task, payload):
        del task
        self.payload = payload
        candidates = payload["candidate_expansions"]
        preferred = [
            alias
            for alias, row in candidates.items()
            if row["target_semantic_key"] == "person enters room"
        ]
        return {
            "status": "tie",
            "preferred": preferred,
            "trajectory_lifecycle": {
                alias: {"status": "active", "rationale": "preserve hypothesis"}
                for alias in payload["trajectory_pool"]
            },
            "rationale": "the shared read distinguishes both hypotheses",
        }


def test_direct_iwm_receives_joint_temporal_semantic_correlation_belief_context() -> (
    None
):
    client = _RecordingMultiTrajectoryClient()
    decision = MultiTrajectoryIWMPlanner(GPTOSSMultiTrajectoryIWM(client)).plan(
        _pool(), _graph()
    )

    assert decision.selected_action.target_id == "l1:first"
    assert client.payload is not None
    assert len(client.payload["trajectory_pool"]) == 2
    assert len(client.payload["candidate_expansions"]) == 8
    first_candidate = next(
        row
        for row in client.payload["candidate_expansions"].values()
        if row["target_semantic_key"] == "person enters room"
    )
    assert first_candidate["target_time_span"] == {"start_s": 0.0, "end_s": 1.0}
    assert (
        "temporal_semantic_correlation_and_belief_context_present"
        in (client.payload["required_contract"])
    )
    assert client.payload["required_contract"]["no_top_k_or_beam"] is True


def test_imagined_lifecycle_cannot_terminate_real_trajectory() -> None:
    with pytest.raises(ValueError, match="imagined lifecycle"):
        PredictedLifecycle("trajectory:a", TrajectoryStatus.CONTRADICTED)


class _SequentialSharedIWM:
    model_name = "sequential-shared-test-iwm"

    def prefer(self, pool, expansions, graph):
        del graph
        target = (
            "l1:first"
            if not pool.expandable[0].belief.acquired_evidence
            else "l1:second"
        )
        preferred = tuple(
            row.expansion_id for row in expansions if row.action.target_id == target
        )
        return MultiTrajectoryIWMDecision(
            status=(
                PoolPreferenceStatus.UNIQUE
                if len(preferred) == 1
                else PoolPreferenceStatus.TIE
            ),
            preferred_expansion_ids=preferred,
        )


def test_closed_loop_tracks_alternatives_across_multiple_shared_reads() -> None:
    trace = run_multi_trajectory_closed_loop(
        _pool(),
        _graph(),
        MultiTrajectoryIWMPlanner(_SequentialSharedIWM()),
        max_decisions=3,
        assessor=_HypothesisAssessor(),
    )

    assert trace.termination == "read_budget_exhausted"
    assert [row.observation_id for row in trace.steps] == ["l1:first", "l1:second"]
    assert trace.final_pool.trajectories[0].belief.acquired_evidence == (
        "l1:first",
        "l1:second",
    )
    assert trace.final_pool.trajectories[1].status is TrajectoryStatus.CONTRADICTED


class _RecordingRealAssessmentClient:
    model = "test-real-trajectory-assessor"

    def __init__(self):
        self.calls = 0
        self.payload = None

    def complete_json(self, *, task, payload):
        del task
        self.calls += 1
        self.payload = payload
        return {
            "assessments": {
                alias: {
                    "effect": (
                        "support"
                        if row["hypothesis"].startswith("the entrant")
                        else "counterevidence"
                    ),
                    "rationale": "grounded shared observation",
                }
                for alias, row in payload["competing_trajectories"].items()
            }
        }


def test_real_observation_assessor_updates_all_hypotheses_in_one_call() -> None:
    client = _RecordingRealAssessmentClient()
    pool = _pool()
    planner = MultiTrajectoryIWMPlanner(_SharedFirstIWM())
    execution = execute_shared_trajectory_action(
        pool,
        planner.plan(pool, _graph()),
        _graph(),
        assessor=GPTOSSRealTrajectoryEvidenceAssessor(client),
    )

    assert client.calls == 1
    assert client.payload is not None
    assert len(client.payload["competing_trajectories"]) == 2
    assert [row.effect for row in execution.assessments] == [
        EvidenceEffect.SUPPORT,
        EvidenceEffect.COUNTEREVIDENCE,
    ]
    assert [row.status for row in execution.pool.trajectories] == [
        TrajectoryStatus.SUPPORTED,
        TrajectoryStatus.CONTRADICTED,
    ]


def test_cursor_only_backtrack_does_not_change_other_trajectory_lifecycle() -> None:
    base = CursorBeliefState(
        "belief:a",
        "question",
        current_node_id="l1:second",
        acquired_evidence=("l1:first", "l1:second"),
        remaining_reads=1,
    )
    first = ReasoningTrajectory(
        "trajectory:a", "hypothesis a", base, status=TrajectoryStatus.SUPPORTED
    )
    second = ReasoningTrajectory(
        "trajectory:b",
        "hypothesis b",
        replace(base, belief_id="belief:b", current_node_id="l1:first"),
        status=TrajectoryStatus.INCONCLUSIVE,
    )
    pool = TrajectoryPool("pool:backtrack", (first, second))
    action = next(
        row
        for row in GraphActionCompiler().compile(first.belief, _graph())
        if row.kind is ActionKind.BACKTRACK
    )
    expansion = TrajectoryExpansion("expansion:backtrack", first.trajectory_id, action)
    decision = MultiTrajectoryPlanDecision(
        selected_action=action,
        planning_status="test_backtrack",
        expansions=(expansion,),
        preferred_expansion_ids=(expansion.expansion_id,),
        lifecycle_predictions=(),
        legal_expansion_count=1,
    )
    execution = execute_shared_trajectory_action(pool, decision, _graph())

    assert execution.observation is None
    assert execution.assessments == ()
    assert execution.pool.trajectories[0].belief.current_node_id == "l1:first"
    assert execution.pool.trajectories[0].status is TrajectoryStatus.SUPPORTED
    assert execution.pool.trajectories[1] == second
