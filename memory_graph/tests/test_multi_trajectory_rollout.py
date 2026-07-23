from __future__ import annotations

from memory_graph.types import MemoryNode, TimeSpan
from steam_video_new.implicit_world_model.full_graph_iwm import (
    ActionKind,
    AnswerabilityState,
    CategoricalBeliefDelta,
    CursorBeliefState,
    EvidenceOutcome,
    GPTOSSCategoricalMultiTrajectoryModel,
    HypothesisPathPreference,
    ImaginedTransition,
    MultiTrajectoryRolloutPlanner,
    PredictedObservation,
    PreferenceLabel,
    ProgressChange,
    ReactiveMultiTrajectoryPlanner,
    RetainedEvidenceGraph,
    TemporalNavigationEdge,
    audit_permutation_invariance,
    execute_shared_trajectory_action,
    initialize_trajectory_pool,
)


def _node(node_id: str, start: float, text: str) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        video_id="video:rollout",
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
        graph_id="graph:rollout",
        nodes=(first, second),
        temporal_edges=(
            TemporalNavigationEdge(
                "edge:first-second", first.node_id, second.node_id, "temporal_next"
            ),
        ),
        correlation_edges=(),
        capacity=2,
    )


def _pool():
    return initialize_trajectory_pool(
        CursorBeliefState(
            belief_id="belief:rollout",
            question="Who opened the door after entering?",
            localized_entry_node_ids=("l1:first", "l1:second"),
            required_roles=("identity", "after_event"),
            missing_roles=("identity", "after_event"),
            remaining_reads=2,
        ),
        ("the entrant opened it", "another person opened it"),
    )


class _CategoricalWorldModel:
    model_name = "test-categorical-world-model"

    def predict_batch(self, requests, graph):
        del graph
        return tuple(
            ImaginedTransition(
                request.action,
                PredictedObservation(
                    request.action.target_id,
                    (
                        EvidenceOutcome.SUPPORT
                        if request.action.reads_evidence
                        else EvidenceOutcome.INCONCLUSIVE
                    ),
                ),
                CategoricalBeliefDelta(
                    progress=(
                        ProgressChange.ADVANCED
                        if request.action.reads_evidence
                        else ProgressChange.UNCHANGED
                    ),
                    answerability_after=(
                        AnswerabilityState.READY
                        if request.action.target_id == "l1:second"
                        else AnswerabilityState.NOT_READY
                    ),
                ),
            )
            for request in requests
        )


class _DelayedPreference:
    model_name = "test-delayed-preference"

    def __init__(self):
        self.last_pairs = ()
        self.last_include = None

    def compare_batch(self, pool, pairs, graph, *, include_imagined_transitions):
        del pool, graph
        self.last_pairs = tuple(pairs)
        self.last_include = include_imagined_transitions

        def desirable(path):
            return (
                path.first_action.target_id == "l1:first"
                and len(path.transitions) == 2
                and path.transitions[1].action.target_id == "l1:second"
            )

        rows = []
        for pair in pairs:
            left = desirable(pair.left)
            right = desirable(pair.right)
            label = (
                PreferenceLabel.PREFER_LEFT
                if left and not right
                else PreferenceLabel.PREFER_RIGHT
                if right and not left
                else PreferenceLabel.TIE
            )
            rows.append(
                HypothesisPathPreference(
                    pair.comparison_id,
                    pair.left.path_id,
                    pair.right.path_id,
                    label,
                )
            )
        return tuple(rows)


def test_horizon_two_selects_shared_delayed_first_hop_with_complete_coverage() -> None:
    preference = _DelayedPreference()
    planner = MultiTrajectoryRolloutPlanner(
        _CategoricalWorldModel(), preference, horizon=2
    )
    decision = planner.plan(_pool(), _graph())

    assert decision.selected_action.target_id == "l1:first"
    assert decision.planning_horizon == 2
    assert len(decision.imagined_paths) == 9
    assert len(preference.last_pairs) == 36
    assert all(
        len(path.conditioned_outcomes) == 2 for path in decision.imagined_paths
    )
    selected_tree = next(
        row
        for row in decision.imagined_paths
        if row.first_action.target_id == "l1:first"
        and len(row.transitions) == 2
        and row.transitions[1].action.target_id == "l1:second"
    )
    assert selected_tree.transitions[1].action.target_id == "l1:second"
    assert decision.preference_audit["complete_coverage"] is True
    assert decision.preference_audit["top_k_applied"] is False
    assert preference.last_include is True

    execution = execute_shared_trajectory_action(_pool(), decision, _graph())
    assert execution.observation.node_id == "l1:first"
    assert all(
        row.belief.acquired_evidence == ("l1:first",)
        for row in execution.pool.trajectories
    )


def test_immediate_only_has_no_second_hop_paths() -> None:
    planner = MultiTrajectoryRolloutPlanner(
        _CategoricalWorldModel(), _DelayedPreference(), horizon=1
    )
    decision = planner.plan(_pool(), _graph())

    assert decision.planning_horizon == 1
    assert all(len(path.transitions) == 1 for path in decision.imagined_paths)
    assert all(not path.continuations for path in decision.imagined_paths)
    assert decision.preference_audit["second_expansion_count"] == 0


def test_reactive_arm_hides_imagined_transitions() -> None:
    preference = _DelayedPreference()
    decision = ReactiveMultiTrajectoryPlanner(preference).plan(_pool(), _graph())

    assert decision.planning_horizon == 0
    assert decision.imagined_paths == ()
    assert preference.last_include is False
    assert decision.preference_audit["world_model_predictions_visible"] is False


def test_partial_order_aggregation_is_input_order_invariant() -> None:
    preference = _DelayedPreference()
    decision = MultiTrajectoryRolloutPlanner(
        _CategoricalWorldModel(), preference, horizon=2
    ).plan(_pool(), _graph())
    audit = audit_permutation_invariance(
        decision.imagined_paths,
        preference.last_pairs,
        preference.compare_batch(
            _pool(), preference.last_pairs, _graph(), include_imagined_transitions=True
        ),
    )

    assert audit["preferred_path_set_equal"] is True
    assert audit["top_k_applied"] is False


class _SchemaClient:
    model = "schema-test-model"

    def __init__(self):
        self.payloads = []

    def complete_json(self, *, task, payload):
        del task
        self.payloads.append(payload)
        if "requests" in payload:
            return {
                "predictions": {
                    alias: {
                        "outcome": "support",
                        "progress": "advanced",
                        "answerability_after": "not_ready",
                        "frontier_change": "opened",
                        "contradiction_change": "unchanged",
                        "resolved_roles": [],
                        "opened_roles": [],
                        "relation_updates": [],
                        "rationale": "grounded target may advance the hypothesis",
                    }
                    for alias in payload["requests"]
                }
            }
        return {
            "comparisons": {
                alias: {"label": "tie", "rationale": "both remain plausible"}
                for alias in payload["comparisons"]
            }
        }


def test_gpt_adapter_batches_without_pruning_and_covers_every_pair() -> None:
    client = _SchemaClient()
    model = GPTOSSCategoricalMultiTrajectoryModel(
        client,
        transition_batch_size=3,
        comparison_batch_size=7,
    )
    decision = MultiTrajectoryRolloutPlanner(model, model, horizon=1).plan(
        _pool(), _graph()
    )

    assert decision.selected_action.kind is ActionKind.ABSTAIN
    assert decision.preference_audit["complete_coverage"] is True
    transition_audits = [
        row
        for row in model.transport_audits
        if row["operation"] == "categorical_transition"
    ]
    comparison_audits = [
        row
        for row in model.transport_audits
        if row["operation"] == "categorical_path_preference"
    ]
    assert transition_audits[-1]["item_count"] == 8
    assert transition_audits[-1]["batch_count"] == 3
    assert comparison_audits[-1]["item_count"] == 6
    assert comparison_audits[-1]["batch_count"] == 1
    transition_payloads = [row for row in client.payloads if "requests" in row]
    assert {
        request["hypothesis"]
        for payload in transition_payloads
        for request in payload["requests"].values()
    } == {"the entrant opened it", "another person opened it"}
    assert all(row["top_k_applied"] is False for row in model.transport_audits)
