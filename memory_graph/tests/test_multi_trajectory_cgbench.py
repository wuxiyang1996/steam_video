from __future__ import annotations

from memory_graph.types import MemoryNode, TimeSpan
from steam_video_new.implicit_world_model.full_graph_iwm import (
    CursorBeliefState,
    RetainedEvidenceGraph,
    TemporalNavigationEdge,
)
from steam_video_new.implicit_world_model.full_graph_iwm.closed_loop import (
    ClueInterval,
    action_divergence,
)
from steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory import (
    MultiTrajectoryIWMDecision,
    MultiTrajectoryIWMPlanner,
    PoolPreferenceStatus,
)
from steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory_cgbench import (
    GPTOSSEvidenceSufficiencyEvaluator,
    GPTOSSMultiTrajectoryAnswerSelector,
    MULTI_ARMS,
    _aggregate,
    _planner_for_arm,
    run_multi_trajectory_case,
)
from steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory_rollout import (
    GPTOSSCategoricalMultiTrajectoryModel,
    MultiTrajectoryRolloutPlanner,
    ReactiveMultiTrajectoryPlanner,
    ShuffledHypothesisWorldModel,
)
from steam_video_new.implicit_world_model.full_graph_iwm.pilot_analysis import (
    _transition_over_crediting_candidates,
)
from steam_video_new.implicit_world_model.full_graph_iwm.localization import (
    GPTOSSEntryLocalizer,
)


def _node(node_id: str, start: float, text: str) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        video_id="video:cgbench-multi",
        time_span=TimeSpan(start, start + 1.0),
        provenance={"producer": "test"},
        node_type="observation",
        text=text,
        metadata={"predicate": text, "layer": "L1"},
    )


def test_action_divergence_reads_multi_trajectory_step_schema() -> None:
    reference = {
        "arm": "world_model_guided",
        "steps": [
            {
                "observation_id": "l1:first",
                "decision": {"selected_action": {"action_id": "read:first"}},
            },
            {
                "observation_id": "l1:second",
                "decision": {"selected_action": {"action_id": "read:second"}},
            },
        ],
    }
    candidate = {
        "arm": "immediate_effect_only",
        "steps": [
            {
                "observation_id": "l1:first",
                "decision": {"selected_action": {"action_id": "read:first"}},
            },
            {
                "observation_id": "l1:first",
                "decision": {"selected_action": {"action_id": "read:first"}},
            },
        ],
    }

    audit = action_divergence(reference, candidate)

    assert audit["action_diverged"] is True
    assert audit["first_divergence_step"] == 1


def _graph() -> RetainedEvidenceGraph:
    first = _node("l1:first", 0.0, "woman enters a box")
    second = _node("l1:second", 1.0, "a cat attacks the woman")
    return RetainedEvidenceGraph(
        graph_id="graph:cgbench-multi",
        nodes=(first, second),
        temporal_edges=(
            TemporalNavigationEdge(
                "edge:first-second", first.node_id, second.node_id, "temporal_next"
            ),
        ),
        correlation_edges=(),
        capacity=2,
    )


class _SequentialIWM:
    model_name = "sequential-test-iwm"

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
            status=PoolPreferenceStatus.TIE,
            preferred_expansion_ids=preferred,
        )


class _AnswerClient:
    model = "answer-schema-test"

    def __init__(self):
        self.payload = None

    def complete_json(self, *, task, payload):
        del task
        self.payload = payload
        selected = next(
            alias
            for alias, choice in payload["choices"].items()
            if choice == "attacked by a cat"
        )
        return {
            "status": "select",
            "choice": selected,
            "rationale": "the acquired observation directly shows the cat attack",
        }


def test_cgbench_case_uses_public_choices_and_joins_hidden_labels_after_run() -> None:
    client = _AnswerClient()
    result = run_multi_trajectory_case(
        case_id="case:test",
        question="What happened after she entered the box?",
        choices=("visited by a squirrel", "attacked by a cat"),
        graph=_graph(),
        clue_intervals=(ClueInterval(0.0, 1.0), ClueInterval(1.0, 2.0)),
        planner=MultiTrajectoryIWMPlanner(_SequentialIWM()),
        answer_selector=GPTOSSMultiTrajectoryAnswerSelector(client),
        read_budget=2,
        arm="world_model_guided",
        initial_entry_node_ids=("l1:first", "l1:second"),
        evaluator_answer="attacked by a cat",
    )

    assert result["hypothesis_source"] == "public_answer_choices"
    assert result["schema_version"] == "steam-multi-trajectory-closed-loop-run/v0.2"
    assert result["runtime_contract_version"] == (
        "steam-multi-trajectory-runtime/v1.0"
    )
    assert result["initial_hypothesis_count"] == 2
    assert result["metrics"]["answer_correct"] is True
    assert result["metrics"]["clue_recall"] == 1.0
    assert result["metrics"]["correct_hypothesis_survived"] is True
    assert result["hidden_evaluator_feedback_to_planner"] is False
    assert "attacked by a cat" in client.payload["choices"].values()
    assert "evaluator_answer" not in str(client.payload)
    evidence = client.payload["acquired_real_evidence"]
    assert evidence[0]["time_span"] == {"start_s": 0.0, "end_s": 1.0}
    assert evidence[0]["structured_observation"] == {}
    assert evidence[0]["provenance"]["video_id"] == "video:cgbench-multi"


def test_terminal_answer_selector_rejects_numeric_model_output() -> None:
    class _NumericClient:
        model = "invalid-numeric-model"

        def complete_json(self, *, task, payload):
            del task, payload
            return {"status": "abstain", "choice": None, "rationale": 0.7}

    belief = CursorBeliefState(
        "belief:test",
        "question",
        current_node_id="l1:first",
        acquired_evidence=("l1:first",),
    )
    from steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory import (
        initialize_trajectory_pool,
    )

    pool = initialize_trajectory_pool(belief, ("left", "right"))
    selector = GPTOSSMultiTrajectoryAnswerSelector(_NumericClient())

    import pytest

    with pytest.raises(ValueError, match="numeric"):
        selector.select(pool, ("left", "right"), _graph())


def test_terminal_answer_selector_cannot_guess_without_real_evidence() -> None:
    client = _AnswerClient()
    from steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory import (
        initialize_trajectory_pool,
    )

    pool = initialize_trajectory_pool(
        CursorBeliefState("belief:empty", "question"),
        ("left", "right"),
    )
    decision = GPTOSSMultiTrajectoryAnswerSelector(client).select(
        pool, ("left", "right"), _graph()
    )

    assert decision.status == "abstain"
    assert decision.selected_choice is None
    assert client.payload is None


def test_evidence_sufficiency_is_categorical_and_evaluator_only() -> None:
    class _SufficiencyClient:
        model = "sufficiency-test"

        def __init__(self):
            self.payload = None

        def complete_json(self, *, task, payload):
            del task
            self.payload = payload
            return {
                "status": "supports_reference",
                "rationale": "the observed attack distinguishes the answer",
            }

    client = _SufficiencyClient()
    decision = GPTOSSEvidenceSufficiencyEvaluator(client).evaluate(
        question="What happened?",
        choices=("visited", "attacked"),
        evaluator_answer="attacked",
        observations=(_graph().node_by_id["l1:second"],),
    )

    assert decision.status == "supports_reference"
    assert decision.evaluator_only is True
    assert decision.fed_back_to_planner is False
    assert client.payload["contract"]["not_available_to_planner"] is True


def test_inconclusive_entry_localization_preserves_multiple_paths() -> None:
    class _InconclusiveEntryClient:
        model = "inconclusive-entry-test"

        def complete_json(self, *, task, payload):
            del task
            assert payload["required_contract"][
                "one_representative_cursor_per_missing_role_or_event_sequence"
            ] is True
            assert payload["required_contract"][
                "do_not_enumerate_temporal_repetitions_of_the_same_event"
            ] is True
            aliases = list(payload["candidate_addresses"])
            return {
                "status": "inconclusive",
                "preferred": aliases,
                "rationale": "both grounded entries remain plausible",
            }

    selected = GPTOSSEntryLocalizer(_InconclusiveEntryClient()).localize(
        question="what happened?",
        missing_roles=("event",),
        graph=_graph(),
    )

    assert selected == ("l1:first", "l1:second")


def test_role_conditioned_entry_localization_flattens_without_ranking() -> None:
    class _RoleConditionedEntryClient:
        model = "role-conditioned-entry-test"

        def complete_json(self, *, task, payload):
            del task, payload
            return {
                "status": "located",
                "preferred": {
                    "event": "anchor_b",
                    "actor_identity": ["anchor_a", "anchor_b"],
                },
                "rationale": "one cursor can ground multiple roles",
            }

    localizer = GPTOSSEntryLocalizer(_RoleConditionedEntryClient())
    selected = localizer.localize(
        question="who did what?",
        missing_roles=("actor identity", "event"),
        graph=_graph(),
    )

    assert selected == ("l1:first", "l1:second")
    assert localizer.audits[0]["role_conditioned_output"] is True
    assert localizer.audits[0]["role_key_normalization_count"] == 1
    assert localizer.audits[0]["top_k_applied"] is False


def test_five_matched_arms_have_distinct_world_model_interventions() -> None:
    model = GPTOSSCategoricalMultiTrajectoryModel(_AnswerClient())

    normal = _planner_for_arm("world_model_guided", model, max_complete_pairs=32)
    immediate = _planner_for_arm("immediate_effect_only", model, max_complete_pairs=32)
    shuffled = _planner_for_arm(
        "shuffled_world_model_prediction", model, max_complete_pairs=32
    )
    reactive = _planner_for_arm("no_world_model", model, max_complete_pairs=32)

    assert MULTI_ARMS == (
        "world_model_guided",
        "no_world_model",
        "shuffled_world_model_prediction",
        "immediate_effect_only",
        "oracle_clue_ceiling",
    )
    assert isinstance(normal, MultiTrajectoryRolloutPlanner) and normal.horizon == 2
    assert (
        isinstance(immediate, MultiTrajectoryRolloutPlanner) and immediate.horizon == 1
    )
    assert isinstance(shuffled.world_model, ShuffledHypothesisWorldModel)
    assert isinstance(reactive, ReactiveMultiTrajectoryPlanner)


def test_matched_metrics_remain_separate_without_aggregate_reward() -> None:
    report = _aggregate(
        [
            {
                "arm": "world_model_guided",
                "metrics": {
                    "answer_correct": True,
                    "clue_recall": 0.5,
                    "real_read_count": 2,
                    "read_efficiency": 0.5,
                    "answer_abstained": False,
                    "delayed_reasoning_success": False,
                    "correct_hypothesis_survived": True,
                    "false_correct_hypothesis_elimination": False,
                    "complete_comparison_budget_failure": False,
                },
            }
        ],
        ("world_model_guided",),
    )

    metrics = report["world_model_guided"]
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["mean_clue_recall"] == 0.5
    assert not ({"reward", "score", "utility", "passed"} & set(metrics))


def test_transition_over_crediting_uses_corrected_belief_without_hidden_gt(
    tmp_path,
) -> None:
    action = {
        "kind": "start_at",
        "source_id": None,
        "target_id": "l1:first",
        "edge_id": None,
    }
    belief = {
        "missing_roles": ["event"],
        "contradictions": [],
        "answerability": "not_ready",
    }
    runs = [
        {
            "case_id": "case:test",
            "steps": [
                {
                    "observation_id": "l1:first",
                    "pool_before": {
                        "trajectories": [
                            {"trajectory_id": "trajectory:a", "belief": belief}
                        ]
                    },
                    "pool_after": {
                        "trajectories": [
                            {"trajectory_id": "trajectory:a", "belief": belief}
                        ]
                    },
                    "decision": {
                        "selected_action": action,
                        "imagined_paths": [
                            {
                                "transitions": [{"action": action}],
                                "conditioned_outcomes": [
                                    {
                                        "trajectory_id": "trajectory:a",
                                        "transitions": [
                                            {
                                                "belief_delta": {
                                                    "progress": "advanced",
                                                    "answerability_after": "ready",
                                                    "resolved_roles": ["event"],
                                                }
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    },
                }
            ],
        }
    ]

    records = _transition_over_crediting_candidates(runs, tmp_path)

    assert len(records) == 1
    assert records[0]["provisional_label"] == "over_crediting"
    assert records[0]["unsupported_predicted_resolved_roles"] == ["event"]
    assert records[0]["hidden_evaluator_labels_included"] is False
    assert records[0]["training_allowed"] is False
