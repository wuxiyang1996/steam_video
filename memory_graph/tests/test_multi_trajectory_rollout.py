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
    LegalGraphAction,
    MultiTrajectoryRolloutPlanner,
    PredictedObservation,
    PreferenceLabel,
    ProgressChange,
    ReactiveMultiTrajectoryPlanner,
    RetainedEvidenceGraph,
    ShuffledHypothesisWorldModel,
    TemporalNavigationEdge,
    audit_permutation_invariance,
    compile_hypothesis_requests,
    execute_shared_trajectory_action,
    initialize_trajectory_pool,
)
from steam_video_new.implicit_world_model.full_graph_iwm.planner import (
    project_imagined_belief,
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


def test_imagined_reread_preserves_real_grounding() -> None:
    belief = CursorBeliefState(
        belief_id="belief:grounded",
        question="What happened?",
        current_node_id="l1:first",
        acquired_evidence=("l1:first",),
        required_roles=("event",),
        grounded_role_evidence=(("event", "l1:first"),),
        remaining_reads=1,
    )
    action = LegalGraphAction(
        action_id="read:first-again",
        kind=ActionKind.START_AT,
        target_id="l1:first",
        reads_evidence=True,
    )
    projected = project_imagined_belief(
        belief,
        ImaginedTransition(
            action,
            PredictedObservation("l1:first", EvidenceOutcome.SUPPORT),
            CategoricalBeliefDelta(
                progress=ProgressChange.UNCHANGED,
                answerability_after=AnswerabilityState.NOT_READY,
            ),
        ),
    )

    assert projected.grounded_role_evidence == (("event", "l1:first"),)
    assert projected.imagined_evidence == ()


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
                        else request.belief.answerability
                        if not request.action.reads_evidence
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
            return path.first_action.target_id == "l1:first" and any(
                len(outcome.transitions) == 2
                and outcome.transitions[1].action.target_id == "l1:second"
                for outcome in path.conditioned_outcomes
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
    assert len(decision.imagined_paths) == 2
    assert len(preference.last_pairs) == 1
    assert all(
        {outcome.trajectory_id for outcome in path.conditioned_outcomes}
        == {row.trajectory_id for row in _pool().expandable}
        for path in decision.imagined_paths
    )
    selected_tree = next(
        row
        for row in decision.imagined_paths
        if row.first_action.target_id == "l1:first"
        and any(
            len(outcome.transitions) == 2
            and outcome.transitions[1].action.target_id == "l1:second"
            for outcome in row.conditioned_outcomes
        )
    )
    assert any(
        len(outcome.transitions) == 2
        and outcome.transitions[1].action.target_id == "l1:second"
        for outcome in selected_tree.conditioned_outcomes
    )
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
        if "shared_action_groups" in payload:
            return {
                "predictions": {
                    alias: {
                        "outcome": "support",
                        "progress": "advanced",
                        "answerability_after": "not_ready",
                        "frontier_change": "opened",
                        "contradiction_change": "unchanged",
                        "resolved_roles": "none",
                        "opened_roles": "none",
                        "relation_updates": "event_frequency_counting",
                        "rationale": "grounded target may advance the hypothesis",
                    }
                    for group in payload["shared_action_groups"].values()
                    for alias in group["conditioned_requests"]
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
    assert transition_audits[-1]["item_count"] == 4
    assert transition_audits[-1]["model_predicted_item_count"] == 4
    assert transition_audits[-1]["neutral_no_read_item_count"] == 0
    assert transition_audits[-1]["shared_action_group_count"] == 2
    assert transition_audits[-1]["batch_count"] == 1
    assert comparison_audits[-1]["item_count"] == 1
    assert comparison_audits[-1]["batch_count"] == 1
    transition_payloads = [
        row for row in client.payloads if "shared_action_groups" in row
    ]
    assert {
        request["hypothesis"]
        for payload in transition_payloads
        for group in payload["shared_action_groups"].values()
        for request in group["conditioned_requests"].values()
    } == {"the entrant opened it", "another person opened it"}
    assert all(
        len(group["conditioned_requests"]) == 2
        for payload in transition_payloads
        for group in payload["shared_action_groups"].values()
    )
    assert all(
        alias.startswith("prediction_")
        for payload in transition_payloads
        for group in payload["shared_action_groups"].values()
        for alias in group["conditioned_requests"]
    )
    assert all(row["top_k_applied"] is False for row in model.transport_audits)


class _OrderedSequenceSchemaClient(_SchemaClient):
    def complete_json(self, *, task, payload):
        del task
        self.payloads.append(payload)
        if "shared_action_groups" in payload:
            return {
                "predictions": [
                    {
                        "action": alias,
                        "outcome": "support",
                        "progress": "advanced",
                        "answerability_after": "not_ready",
                        "frontier_change": "opened",
                        "contradiction_change": "unchanged",
                        "resolved_roles": [],
                        "opened_roles": [],
                        "relation_updates": [],
                        "rationale": "ordered complete transition",
                    }
                    for group in payload["shared_action_groups"].values()
                    for alias in group["conditioned_requests"]
                ]
            }
        return {
            "comparisons": [
                {
                    "comparison": alias,
                    "label": "tie",
                    "rationale": "ordered complete comparison",
                }
                for alias in payload["comparisons"]
            ]
        }


def test_gpt_adapter_strictly_normalizes_complete_ordered_sequences() -> None:
    model = GPTOSSCategoricalMultiTrajectoryModel(
        _OrderedSequenceSchemaClient(),
        transition_batch_size=3,
        comparison_batch_size=4,
    )

    decision = MultiTrajectoryRolloutPlanner(model, model, horizon=1).plan(
        _pool(), _graph()
    )

    assert decision.selected_action.kind is ActionKind.ABSTAIN
    transition_audit, comparison_audit = model.transport_audits
    assert transition_audit["ordered_sequence_normalization_count"] == 1
    assert comparison_audit["ordered_sequence_normalization_count"] == 1
    assert transition_audit["complete_coverage"] is True
    assert comparison_audit["complete_coverage"] is True


class _SplitRequiredClient(_SchemaClient):
    def complete_json(self, *, task, payload):
        if (
            "shared_action_groups" in payload
            and len(payload["shared_action_groups"]) > 1
        ):
            self.payloads.append(payload)
            raise ValueError("simulated transport truncation; finish_reason=length")
        return super().complete_json(task=task, payload=payload)


def test_transition_transport_adaptively_splits_without_pruning() -> None:
    client = _SplitRequiredClient()
    model = GPTOSSCategoricalMultiTrajectoryModel(
        client,
        transition_batch_size=3,
    )

    decision = MultiTrajectoryRolloutPlanner(model, model, horizon=1).plan(
        _pool(), _graph()
    )

    assert decision.preference_audit["complete_coverage"] is True
    audit = next(
        row
        for row in model.transport_audits
        if row["operation"] == "categorical_transition"
    )
    assert audit["adaptive_transport_split_count"] == 1
    assert audit["complete_coverage"] is True
    assert audit["top_k_applied"] is False
    oversized_calls = [
        payload
        for payload in client.payloads
        if len(payload.get("shared_action_groups") or {}) > 1
    ]
    assert len(oversized_calls) == 1


class _GPT5MiniTransportClient(_SchemaClient):
    def complete_json(self, *, task, payload):
        del task
        self.payloads.append(payload)
        if "shared_action_groups" in payload:
            alias_fields = ("alias", "outcome_id", "outcome_alias")
            aliases = [
                alias
                for group in payload["shared_action_groups"].values()
                for alias in group["conditioned_requests"]
            ]
            return {
                "predictions": [
                    {
                        alias_fields[index % len(alias_fields)]: alias,
                        "outcome": "inconclusive",
                        "progress": "unchanged",
                        "answerability_after": "not_ready",
                        "frontier_change": "unchanged",
                        "contradiction_change": "unchanged",
                        "resolved_roles": [],
                        "opened_roles": [],
                        "relation_updates": "unchanged",
                        "rationale": "complete categorical transport row",
                    }
                    for index, alias in enumerate(aliases)
                ]
            }
        if "candidate_chains" in payload:
            return {
                "status": "incomparable",
                "preferred": list(payload["candidate_chains"]),
                "rationale": "all chains remain on the undominated frontier",
            }
        if "surviving_reasoning_paths" in payload:
            return {
                "status": "select",
                "selected": next(iter(payload["surviving_reasoning_paths"])),
                "rationale": "inspect one grounded address to discriminate paths",
            }
        raise AssertionError("unexpected transport payload")


def test_gpt5mini_transport_aliases_preserve_incomparable_frontier() -> None:
    client = _GPT5MiniTransportClient()
    model = GPTOSSCategoricalMultiTrajectoryModel(
        client,
        transition_batch_size=3,
    )

    decision = MultiTrajectoryRolloutPlanner(
        model,
        model,
        horizon=1,
        setwise_preference_model=model,
        evidence_scheduler=model,
    ).plan(_pool(), _graph())

    assert decision.selected_action.reads_evidence is True
    assert set(decision.preferred_path_ids) == {
        path.path_id for path in decision.imagined_paths
    }
    assert decision.planning_status == "rollout_scheduled_discriminating_first_hop"
    assert decision.preference_audit["complete_coverage"] is True
    assert decision.preference_audit["evidence_scheduler_used"] is True
    assert decision.preference_audit["surviving_trajectory_paths_preserved"] is True
    setwise_payload = next(
        payload for payload in client.payloads if "candidate_chains" in payload
    )
    assert all(
        "hypothesis_conditioned_outcome_classes" in row
        for row in setwise_payload["candidate_chains"].values()
    )
    represented_members = sum(
        len(group["conditioned_members"])
        for row in setwise_payload["candidate_chains"].values()
        for group in row["hypothesis_conditioned_outcome_classes"].values()
    )
    assert represented_members == sum(
        len(path.conditioned_outcomes) for path in decision.imagined_paths
    )


def test_no_read_controls_bypass_model_and_remain_neutral() -> None:
    client = _GPT5MiniTransportClient()
    model = GPTOSSCategoricalMultiTrajectoryModel(client)
    requests = compile_hypothesis_requests(_pool(), _graph())

    predictions = model.predict_batch(requests, _graph())

    controls = [
        prediction
        for request, prediction in zip(requests, predictions)
        if not request.action.reads_evidence
    ]
    assert controls
    assert all(
        row.observation.outcome is EvidenceOutcome.INCONCLUSIVE for row in controls
    )
    assert all(
        row.belief_delta.progress is ProgressChange.UNCHANGED for row in controls
    )
    assert all(
        row.belief_delta.answerability_after is AnswerabilityState.NOT_READY
        for row in controls
    )
    assert all(row.belief_delta.resolved_roles == () for row in controls)
    audit = model.transport_audits[-1]
    assert audit["neutral_no_read_item_count"] == len(controls)
    sent_aliases = [
        alias
        for payload in client.payloads
        if "shared_action_groups" in payload
        for group in payload["shared_action_groups"].values()
        for alias in group["conditioned_requests"]
    ]
    assert len(sent_aliases) == len(requests) - len(controls)


def test_shuffled_intervention_preserves_no_read_protocol_invariants() -> None:
    requests = compile_hypothesis_requests(_pool(), _graph())
    shuffled_model = ShuffledHypothesisWorldModel(_CategoricalWorldModel())

    predictions = shuffled_model.predict_batch(requests, _graph())

    controls = [
        prediction
        for request, prediction in zip(requests, predictions)
        if not request.action.reads_evidence
    ]
    assert controls
    assert all(
        row.observation.outcome is EvidenceOutcome.INCONCLUSIVE for row in controls
    )
    assert all(
        row.belief_delta.progress is ProgressChange.UNCHANGED for row in controls
    )
    assert shuffled_model.audits[-1]["preserved_no_read_count"] == len(controls)


class _UnknownRoleRepairClient(_SchemaClient):
    def complete_json(self, *, task, payload):
        self.payloads.append(payload)
        if "shared_action_groups" in payload:
            repaired = "copied verbatim" in task
            return {
                "predictions": {
                    alias: {
                        "outcome": "inconclusive",
                        "progress": "unchanged",
                        "answerability_after": "not_ready",
                        "frontier_change": "unchanged",
                        "contradiction_change": "unchanged",
                        "resolved_roles": (
                            [] if repaired else ["rephrased_role_not_present_in_belief"]
                        ),
                        "opened_roles": [],
                        "relation_updates": [],
                        "rationale": "repair exact role membership",
                    }
                    for group in payload["shared_action_groups"].values()
                    for alias in group["conditioned_requests"]
                }
            }
        if "candidate_chains" in payload:
            preferred = next(iter(payload["candidate_chains"]))
            return {
                "status": "unique",
                "preferred": [preferred],
                "rationale": "one chain remains",
            }
        raise AssertionError("unexpected repair payload")


def test_transition_repair_forbids_invented_resolved_roles() -> None:
    client = _UnknownRoleRepairClient()
    model = GPTOSSCategoricalMultiTrajectoryModel(client)

    decision = MultiTrajectoryRolloutPlanner(
        model,
        model,
        horizon=1,
        setwise_preference_model=model,
    ).plan(_pool(), _graph())

    assert decision.preference_audit["complete_coverage"] is True
    assert any(
        payload["required_output"].get("resolved_roles")
        for payload in client.payloads
        if "shared_action_groups" in payload
    )


class _DirectAliasMapClient(_GPT5MiniTransportClient):
    def complete_json(self, *, task, payload):
        result = super().complete_json(task=task, payload=payload)
        if "shared_action_groups" in payload:
            rows = {}
            for raw in result["predictions"]:
                row = dict(raw)
                alias = next(
                    row.pop(field)
                    for field in ("alias", "outcome_id", "outcome_alias")
                    if field in row
                )
                rows[alias] = row
            return rows
        return result


def test_transition_accepts_exact_top_level_alias_map() -> None:
    model = GPTOSSCategoricalMultiTrajectoryModel(_DirectAliasMapClient())

    decision = MultiTrajectoryRolloutPlanner(
        model,
        model,
        horizon=1,
        setwise_preference_model=model,
    ).plan(_pool(), _graph())

    assert decision.preference_audit["complete_coverage"] is True


class _SetwiseSchemaClient(_SchemaClient):
    def complete_json(self, *, task, payload):
        if "candidate_chains" in payload:
            self.payloads.append(payload)
            preferred = next(iter(payload["candidate_chains"]))
            return {
                "status": "unique",
                "preferred": [preferred],
                "rationale": "one complete chain is preferred",
            }
        return super().complete_json(task=task, payload=payload)


def test_setwise_frontier_covers_all_joint_chains_without_pair_calls() -> None:
    client = _SetwiseSchemaClient()
    model = GPTOSSCategoricalMultiTrajectoryModel(client)

    decision = MultiTrajectoryRolloutPlanner(
        model,
        model,
        horizon=1,
        setwise_preference_model=model,
    ).plan(_pool(), _graph())

    assert decision.preference_audit["preference_mode"] == "setwise_full_frontier"
    assert decision.preference_audit["comparison_count"] == 1
    assert decision.preference_audit["complete_coverage"] is True
    setwise_audit = next(
        row
        for row in model.transport_audits
        if row["operation"] == "categorical_setwise_frontier"
    )
    assert setwise_audit["item_count"] == len(decision.imagined_paths)
    assert setwise_audit["top_k_applied"] is False
    assert not any(
        row["operation"] == "categorical_path_preference"
        for row in model.transport_audits
    )
