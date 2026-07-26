from __future__ import annotations

from memory_graph.types import MemoryNode, TimeSpan
from steam_video_new.implicit_world_model.full_graph_iwm import (
    CursorBeliefState,
    RetainedEvidenceGraph,
    TemporalNavigationEdge,
    execute_shared_trajectory_action,
    initialize_trajectory_pool,
)
from steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory_rollout import (
    MultiTrajectoryRolloutPlanner,
)
from steam_video_new.implicit_world_model.iwm_9b.runtime import (
    Structured9BMultiTrajectoryModel,
)


def _node(node_id: str, start: float, predicate: str) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        video_id="video:test",
        time_span=TimeSpan(start, start + 1.0),
        provenance={"producer": "test"},
        node_type="observation",
        text=predicate,
        metadata={"predicate": predicate, "layer": "L1"},
    )


def _graph() -> RetainedEvidenceGraph:
    first = _node("l1:first", 0.0, "person enters room")
    second = _node("l1:second", 1.0, "person opens door")
    return RetainedEvidenceGraph(
        graph_id="graph:test",
        nodes=(first, second),
        temporal_edges=(
            TemporalNavigationEdge(
                "time:first-second",
                first.node_id,
                second.node_id,
                "temporal_next",
            ),
        ),
        correlation_edges=(),
        capacity=2,
    )


def _pool():
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


class _StructuredClient:
    model = "Qwen/Qwen3.5-9B-scripted"

    def __init__(self) -> None:
        self.frontier_calls = 0
        self.planner_payloads = []

    def complete_json(self, *, task, payload):
        del task
        if "requests" in payload:
            return {
                "predictions": {
                    alias: self._transition(request)
                    for alias, request in payload["requests"].items()
                }
            }
        if "candidate_joint_chains" in payload:
            self.planner_payloads.append(payload)
            preferred = self._preferred_alias(payload["candidate_joint_chains"])
            self.frontier_calls += 1
            return {
                "status": "unique",
                "preferred": [preferred],
                "rationale": "preferred complete multi-path reasoning chain",
            }
        raise AssertionError(f"unexpected payload keys: {sorted(payload)}")

    def _transition(self, request):
        target = (request["action"].get("target") or {}).get("semantic_key")
        hypothesis = request["hypothesis"]
        if target == "person enters room":
            resolved = ["identity"] if hypothesis.startswith("the entrant") else []
            event = "candidate subject enters the room"
            supported = (
                ["entrant identity hypothesis"]
                if hypothesis.startswith("the entrant")
                else []
            )
            contradicted = (
                []
                if hypothesis.startswith("the entrant")
                else ["different-person interpretation may remain"]
            )
        elif target == "person opens door":
            resolved = ["after_event"]
            event = "a person opens the door after the entry"
            supported = ["after-event link"]
            contradicted = []
        else:
            resolved = []
            event = "no grounded target observation predicted"
            supported = []
            contradicted = []
        progress = "advanced" if resolved else "unchanged"
        return {
            "observation_patch": {
                "event_or_state": event,
                "entity_bindings": {},
                "temporal_binding": (
                    "after" if target == "person opens door" else "unknown"
                ),
                "evidence_role": "bridge_evidence" if resolved else "inconclusive",
                "alternatives": ["observation may remain inconclusive"],
            },
            "belief_patch": {
                "supported_claims": supported,
                "contradicted_claims": contradicted,
                "newly_bound_variables": resolved,
                "opened_dependencies": (
                    ["inspect the later door event"]
                    if target == "person enters room"
                    else []
                ),
                "resolved_dependencies": resolved,
            },
            "categorical_audit": {
                "observation_outcome": (
                    "bridge_evidence" if resolved else "inconclusive"
                ),
                "progress": progress,
                "answerability_after": "not_ready",
                "frontier_change": "opened" if resolved else "unchanged",
                "contradiction_change": "unchanged",
                "resolved_roles": resolved,
                "opened_roles": [],
                "relation_updates": [],
            },
        }

    def _preferred_alias(self, candidates):
        desired_first = (
            "person enters room" if self.frontier_calls == 0 else "person opens door"
        )
        matching = [
            alias
            for alias, chain in candidates.items()
            if chain["action_sequence"][0]["target_semantic_key"] == [desired_first]
        ]
        assert matching
        if self.frontier_calls == 0:
            delayed = [
                alias
                for alias in matching
                if any(
                    len(outcome["imagined_transitions"]) == 2
                    and outcome["imagined_transitions"][1]["target_semantic_key"]
                    == ["person opens door"]
                    for outcome in candidates[alias]["hypothesis_conditioned_outcomes"]
                )
            ]
            if delayed:
                return delayed[0]
        return matching[0]


def test_structured_9b_closes_two_multi_path_planning_cycles() -> None:
    graph = _graph()
    pool = _pool()
    client = _StructuredClient()
    model = Structured9BMultiTrajectoryModel(client)
    planner = MultiTrajectoryRolloutPlanner(
        model,
        model,
        horizon=2,
        setwise_preference_model=model,
    )

    first = planner.plan(pool, graph)

    assert first.selected_action.target_id == "l1:first"
    expected_ids = {row.trajectory_id for row in pool.expandable}
    assert all(
        {outcome.trajectory_id for outcome in path.conditioned_outcomes} == expected_ids
        for path in first.imagined_paths
    )
    assert any(
        transition.structured_patch["belief_patch"]["supported_claims"]
        for path in first.imagined_paths
        for outcome in path.conditioned_outcomes
        for transition in outcome.transitions
    )
    assert all(
        "structured_belief_event_patch" in transition
        for chain in client.planner_payloads[0]["candidate_joint_chains"].values()
        for outcome in chain["hypothesis_conditioned_outcomes"]
        for transition in outcome["imagined_transitions"]
    )

    execution = execute_shared_trajectory_action(pool, first, graph)
    corrected_pool = execution.pool

    assert execution.observation.node_id == "l1:first"
    assert all(
        trajectory.belief.acquired_evidence == ("l1:first",)
        for trajectory in corrected_pool.trajectories
    )
    assert all(
        trajectory.belief.imagined_evidence == ()
        for trajectory in corrected_pool.trajectories
    )

    second = planner.plan(corrected_pool, graph)

    assert second.selected_action.target_id == "l1:second"
    assert client.frontier_calls == 2
    corrected_ids = {row.trajectory_id for row in corrected_pool.expandable}
    assert all(
        {outcome.trajectory_id for outcome in path.conditioned_outcomes}
        == corrected_ids
        for path in second.imagined_paths
    )
    assert all(audit["numeric_reward_present"] is False for audit in model.audits)
    assert all(audit["top_k_applied"] is False for audit in model.audits)


class _TransportVariantClient:
    model = "openai/gpt-5-mini-transport-fixture"

    def complete_json(self, *, task, payload):
        del task
        if "requests" in payload:
            # Real OpenAI-compatible responses may flatten the sole
            # ``predictions`` object, wrap enums in singleton arrays and encode
            # empty/string-list fields as categorical scalar sentinels.
            return {
                alias: {
                    "observation_patch": {
                        "event_or_state": "candidate observation",
                        "entity_bindings": (
                            "entrant: orange-haired person"
                            if request["action"]["target"]
                            else "unbound"
                        ),
                        "temporal_binding": "unknown",
                        "evidence_role": "identity_evidence",
                        "alternatives": "none",
                    },
                    "belief_patch": {
                        "supported_claims": "none",
                        "contradicted_claims": "none",
                        "newly_bound_variables": "none",
                        "opened_dependencies": "none",
                        "resolved_dependencies": "none",
                    },
                    "categorical_audit": {
                        "observation_outcome": ["inconclusive"],
                        "progress": ["unchanged"],
                        "answerability_after": ["not_ready"],
                        "frontier_change": ["unchanged"],
                        "contradiction_change": ["unchanged"],
                        "resolved_roles": "",
                        "opened_roles": "identity,after_event",
                        "relation_updates": "",
                    },
                }
                for alias, request in payload["requests"].items()
            }
        aliases = list(payload["candidate_joint_chains"])
        return {
            "status": "incomparable",
            "preferred": aliases,
            "rationale": "retain the complete undominated frontier",
        }


def test_gpt5mini_transport_variants_preserve_incomparable_frontier() -> None:
    model = Structured9BMultiTrajectoryModel(_TransportVariantClient())
    decision = MultiTrajectoryRolloutPlanner(
        model,
        model,
        horizon=1,
        setwise_preference_model=model,
    ).plan(_pool(), _graph())

    assert decision.selected_action.kind.value == "abstain"
    assert decision.planning_status == "rollout_abstain_non_unique_first_hops"
    assert set(decision.preferred_path_ids) == {
        path.path_id for path in decision.imagined_paths
    }
    assert all(
        isinstance(
            transition.structured_patch["observation_patch"]["entity_bindings"],
            dict,
        )
        for path in decision.imagined_paths
        for outcome in path.conditioned_outcomes
        for transition in outcome.transitions
    )
