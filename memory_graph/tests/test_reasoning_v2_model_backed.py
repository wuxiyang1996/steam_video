from __future__ import annotations

import json

from steam_video_new.implicit_world_model.full_graph_iwm.transition_cache import (
    PersistentCategoricalResponseCacheClient,
)
from steam_video_new.implicit_world_model.reasoning_v2.evaluation.model_smoke import (
    run_smoke,
)
from steam_video_new.implicit_world_model.reasoning_v2.evidence import (
    EvidenceAddress,
    EvidenceMemory,
    EvidenceRecord,
    EvidenceValue,
)
from steam_video_new.implicit_world_model.reasoning_v2.navigation import (
    ModelBackedEntryLocalizer,
    NavigationGraph,
)
from steam_video_new.implicit_world_model.reasoning_v2.planner import (
    ModelBackedJointTreePlanner,
    PersistentMultiPathPlanner,
    apply_real_read,
    initialize_forest,
)
from steam_video_new.implicit_world_model.reasoning_v2.world_model import (
    Answerability,
    BeliefState,
    ModelBackedHypothesisEffectModel,
    ModelBackedObservationWorldModel,
    ModelBackedRealEffectCorrector,
    Progress,
)


class _FixtureClient:
    model = "fixture/categorical"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(self, *, task, payload):
        self.calls.append(task)
        if "candidate_addresses" in payload:
            return {
                "status": "located",
                "preferred": ["address_0"],
                "rationale": "first address is the fixture entry",
            }
        if "requests" in payload and "kind" in payload["allowed_output"]:
            return {
                "predictions": [
                    {
                        "request_id": row["request_id"],
                        "kind": "bridge"
                        if row["action"]["target_id"] == "a"
                        else "event",
                        "event_family": row["target_address"]["event_family"],
                        "entity_facts": [],
                        "state_facts": [],
                        "state_delta": [],
                        "rationale": "imagined physical descriptor",
                    }
                    for row in payload["requests"]
                ]
            }
        if "requests" in payload and "progress" in payload["allowed_output"]:
            return {
                "effects": [
                    {
                        "request_id": row["request_id"],
                        "progress": (
                            "advanced"
                            if row["imagined_observation"]["target_id"] == "a"
                            and row["hypothesis"] == "left"
                            else "unchanged"
                        ),
                        "answerability_after": "not_ready",
                        "contradiction_change": "unchanged",
                        "frontier_change": (
                            "opened"
                            if row["imagined_observation"]["target_id"] == "a"
                            else "unchanged"
                        ),
                        "resolved_roles": [],
                        "opened_roles": [],
                        "rationale": "hypothesis-conditioned effect",
                    }
                    for row in payload["requests"]
                ]
            }
        if "real_observation" in payload:
            return {
                "corrections": [
                    {
                        "path_id": row["path_id"],
                        "hypothesis_status": (
                            "support"
                            if row["hypothesis"] == "left"
                            else "counterevidence"
                        ),
                        "resolved_roles": (
                            row["missing_roles"][:1]
                            if row["hypothesis"] == "left"
                            else []
                        ),
                        "answerability_after": (
                            "ready" if row["hypothesis"] == "left" else "not_ready"
                        ),
                        "rationale": "direct grounded interpretation",
                    }
                    for row in payload["paths"]
                ]
            }
        trees = payload["joint_action_trees"]
        selected = next(
            (
                row
                for row in trees
                if row["shared_first_read"]["target_id"] == "a"
            ),
            trees[0],
        )
        return {
            "status": "select",
            "frontier_tree_ids": [selected["tree_id"]],
            "selected_tree_id": selected["tree_id"],
            "rationale": "prefer the joint tree with a differentiating future",
        }


def _memory() -> EvidenceMemory:
    return EvidenceMemory(
        "memory:model-backed",
        tuple(
            EvidenceRecord(
                EvidenceAddress(
                    node_id,
                    "video",
                    float(index),
                    float(index + 1),
                    "action",
                    semantic_key=f"safe address {node_id}",
                ),
                EvidenceValue(f"grounded {node_id}", f"predicate_{node_id}"),
            )
            for index, node_id in enumerate(("a", "b"))
        ),
        (),
    )


def test_model_backed_heads_feed_one_joint_tree_decision() -> None:
    client = _FixtureClient()
    planner = PersistentMultiPathPlanner(
        ModelBackedObservationWorldModel(client),
        ModelBackedHypothesisEffectModel(client),
        ModelBackedJointTreePlanner(client),
        horizon=1,
    )
    belief = BeliefState(
        "question",
        required_roles=("answer",),
        missing_roles=("answer",),
        answerability=Answerability.NOT_READY,
    )
    forest = initialize_forest(
        forest_id="forest:model-backed",
        belief=belief,
        hypotheses=("left", "right"),
        read_budget=2,
    )
    decision = planner.plan(
        forest,
        _memory(),
        NavigationGraph("graph", ("a", "b"), (), entry_node_ids=("a", "b")),
    )

    assert len(client.calls) == 3
    assert decision.selected_action is not None
    assert decision.selected_action.target_id == "a"
    selected = next(
        row
        for row in decision.candidates
        if row.first_action.shared_key == decision.selected_action.shared_key
    )
    assert set(selected.covered_hypotheses) == {"left", "right"}
    effects = [trajectory.steps[0].effect for trajectory in selected.trajectories]
    assert {effect.progress for effect in effects} == {
        Progress.ADVANCED,
        Progress.UNCHANGED,
    }

    updated = apply_real_read(
        forest,
        decision,
        _memory(),
        corrector=ModelBackedRealEffectCorrector(client),
    )
    by_hypothesis = {path.hypothesis: path.belief for path in updated.paths}
    assert by_hypothesis["left"].answerability is Answerability.READY
    assert by_hypothesis["right"].answerability is Answerability.NOT_READY
    assert by_hypothesis["right"].contradictions


class _NestedObservationClient:
    model = "fixture/nested-observation"

    def __init__(self, *, fail=False) -> None:
        self.fail = fail
        self.calls = 0

    def complete_json(self, *, task, payload):
        del task
        if self.fail:
            raise AssertionError("replay called delegate")
        self.calls += 1
        row = payload["requests"][0]
        prediction = {
            "request_id": row["request_id"],
            "kind": "event",
            "event_family": "action",
            "entity_facts": [],
            "state_facts": [],
            "state_delta": [],
            "rationale": "categorical fixture",
        }
        if (payload.get("repair_feedback") or {}).get("repair_attempt") == 2:
            return {"predictions": [prediction]}
        return {
            "predictions": [
                {
                    "request_id": row["request_id"],
                    "prediction": prediction,
                }
            ]
        }


def test_cached_schema_repair_uses_distinct_attempt_keys(tmp_path) -> None:
    from steam_video_new.implicit_world_model.reasoning_v2.navigation.actions import (
        compile_legal_actions,
    )
    from steam_video_new.implicit_world_model.reasoning_v2.world_model import (
        ObservationContext,
        ObservationRequest,
    )

    memory = _memory()
    graph = NavigationGraph("graph", ("a", "b"), (), entry_node_ids=("a",))
    action = compile_legal_actions(graph, cursor_id=None, acquired_ids=())[0]
    request = ObservationRequest(
        "request",
        BeliefState("question"),
        action,
        memory.read("a").address,
        ObservationContext(()),
    )
    path = tmp_path / "responses.json"
    delegate = _NestedObservationClient()
    recorder = PersistentCategoricalResponseCacheClient(delegate, path)

    result = ModelBackedObservationWorldModel(recorder).predict_batch((request,))

    assert result[0].descriptor.kind.value == "event"
    assert delegate.calls == 3
    assert json.loads(path.read_text())["entry_count"] == 3

    replay = PersistentCategoricalResponseCacheClient(
        _NestedObservationClient(fail=True), path, mode="replay"
    )
    replayed = ModelBackedObservationWorldModel(replay).predict_batch((request,))
    assert replayed == result


class _LocalizationClient:
    model = "fixture/localizer"

    def __init__(self) -> None:
        self.payload = None

    def complete_json(self, *, task, payload):
        del task
        self.payload = payload
        return {
            "status": "located",
            "preferred": ["address_1"],
            "rationale": "second safe address matches the question",
        }


def test_entry_localizer_sees_all_safe_keys_but_no_unread_values() -> None:
    client = _LocalizationClient()
    selected = ModelBackedEntryLocalizer(client).localize(
        question="question",
        missing_roles=("answer",),
        memory=_memory(),
    )

    assert selected == ("b",)
    assert client.payload is not None
    addresses = client.payload["candidate_addresses"]
    assert set(addresses) == {"address_0", "address_1"}
    assert addresses["address_0"]["semantic_key"] == "safe address a"
    assert "grounded a" not in repr(addresses)


def test_dual_entry_complete_graph_smoke_runs_two_closed_loop_reads(tmp_path) -> None:
    dataset = tmp_path / "dataset.json"
    gate = tmp_path / "gate.json"
    graph_root = tmp_path / "graphs"
    graph_dir = graph_root / "video"
    graph_dir.mkdir(parents=True)
    output = tmp_path / "result.json"
    dataset.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "case_id": "case",
                        "video_id": "video",
                        "planner_input": {
                            "question": "question",
                            "choices": ["left", "right"],
                        },
                    }
                ]
            }
        )
    )
    gate.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case",
                        "retained_l1_node_ids_by_clue": [["a"], ["b"]],
                    }
                ]
            }
        )
    )
    (graph_dir / "l1_l15_navigation_graph.json").write_text(
        json.dumps(
            {
                "graph_id": "graph",
                "metadata": {"question_independent": True},
                "nodes": [
                    {
                        "node_id": node_id,
                        "video_id": "video",
                        "time_span": {"start_s": index, "end_s": index + 1},
                        "node_type": "observation",
                        "text": f"grounded {node_id}",
                        "metadata": {
                            "predicate": f"predicate_{node_id}",
                            "action_kind": "action",
                        },
                    }
                    for index, node_id in enumerate(("a", "b"))
                ],
                "temporal_edges": [
                    {
                        "edge_id": "edge:ab",
                        "src": "a",
                        "dst": "b",
                        "relation": "temporal_next",
                    }
                ],
                "correlation_edges": [],
            }
        )
    )

    result = run_smoke(
        client=_FixtureClient(),
        navigation_dataset_path=dataset,
        gate_path=gate,
        graph_root=graph_root,
        case_id="case",
        output_path=output,
    )

    assert set(result["protocols"]) == {"oracle_entry", "learned_entry"}
    for protocol in result["protocols"].values():
        assert protocol["complete_graph"]["preserved"] is True
        assert protocol["arms"]["iwm"]["executed_target_ids"] == ["a", "b"]
        assert protocol["gates"][
            "iwm_replan_moves_toward_later_clue_shortest_path"
        ] is True
        assert protocol["gates"]["iwm_later_clue_reached_strict_interval"] is True
        assert protocol["route_budget_audit_hidden_evaluator_only"] == {
            "entry_to_later_clue_edge_distance": 1,
            "minimum_reads_including_entry": 2,
            "matched_read_budget": 2,
            "budget_feasible": True,
        }
    assert set(result["metric_families"]) == {
        "answer_accuracy",
        "read_efficiency",
        "action_divergence",
        "strict_localization",
    }
    assert result["metric_families"]["read_efficiency"][
        "matched_read_budget"
    ] == 2
    assert output.with_suffix(
        output.suffix + ".training_export.blocked.json"
    ).is_file()
    assert result["training_performed"] is False
