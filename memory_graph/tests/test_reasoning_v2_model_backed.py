from __future__ import annotations

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
                            ["answer"] if row["hypothesis"] == "left" else []
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
            row for row in trees if row["shared_first_read"]["target_id"] == "a"
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
