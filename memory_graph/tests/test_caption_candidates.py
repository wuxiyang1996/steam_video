from __future__ import annotations

from memory_graph.types import MemoryNode, TimeSpan
from steam_video_new.implicit_world_model.full_graph_iwm.action_compiler import (
    GraphActionCompiler,
)
from steam_video_new.implicit_world_model.full_graph_iwm.caption_candidates import (
    augment_graph_with_caption_candidates,
    propose_caption_candidate_overlay,
)
from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
    ActionKind,
    AnswerabilityState,
    CategoricalBeliefDelta,
    CursorBeliefState,
    EvidenceOutcome,
    ImaginedTransition,
    LegalGraphAction,
    PredictedObservation,
    ProgressChange,
    RetainedEvidenceGraph,
    TrajectoryPrediction,
)
from steam_video_new.implicit_world_model.full_graph_iwm.gpt_oss import (
    GPTOSSFullGraphSetwisePreferenceModel,
)


def _node(node_id: str, start: float, text: str) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        video_id="video:test",
        time_span=TimeSpan(start, start + 1.0),
        provenance={"producer": "test"},
        node_type="observation",
        text=text,
        metadata={"predicate": text, "participants": []},
    )


class _CaptionClient:
    model = "test-caption-model"

    def complete_json(self, *, task, payload):
        assert payload["question_independent"] is True
        assert "question" not in payload
        return {
            "pairs": [
                {
                    "src": "n0",
                    "dst": "n1",
                    "category": "event_continuation",
                    "direction": "src_to_dst",
                    "src_evidence": "person enters",
                    "dst_evidence": "person sits",
                }
            ]
        }


def test_caption_candidate_overlay_adds_scoreless_legal_hop() -> None:
    nodes = (_node("n0", 0.0, "person enters"), _node("n1", 10.0, "person sits"))
    graph = RetainedEvidenceGraph(
        graph_id="graph:test",
        nodes=nodes,
        temporal_edges=(),
        correlation_edges=(),
        capacity=2,
        metadata={"source_l1_fingerprint": "frozen"},
    )

    artifact = propose_caption_candidate_overlay(graph, _CaptionClient())
    augmented = augment_graph_with_caption_candidates(graph, artifact)

    assert artifact["numeric_score_present"] is False
    assert artifact["contains_question_answer_or_clue"] is False
    assert len(augmented.candidate_edges) == 1
    edge = augmented.candidate_edges[0]
    assert edge.to_dict()["verified_relation"] is False
    belief = CursorBeliefState(
        belief_id="belief:test",
        question="question",
        current_node_id="n0",
        acquired_evidence=("n0",),
        remaining_reads=1,
    )
    actions = GraphActionCompiler().compile(belief, augmented)
    candidate = [action for action in actions if action.edge_id == edge.edge_id]
    assert len(candidate) == 1
    assert candidate[0].kind is ActionKind.FOLLOW_CORRELATION
    assert candidate[0].target_id == "n1"
    assert candidate[0].relation == "caption_bridge"


class _SetwiseClient:
    model = "test-setwise-model"

    def complete_json(self, *, task, payload):
        assert payload["required_contract"]["complete_candidate_set_present"] is True
        aliases = list(payload["trajectories"])
        return {"status": "unique", "preferred": [aliases[1]], "rationale": "better"}


def test_setwise_preference_returns_trajectory_id_without_scores() -> None:
    actions = tuple(
        LegalGraphAction(
            action_id=f"action:{index}",
            kind=ActionKind.START_AT,
            target_id=f"n{index}",
            reads_evidence=True,
        )
        for index in range(2)
    )
    trajectories = tuple(
        TrajectoryPrediction(
            trajectory_id=f"trajectory:{index}",
            transitions=(
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
    belief = CursorBeliefState("belief:test", "question")

    selected = GPTOSSFullGraphSetwisePreferenceModel(_SetwiseClient()).select(
        trajectories, belief
    )

    assert selected == ("trajectory:1",)


def test_setwise_tournament_covers_all_groups_without_top_k() -> None:
    class FirstClient:
        model = "test-tournament"

        def __init__(self):
            self.seen = []
            self.grouped_calls = 0

        def complete_json(self, *, task, payload):
            if "independent_groups" in payload:
                self.grouped_calls += 1
                decisions = {}
                for group_alias, group in payload["independent_groups"].items():
                    aliases = list(group["trajectories"])
                    self.seen.append(len(aliases))
                    decisions[group_alias] = {
                        "status": "unique",
                        "preferred": [aliases[0]],
                        "rationale": "categorical",
                    }
                return {"decisions": decisions}
            aliases = list(payload["trajectories"])
            self.seen.append(len(aliases))
            return {
                "status": "unique",
                "preferred": [aliases[0]],
                "rationale": "categorical",
            }

    actions = tuple(
        LegalGraphAction(
            action_id=f"tournament-action:{index}",
            kind=ActionKind.START_AT,
            target_id=f"n{index}",
            reads_evidence=True,
        )
        for index in range(5)
    )
    trajectories = tuple(
        TrajectoryPrediction(
            trajectory_id=f"tournament:{index}",
            transitions=(
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
    client = FirstClient()
    model = GPTOSSFullGraphSetwisePreferenceModel(client, batch_size=2)

    selected = model.select(
        trajectories, CursorBeliefState("belief:tournament", "question")
    )

    assert selected == ("tournament:0",)
    assert client.seen == [2, 2, 2, 2]
    assert client.grouped_calls == 1
    assert model.tournament_audits[-1]["rounds"][0]["input_count"] == 5
