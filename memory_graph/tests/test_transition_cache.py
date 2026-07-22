from __future__ import annotations

from pathlib import Path

import pytest

from steam_video_new.implicit_world_model.cgbench_grounded_navigation.transition_descriptor_cache import (
    build_grounded_transition_corpus,
)
from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
    ActionKind,
    AnswerabilityState,
    CategoricalBeliefDelta,
    CursorBeliefState,
    EvidenceOutcome,
    IWMGraphInput,
    IWMRequest,
    ImaginedTransition,
    LegalGraphAction,
    NodeKey,
    NodeModelView,
    PredictedObservation,
    ProgressChange,
)
from steam_video_new.implicit_world_model.full_graph_iwm.transition_cache import (
    PersistentCategoricalResponseCacheClient,
    PersistentQuestionRoleCache,
    PersistentTransitionCacheWorldModel,
)
from steam_video_new.implicit_world_model.full_graph_iwm.gpt_oss import (
    GPTOSSFullGraphWorldModel,
)


class _WorldModel:
    model_name = "test-categorical-model"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def predict_batch(self, requests):
        if self.fail:
            raise AssertionError("replay called the delegate")
        self.calls += 1
        return tuple(
            ImaginedTransition(
                action=request.action,
                observation=PredictedObservation(
                    target_id=request.action.target_id,
                    outcome=EvidenceOutcome.SUPPORT,
                    descriptor=("grounded event",),
                ),
                belief_delta=CategoricalBeliefDelta(
                    progress=ProgressChange.ADVANCED,
                    answerability_after=AnswerabilityState.NOT_READY,
                    resolved_roles=("effect",),
                ),
            )
            for request in requests
        )


class _InterruptingWorldModel(_WorldModel):
    batch_size = 2

    def predict_batch(self, requests):
        if self.calls == 1:
            raise RuntimeError("simulated interrupted gather")
        return super().predict_batch(requests)


class _Initializer:
    model_name = "test-categorical-model"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def initialize(self, question: str) -> tuple[str, ...]:
        if self.fail:
            raise AssertionError("role replay called the delegate")
        self.calls += 1
        return ("cause", "effect")


class _CategoricalClient:
    model = "test-categorical-client"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.last_response_audit = {}

    def complete_json(self, *, task, payload):
        if self.fail:
            raise AssertionError("response replay called the delegate")
        self.calls += 1
        self.last_response_audit = {"finish_reason": "stop"}
        return {"status": "tie", "preferred": ["left", "right"]}


def _request(
    *,
    question: str = "what happened next?",
    belief_id: str = "belief:initial",
) -> IWMRequest:
    action = LegalGraphAction(
        action_id="action:read",
        kind=ActionKind.START_AT,
        target_id="l1:event",
        reads_evidence=True,
    )
    belief = CursorBeliefState(
        belief_id=belief_id,
        question=question,
        missing_roles=("effect",),
    )
    graph_input = IWMGraphInput(
        question=question,
        current_node_id=None,
        nodes=(
            NodeModelView(
                key=NodeKey(
                    node_id="l1:event",
                    node_type="observation",
                    start_s=1.0,
                    end_s=2.0,
                    semantic_key="dog becomes wet",
                    embedding_ref=None,
                ),
                acquired=False,
            ),
        ),
        temporal_edges=(),
        correlation_edges=(),
        candidate_edges=(),
        verified_relations=(),
        legal_actions=(action,),
        acquired_evidence=(),
        missing_roles=("effect",),
        contradictions=(),
        answerability=AnswerabilityState.NOT_READY,
    )
    return IWMRequest(belief=belief, graph_input=graph_input, action=action)


def test_persistent_transition_cache_records_then_replays(tmp_path: Path) -> None:
    cache_path = tmp_path / "transitions.json"
    recorder_delegate = _WorldModel()
    recorder = PersistentTransitionCacheWorldModel(
        recorder_delegate, cache_path, mode="record"
    )
    recorded = recorder.predict_batch((_request(),))

    assert recorder_delegate.calls == 1
    assert recorder.entry_count == 1
    assert recorded[0].observation.descriptor == ("dog becomes wet",)

    replay = PersistentTransitionCacheWorldModel(
        _WorldModel(fail=True), cache_path, mode="replay"
    )
    replayed = replay.predict_batch((_request(),))
    assert replayed == recorded
    assert replay.cache_audits[-1]["hit_count"] == 1

    with pytest.raises(RuntimeError, match="cache miss"):
        replay.predict_batch((_request(question="different question"),))


def test_transition_cache_commits_each_transport_batch(tmp_path: Path) -> None:
    path = tmp_path / "partial-transitions.json"
    source = _InterruptingWorldModel()
    requests = tuple(_request(question=f"question {index}") for index in range(3))
    record = PersistentTransitionCacheWorldModel(source, path, mode="record")
    with pytest.raises(RuntimeError, match="interrupted gather"):
        record.predict_batch(requests)

    persisted = PersistentTransitionCacheWorldModel(
        _WorldModel(fail=True), path, mode="replay"
    )
    assert len(persisted.predict_batch(requests[:2])) == 2
    with pytest.raises(RuntimeError, match="cache miss"):
        persisted.predict_batch(requests[2:])


def test_world_model_transport_batches_limit_unique_contexts() -> None:
    model = GPTOSSFullGraphWorldModel(
        _WorldModel(), batch_size=48, max_contexts_per_batch=2
    )
    batches = model._transport_batches(
        tuple(
            _request(question=f"question {index}", belief_id=f"belief:{index}")
            for index in range(5)
        )
    )
    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_question_role_cache_stabilizes_retry_state(tmp_path: Path) -> None:
    path = tmp_path / "roles.json"
    source = _Initializer()
    record = PersistentQuestionRoleCache(source, path, mode="record")
    assert record.initialize("what happened?") == ("cause", "effect")
    assert source.calls == 1

    replay = PersistentQuestionRoleCache(
        _Initializer(fail=True), path, mode="replay"
    )
    assert replay.initialize("what happened?") == ("cause", "effect")
    with pytest.raises(RuntimeError, match="role cache miss"):
        replay.initialize("different question")


def test_categorical_response_cache_replays_without_storing_prompt(tmp_path: Path) -> None:
    path = tmp_path / "responses.json"
    source = _CategoricalClient()
    record = PersistentCategoricalResponseCacheClient(source, path, mode="record")
    expected = record.complete_json(
        task="compare", payload={"question": "private question", "choices": ["a", "b"]}
    )
    assert source.calls == 1
    assert "private question" not in path.read_text(encoding="utf-8")

    replay = PersistentCategoricalResponseCacheClient(
        _CategoricalClient(fail=True), path, mode="replay"
    )
    assert replay.complete_json(
        task="compare", payload={"question": "private question", "choices": ["a", "b"]}
    ) == expected
    assert replay.response_audits[-1]["cache_hit"] is True
    with pytest.raises(RuntimeError, match="response cache miss"):
        replay.complete_json(task="compare", payload={"question": "changed"})


def test_grounded_transition_corpus_is_video_split_safe() -> None:
    def case(case_id: str, video_id: str, split: str, row: int) -> dict:
        return {
            "case_id": case_id,
            "video_id": video_id,
            "split": split,
            "planner_input": {"question": "what changed?"},
            "executed_transitions": [
                {
                    "transition_id": f"transition:{case_id}",
                    "checkpoint": {"answerability": "unknown"},
                    "action": {
                        "action_type": "read_video_interval",
                        "interval": {"start_s": 1.0, "end_s": 2.0},
                    },
                    "real_observation": {
                        "descriptor_status": "grounded_qwen_vl_read",
                        "descriptor": {
                            "summary": "the object changes",
                            "question_relevance": "relevant",
                            "relevance_reason": "question-conditioned text",
                            "visible_states": ["changed"],
                        },
                        "observed_modalities": ["sampled_video_frames"],
                        "producer": {"model": "Qwen"},
                    },
                    "target": {
                        "observation_descriptor": {
                            "embedding_ref": {
                                "model": "Qwen-Embedding",
                                "row_index": row,
                            }
                        },
                        "belief_delta": {
                            "evidence_progress": "advances_required_clue_coverage"
                        },
                    },
                }
            ],
        }

    corpus = build_grounded_transition_corpus(
        {
            "dataset_id": "test",
            "cases": [
                case("train", "video:train", "train", 0),
                case("test", "video:test", "test", 1),
            ],
        }
    )

    assert corpus["record_count"] == 2
    assert corpus["split_contract"] == "video_disjoint"
    assert corpus["training_performed"] is False
    assert "question_relevance" not in corpus["records"][0][
        "real_transition_target"
    ]["observation_descriptor"]
