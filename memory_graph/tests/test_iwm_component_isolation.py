from __future__ import annotations

from steam_video_new.implicit_world_model.full_graph_iwm import (
    ActionKind,
    CursorBeliefState,
    HypothesisExpansionRequest,
    LegalGraphAction,
    RetainedEvidenceGraph,
)
from steam_video_new.implicit_world_model.full_graph_iwm.component_isolation import (
    ComposedTeacherForcedWorldModel,
    component_isolation_readiness,
)
from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
    AnswerabilityState,
    CategoricalBeliefDelta,
    ContradictionChange,
    EvidenceOutcome,
    FrontierChange,
    ImaginedTransition,
    PredictedObservation,
    ProgressChange,
)


class _FixedWorldModel:
    def __init__(self, outcome, progress, answerability):
        self.model_name = f"fixed:{outcome.value}"
        self.outcome = outcome
        self.progress = progress
        self.answerability = answerability

    def predict_batch(self, requests, graph):
        del graph
        return tuple(
            ImaginedTransition(
                action=row.action,
                observation=PredictedObservation(
                    target_id=row.action.target_id,
                    outcome=self.outcome,
                    descriptor=(self.model_name,),
                ),
                belief_delta=CategoricalBeliefDelta(
                    progress=self.progress,
                    answerability_after=self.answerability,
                    frontier_change=FrontierChange.UNCHANGED,
                    contradiction_change=ContradictionChange.UNCHANGED,
                ),
            )
            for row in requests
        )


def _request():
    action = LegalGraphAction(
        "action:read",
        ActionKind.START_AT,
        target_id="node:one",
        reads_evidence=True,
    )
    return HypothesisExpansionRequest(
        request_id="request:one",
        trajectory_id="trajectory:one",
        hypothesis="answer one",
        belief=CursorBeliefState("belief:one", "question", remaining_reads=1),
        action=action,
    )


def test_teacher_forced_world_model_composes_observation_and_effect_channels() -> None:
    predicted = _FixedWorldModel(
        EvidenceOutcome.INCONCLUSIVE,
        ProgressChange.UNCHANGED,
        AnswerabilityState.NOT_READY,
    )
    oracle = _FixedWorldModel(
        EvidenceOutcome.STATE_EVIDENCE,
        ProgressChange.ADVANCED,
        AnswerabilityState.READY,
    )
    model = ComposedTeacherForcedWorldModel(
        predicted,
        oracle,
        observation_source="predicted",
        effect_source="oracle",
    )

    transition = model.predict_batch(
        (_request(),),
        RetainedEvidenceGraph("graph:empty", (), (), (), 1),
    )[0]

    assert transition.observation.outcome is EvidenceOutcome.INCONCLUSIVE
    assert transition.belief_delta.progress is ProgressChange.ADVANCED
    assert transition.belief_delta.answerability_after is AnswerabilityState.READY


def test_component_isolation_readiness_rejects_unsupervised_effect_candidates() -> None:
    report = component_isolation_readiness(
        [
            {
                "input": {"observation_record_id": "observation:one"},
                "target": {
                    "hypothesis_effect": {
                        "value": None,
                        "supervised": False,
                    }
                },
            }
        ]
    )

    assert report["passed"] is False
    assert report["independently_supervised_effect_count"] == 0
    assert report["dataset_clue_overlap_used_as_hypothesis_effect"] is False


def test_component_isolation_readiness_requires_divergent_effects_and_preference() -> (
    None
):
    effects = [
        {
            "input": {"observation_record_id": "observation:one"},
            "target": {
                "hypothesis_effect": {
                    "value": {"effect": effect},
                    "supervised": True,
                }
            },
        }
        for effect in ("support", "counterevidence")
    ]
    preferences = [
        {
            "target": {"preference": {"value": "prefer_left", "supervised": True}},
            "training_eligible": True,
        }
    ]

    report = component_isolation_readiness(effects, preferences)

    assert report["passed"] is True
    assert report["divergent_effect_group_count"] == 1
