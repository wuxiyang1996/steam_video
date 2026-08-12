"""Controlled world-model interventions for matched navigation ablations."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .contracts import (
    AnswerabilityState,
    BatchedCategoricalWorldModel,
    CategoricalBeliefDelta,
    EvidenceOutcome,
    IWMRequest,
    ImaginedTransition,
    PredictedObservation,
    ProgressChange,
)


class NullWorldModel(BatchedCategoricalWorldModel):
    """No-WM control: every action has the same uninformative consequence."""

    model_name = "null-world-model-control"

    def predict_batch(
        self, requests: Sequence[IWMRequest]
    ) -> Sequence[ImaginedTransition]:
        return tuple(
            ImaginedTransition(
                action=request.action,
                observation=PredictedObservation(
                    target_id=request.action.target_id,
                    outcome=EvidenceOutcome.INCONCLUSIVE,
                    descriptor=_bound_descriptor(request),
                ),
                belief_delta=CategoricalBeliefDelta(
                    progress=ProgressChange.UNCHANGED,
                    answerability_after=AnswerabilityState.NOT_READY,
                ),
            )
            for request in requests
        )


class ShuffledWorldModel(BatchedCategoricalWorldModel):
    """Permute imagined consequences while keeping the legal actions fixed."""

    def __init__(self, delegate: BatchedCategoricalWorldModel) -> None:
        self.delegate = delegate
        self.model_name = f"shuffled:{delegate.model_name}"
        self.batch_audits: list[dict[str, object]] = []

    def predict_batch(
        self, requests: Sequence[IWMRequest]
    ) -> Sequence[ImaginedTransition]:
        predictions = tuple(self.delegate.predict_batch(requests))
        if len(predictions) != len(requests):
            raise ValueError("delegate prediction coverage mismatch")
        if len(predictions) < 2:
            self.batch_audits.append(
                {"request_count": len(requests), "permutation": "identity_singleton"}
            )
            return tuple(
                _retarget(prediction, request)
                for prediction, request in zip(predictions, requests)
            )
        donors = predictions[1:] + predictions[:1]
        shuffled = tuple(
            ImaginedTransition(
                action=request.action,
                observation=replace(
                    donor.observation,
                    target_id=request.action.target_id,
                    descriptor=_bound_descriptor(request),
                ),
                belief_delta=donor.belief_delta,
            )
            for request, donor in zip(requests, donors)
        )
        changed = sum(
            _descriptor(original) != _descriptor(intervened)
            for original, intervened in zip(predictions, shuffled)
        )
        self.batch_audits.append(
            {
                "request_count": len(requests),
                "permutation": "deterministic_left_rotation",
                "changed_descriptor_count": changed,
            }
        )
        return shuffled


class FrozenWorldModel(BatchedCategoricalWorldModel):
    """Reuse each action's first prediction after the real belief changes."""

    def __init__(self, delegate: BatchedCategoricalWorldModel) -> None:
        self.delegate = delegate
        self.model_name = f"frozen:{delegate.model_name}"
        self._cache: dict[str, ImaginedTransition] = {}
        self.cache_audits: list[dict[str, int]] = []

    def predict_batch(
        self, requests: Sequence[IWMRequest]
    ) -> Sequence[ImaginedTransition]:
        missing = tuple(
            request
            for request in requests
            if request.action.action_id not in self._cache
        )
        if missing:
            predicted = tuple(self.delegate.predict_batch(missing))
            if len(predicted) != len(missing):
                raise ValueError("delegate prediction coverage mismatch")
            for request, transition in zip(missing, predicted):
                self._cache[request.action.action_id] = transition
        result = tuple(
            _retarget(self._cache[request.action.action_id], request)
            for request in requests
        )
        self.cache_audits.append(
            {
                "request_count": len(requests),
                "new_prediction_count": len(missing),
                "reused_prediction_count": len(requests) - len(missing),
            }
        )
        return result


def _retarget(cached: ImaginedTransition, request: IWMRequest) -> ImaginedTransition:
    return ImaginedTransition(
        action=request.action,
        observation=replace(
            cached.observation,
            target_id=request.action.target_id,
            descriptor=_bound_descriptor(request),
        ),
        belief_delta=cached.belief_delta,
    )


def _descriptor(transition: ImaginedTransition) -> tuple[object, ...]:
    delta = transition.belief_delta
    return (
        transition.observation.outcome,
        delta.progress,
        delta.answerability_after,
        delta.frontier_change,
        delta.contradiction_change,
        delta.resolved_roles,
        delta.opened_roles,
        delta.relation_updates,
    )


def _bound_descriptor(request: IWMRequest) -> tuple[str, ...]:
    if not request.action.reads_evidence:
        return ()
    descriptor = tuple(
        view.key.semantic_key
        for view in request.graph_input.nodes
        if view.key.node_id == request.action.target_id
    )
    if len(descriptor) != 1:
        raise ValueError("action target must bind to exactly one visible node key")
    return descriptor
