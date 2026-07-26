"""Transparent baselines used to verify the v2 execution contract."""

from __future__ import annotations

from typing import Sequence

from .contracts import (
    Answerability,
    BeliefEffect,
    FrontierChange,
    HypothesisEffectRequest,
    ObservationDescriptor,
    ObservationKind,
    ObservationPrediction,
    ObservationRequest,
    Progress,
)


class AddressOnlyObservationBaseline:
    """Conservative baseline; never claims unseen entity/state attributes."""

    model_name = "address-only-observation-baseline/v0.1"

    def predict_batch(
        self, requests: Sequence[ObservationRequest]
    ) -> Sequence[ObservationPrediction]:
        return tuple(
            ObservationPrediction(
                request_id=row.request_id,
                action_id=row.action.action_id,
                target_id=row.target_address.node_id,
                descriptor=ObservationDescriptor(
                    kind=ObservationKind.EVENT,
                    event_family=row.target_address.event_family,
                ),
                rationale="coarse address family only",
            )
            for row in requests
        )


class ConservativeEffectBaseline:
    """Fail-closed baseline used until independent effect labels exist."""

    model_name = "conservative-effect-baseline/v0.1"

    def predict_effect_batch(
        self, requests: Sequence[HypothesisEffectRequest]
    ) -> Sequence[BeliefEffect]:
        return tuple(
            BeliefEffect(
                request_id=row.request_id,
                progress=Progress.UNCHANGED,
                answerability_after=Answerability.NOT_READY,
                frontier_change=FrontierChange.UNCHANGED,
                rationale="no independently supervised hypothesis effect",
            )
            for row in requests
        )
