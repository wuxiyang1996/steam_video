"""Controlled world-model interventions for dependence experiments."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

from memory_graph.navigation import GraphReadAction
from memory_graph.types import CausalTemporalOverlay

from .contracts import (
    BeliefDeltaDescriptor,
    BeliefSnapshot,
    EvidenceRole,
    ObservationBeliefWorldModel,
    ObservationDescriptor,
    PredictedTransition,
    TrajectoryPrediction,
    UncertaintyChange,
)


class TransitionIntervention(str, Enum):
    NORMAL = "normal"
    NULL = "null"
    SHUFFLED = "shuffled"


class FrozenBeliefWorldModel:
    """Predict every step from the first observed belief checkpoint."""

    def __init__(self, base: ObservationBeliefWorldModel) -> None:
        self.base = base
        self._frozen_belief: BeliefSnapshot | None = None

    def predict(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> PredictedTransition:
        if self._frozen_belief is None:
            self._frozen_belief = belief
        return self.base.predict(self._frozen_belief, action, overlay)


def intervene_trajectories(
    trajectories: tuple[TrajectoryPrediction, ...],
    belief: BeliefSnapshot,
    mode: TransitionIntervention,
) -> tuple[TrajectoryPrediction, ...]:
    """Corrupt predictions while preserving legal actions and candidate count."""

    if mode is TransitionIntervention.NORMAL or not trajectories:
        return trajectories
    if mode is TransitionIntervention.NULL:
        return tuple(_null_trajectory(trajectory, belief) for trajectory in trajectories)

    # A deterministic rotation makes the destructive control reproducible while
    # severing the association between each action and its predicted outcome.
    donor_outcomes = [trajectory.transitions for trajectory in trajectories]
    donor_outcomes = donor_outcomes[1:] + donor_outcomes[:1]
    shuffled: list[TrajectoryPrediction] = []
    for recipient, donors in zip(trajectories, donor_outcomes, strict=True):
        transitions = tuple(
            replace(donor, action=recipient.transitions[index].action)
            for index, donor in enumerate(donors[: len(recipient.transitions)])
        )
        if len(transitions) < len(recipient.transitions):
            transitions += recipient.transitions[len(transitions) :]
        shuffled.append(replace(recipient, transitions=transitions))
    return tuple(shuffled)


def _null_trajectory(
    trajectory: TrajectoryPrediction,
    belief: BeliefSnapshot,
) -> TrajectoryPrediction:
    transitions = tuple(
        PredictedTransition(
            action=transition.action,
            observation=ObservationDescriptor(
                role=EvidenceRole.NONE,
                target_ids=(),
                node_kind="none",
                predicted_only=True,
            ),
            belief_delta=BeliefDeltaDescriptor(
                uncertainty_change=UncertaintyChange.UNCHANGED,
                answerability_after=belief.answerability,
                predicted_only=True,
            ),
        )
        for transition in trajectory.transitions
    )
    return replace(trajectory, transitions=transitions)
