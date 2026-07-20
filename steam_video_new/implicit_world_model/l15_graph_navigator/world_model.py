"""Categorical transition and ordinal preference baselines."""

from __future__ import annotations

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay

from .contracts import (
    Answerability,
    BeliefDeltaDescriptor,
    BeliefSnapshot,
    EvidenceRole,
    ObservationDescriptor,
    PairwisePreference,
    PredictedTransition,
    PreferenceLabel,
    TrajectoryPrediction,
    UncertaintyChange,
)


_ACTION_ROLE = {
    NavigationActionType.SEMANTIC: EvidenceRole.SEMANTIC,
    NavigationActionType.TEMPORAL_BACK: EvidenceRole.TEMPORAL,
    NavigationActionType.TEMPORAL_FORWARD: EvidenceRole.TEMPORAL,
    NavigationActionType.TRACK_ENTITY: EvidenceRole.IDENTITY,
    NavigationActionType.INSPECT_STATE_CHANGE: EvidenceRole.STATE_TRANSITION,
    NavigationActionType.FOLLOW_DEPENDENCY: EvidenceRole.DEPENDENCY,
    NavigationActionType.CANDIDATE_CAUSE: EvidenceRole.DEPENDENCY,
    NavigationActionType.EFFECT: EvidenceRole.DEPENDENCY,
    NavigationActionType.FIND_BRIDGE: EvidenceRole.BRIDGE,
    NavigationActionType.SEARCH_COUNTEREVIDENCE: EvidenceRole.COUNTEREVIDENCE,
    NavigationActionType.VERIFY: EvidenceRole.VERIFICATION,
    NavigationActionType.STOP: EvidenceRole.NONE,
}


class RuleBasedObservationBeliefModel:
    """Transparent categorical baseline; it never produces action values."""

    def predict(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> PredictedTransition:
        role = _ACTION_ROLE[action.action_type]
        acquired = set(belief.acquired_evidence)
        unseen_targets = tuple(
            target for target in action.target_ids if target not in acquired
        )
        resolved_roles = (
            (role.value,)
            if unseen_targets
            and role.value in belief.missing_roles
            and _action_can_resolve(action, belief)
            else ()
        )
        remaining_roles = tuple(
            item for item in belief.missing_roles if item not in set(resolved_roles)
        )
        answerability = (
            Answerability.READY
            if (belief.acquired_evidence or unseen_targets)
            and not remaining_roles
            and not belief.contradictions
            else Answerability.NOT_READY
        )
        relation_updates = tuple(
            sorted(
                state.edge_id
                for state in belief.relation_states
                if action.source_id in {state.src, state.dst}
                and any(
                    target in {state.src, state.dst}
                    for target in action.target_ids
                )
            )
        )
        node_kind = _node_kind(action, overlay)
        observation = ObservationDescriptor(
            role=role,
            target_ids=action.target_ids,
            node_kind=node_kind,
            predicted_only=True,
        )
        delta = BeliefDeltaDescriptor(
            resolved_roles=resolved_roles,
            relation_updates=relation_updates,
            uncertainty_change=(
                UncertaintyChange.DECREASE
                if unseen_targets
                else UncertaintyChange.UNCHANGED
            ),
            answerability_after=answerability,
            predicted_only=True,
        )
        return PredictedTransition(action, observation, delta)


class RuleBasedTrajectoryPreferenceModel:
    """Partial-order comparator over categorical trajectory descriptors."""

    def compare(
        self,
        left: TrajectoryPrediction,
        right: TrajectoryPrediction,
        belief: BeliefSnapshot,
    ) -> PairwisePreference:
        left_stop = left.first_action.action_type is NavigationActionType.STOP
        right_stop = right.first_action.action_type is NavigationActionType.STOP

        if left_stop != right_stop:
            if belief.answerability is Answerability.READY:
                return _preference(
                    left,
                    right,
                    PreferenceLabel.PREFER_LEFT if left_stop else PreferenceLabel.PREFER_RIGHT,
                    "stop is preferred because the real belief is answer-ready",
                )
            return _preference(
                left,
                right,
                PreferenceLabel.PREFER_RIGHT if left_stop else PreferenceLabel.PREFER_LEFT,
                "a meaningful graph read is preferred while required roles remain",
            )

        left_outcome = _outcome(left, belief)
        right_outcome = _outcome(right, belief)
        if left_outcome == right_outcome:
            return _preference(
                left,
                right,
                PreferenceLabel.TIE,
                "the categorical trajectory outcomes are equivalent",
            )

        left_roles, left_ready, left_progress = left_outcome
        right_roles, right_ready, right_progress = right_outcome
        if left_ready != right_ready:
            return _preference(
                left,
                right,
                PreferenceLabel.PREFER_LEFT if left_ready else PreferenceLabel.PREFER_RIGHT,
                "one trajectory reaches the answer-ready belief category",
            )
        if left_roles > right_roles and (left_progress or not right_progress):
            return _preference(
                left,
                right,
                PreferenceLabel.PREFER_LEFT,
                "the left trajectory resolves a strict superset of required roles",
            )
        if right_roles > left_roles and (right_progress or not left_progress):
            return _preference(
                left,
                right,
                PreferenceLabel.PREFER_RIGHT,
                "the right trajectory resolves a strict superset of required roles",
            )
        if left_roles and not right_roles:
            return _preference(
                left,
                right,
                PreferenceLabel.PREFER_LEFT,
                "only the left trajectory resolves a required role",
            )
        if right_roles and not left_roles:
            return _preference(
                left,
                right,
                PreferenceLabel.PREFER_RIGHT,
                "only the right trajectory resolves a required role",
            )
        if left_progress != right_progress:
            return _preference(
                left,
                right,
                PreferenceLabel.PREFER_LEFT if left_progress else PreferenceLabel.PREFER_RIGHT,
                "only one trajectory predicts a non-repeated real graph target",
            )
        return _preference(
            left,
            right,
            PreferenceLabel.INCOMPARABLE,
            "the trajectories make different progress without a justified ordering",
        )


def _node_kind(
    action: GraphReadAction,
    overlay: CausalTemporalOverlay,
) -> str:
    event_ids = {node.node_id for node in overlay.atomic_events}
    l1_ids = {node.node_id for node in overlay.l1_observations}
    kinds = {
        "atomic_event" if target in event_ids else "l1_observation"
        for target in action.target_ids
        if target in event_ids or target in l1_ids
    }
    if not kinds:
        return "none"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def _outcome(
    trajectory: TrajectoryPrediction,
    belief: BeliefSnapshot,
) -> tuple[frozenset[str], bool, bool]:
    resolved = frozenset(
        role
        for transition in trajectory.transitions
        for role in transition.belief_delta.resolved_roles
    )
    ready = any(
        transition.belief_delta.answerability_after is Answerability.READY
        for transition in trajectory.transitions
    )
    acquired = set(belief.acquired_evidence)
    progress = any(
        target not in acquired
        for transition in trajectory.transitions
        for target in transition.observation.target_ids
    )
    return resolved, ready, progress


def _preference(
    left: TrajectoryPrediction,
    right: TrajectoryPrediction,
    label: PreferenceLabel,
    rationale: str,
) -> PairwisePreference:
    return PairwisePreference(
        left_id=left.trajectory_id,
        right_id=right.trajectory_id,
        label=label,
        rationale=rationale,
    )


def _action_can_resolve(
    action: GraphReadAction,
    belief: BeliefSnapshot,
) -> bool:
    if action.action_type in {
        NavigationActionType.SEMANTIC,
        NavigationActionType.FIND_BRIDGE,
        NavigationActionType.SEARCH_COUNTEREVIDENCE,
    }:
        return True
    for state in belief.relation_states:
        if action.source_id not in {state.src, state.dst}:
            continue
        if not any(target in {state.src, state.dst} for target in action.target_ids):
            continue
        if action.relation in state.verified_relations:
            return True
        if (
            action.action_type
            in {NavigationActionType.TEMPORAL_BACK, NavigationActionType.TEMPORAL_FORWARD}
            and state.calibration_status == "deterministic"
        ):
            return True
    return False
