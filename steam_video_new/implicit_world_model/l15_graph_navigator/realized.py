"""Derive categorical supervision from two persisted belief snapshots."""

from __future__ import annotations

from .contracts import (
    Answerability,
    BeliefDeltaDescriptor,
    BeliefSnapshot,
    ContradictionChange,
    FrontierChange,
    HypothesisDisposition,
    HypothesisUpdate,
    PathChange,
    RecoveryStatus,
    RelationGrounding,
    UncertaintyChange,
    UncertaintyLevel,
)


def derive_realized_belief_delta(
    before: BeliefSnapshot,
    after: BeliefSnapshot,
) -> BeliefDeltaDescriptor:
    """Recompute a real delta after correction instead of trusting predictions.

    The result contains categories and identifiers only. Numeric solver state,
    imagined observations, and planner preferences cannot enter this contract.
    """

    before_states = {state.edge_id: state for state in before.relation_states}
    after_states = {state.edge_id: state for state in after.relation_states}
    changed_edges = tuple(
        sorted(
            edge_id
            for edge_id in set(before_states) | set(after_states)
            if edge_id not in before_states
            or edge_id not in after_states
            or before_states[edge_id].grounding != after_states[edge_id].grounding
        )
    )
    hypothesis_updates = tuple(
        HypothesisUpdate(
            edge_id=edge_id,
            disposition=_disposition(after_states[edge_id].grounding),
        )
        for edge_id in changed_edges
        if edge_id in after_states
    )
    before_frontier = set(before.frontier)
    after_frontier = set(after.frontier)
    before_contradictions = set(before.contradictions)
    after_contradictions = set(after.contradictions)
    before_blocked = set(before.blocked_edge_ids)
    after_blocked = set(after.blocked_edge_ids)
    resolved_roles = tuple(
        role for role in before.missing_roles if role not in set(after.missing_roles)
    )

    frontier_change = _set_change(
        before_frontier,
        after_frontier,
        opened=FrontierChange.OPENED,
        closed=FrontierChange.CLOSED,
        unchanged=FrontierChange.UNCHANGED,
    )
    contradiction_change = _set_change(
        before_contradictions,
        after_contradictions,
        opened=ContradictionChange.OPENED,
        closed=ContradictionChange.RESOLVED,
        unchanged=ContradictionChange.UNCHANGED,
    )
    path_change = _set_change(
        before_blocked,
        after_blocked,
        opened=PathChange.BLOCKED,
        closed=PathChange.OPENED,
        unchanged=PathChange.UNCHANGED,
    )
    changed = bool(
        resolved_roles
        or changed_edges
        or before_frontier != after_frontier
        or before_contradictions != after_contradictions
        or before_blocked != after_blocked
        or before.answerability != after.answerability
        or before.uncertainty != after.uncertainty
    )
    recovered = bool(
        resolved_roles
        or len(after_contradictions) < len(before_contradictions)
        or len(after_blocked) < len(before_blocked)
        or (
            before.answerability is not Answerability.READY
            and after.answerability is Answerability.READY
        )
    )
    recovery_status = (
        RecoveryStatus.RECOVERED
        if recovered
        else RecoveryStatus.STALLED
        if not changed and after.answerability is not Answerability.READY
        else RecoveryStatus.UNCHANGED
    )
    contradiction_updates = tuple(
        sorted(before_contradictions ^ after_contradictions)
    )
    return BeliefDeltaDescriptor(
        resolved_roles=resolved_roles,
        relation_updates=changed_edges,
        contradiction_updates=contradiction_updates,
        hypothesis_updates=hypothesis_updates,
        frontier_change=frontier_change,
        contradiction_change=contradiction_change,
        path_change=path_change,
        recovery_status=recovery_status,
        uncertainty_change=_uncertainty_change(
            before.uncertainty,
            after.uncertainty,
        ),
        answerability_after=after.answerability,
        predicted_only=False,
    )


def _disposition(grounding: RelationGrounding) -> HypothesisDisposition:
    if grounding is RelationGrounding.VERIFIED:
        return HypothesisDisposition.ACCEPTED
    if grounding is RelationGrounding.CONTRADICTED:
        return HypothesisDisposition.REJECTED
    return HypothesisDisposition.UNRESOLVED


def _set_change(before, after, *, opened, closed, unchanged):
    additions = after - before
    removals = before - after
    if additions:
        return opened
    if removals:
        return closed
    return unchanged


def _uncertainty_change(
    before: UncertaintyLevel,
    after: UncertaintyLevel,
) -> UncertaintyChange:
    order = {
        UncertaintyLevel.LOW: 0,
        UncertaintyLevel.MEDIUM: 1,
        UncertaintyLevel.HIGH: 2,
    }
    if order[after] < order[before]:
        return UncertaintyChange.DECREASE
    if order[after] > order[before]:
        return UncertaintyChange.INCREASE
    return UncertaintyChange.UNCHANGED
