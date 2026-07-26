"""Semantic integrity checks for imagined categorical belief transitions.

These checks do not rank or remove legal graph actions.  They only reject
internally impossible state descriptions before an imagined transition can be
projected into a second-hop belief.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    AnswerabilityState,
    ContradictionChange,
    CursorBeliefState,
    EvidenceOutcome,
    ImaginedTransition,
    ProgressChange,
)


@dataclass(frozen=True)
class TransitionContractReport:
    violations: tuple[str, ...]
    projected_answerability: AnswerabilityState

    @property
    def valid(self) -> bool:
        return not self.violations


class TransitionContractError(ValueError):
    """Raised when a model row is well-formed JSON but semantically impossible."""

    def __init__(self, report: TransitionContractReport) -> None:
        self.report = report
        super().__init__("; ".join(report.violations))


def projected_answerability(
    belief: CursorBeliefState,
    transition: ImaginedTransition,
) -> AnswerabilityState:
    """Compute answerability from the projected state, not from model assertion."""

    delta = transition.belief_delta
    missing = set(belief.missing_roles)
    missing.difference_update(delta.resolved_roles)
    missing.update(delta.opened_roles)

    contradictions_remain = bool(belief.contradictions)
    if delta.contradiction_change is ContradictionChange.RESOLVED:
        contradictions_remain = False
    elif delta.contradiction_change is ContradictionChange.OPENED:
        contradictions_remain = True

    required = set(belief.required_roles or belief.missing_roles)
    required.update(delta.opened_roles)
    evidence_ids = {
        node_id
        for node_id in belief.acquired_evidence
        if node_id not in set(belief.imagined_evidence)
    }
    if transition.action.reads_evidence and transition.action.target_id is not None:
        evidence_ids.add(transition.action.target_id)
    minimum_distinct_evidence = min(2, len(required))

    ready_eligible = (
        not missing
        and not contradictions_remain
        and len(evidence_ids) >= minimum_distinct_evidence
    )
    # The backend gates optimistic READY claims but does not force a model that
    # remains conservative to become ready.  ABSTAIN also remains a categorical
    # forecast rather than being rewritten into progress.
    if delta.answerability_after is AnswerabilityState.ABSTAIN:
        return AnswerabilityState.ABSTAIN
    if delta.answerability_after is AnswerabilityState.READY and ready_eligible:
        return AnswerabilityState.READY
    return AnswerabilityState.NOT_READY


def inspect_transition_contract(
    belief: CursorBeliefState,
    transition: ImaginedTransition,
) -> TransitionContractReport:
    delta = transition.belief_delta
    outcome = transition.observation.outcome
    violations: list[str] = []

    if set(delta.resolved_roles) & set(delta.opened_roles):
        violations.append("a role cannot be both resolved and opened")

    if outcome in {EvidenceOutcome.EMPTY, EvidenceOutcome.INCONCLUSIVE}:
        if delta.resolved_roles:
            violations.append(
                f"{outcome.value} observation cannot resolve evidence roles"
            )
        if delta.progress is ProgressChange.ADVANCED:
            violations.append(
                f"{outcome.value} observation cannot claim advanced progress"
            )
        if delta.contradiction_change is ContradictionChange.RESOLVED:
            violations.append(
                f"{outcome.value} observation cannot resolve a contradiction"
            )

    computed = projected_answerability(belief, transition)
    if delta.answerability_after is not computed:
        violations.append(
            "answerability_after disagrees with the projected categorical belief "
            f"(declared={delta.answerability_after.value}, computed={computed.value})"
        )

    return TransitionContractReport(tuple(violations), computed)


def validate_transition_contract(
    belief: CursorBeliefState,
    transition: ImaginedTransition,
) -> TransitionContractReport:
    report = inspect_transition_contract(belief, transition)
    if not report.valid:
        raise TransitionContractError(report)
    return report
