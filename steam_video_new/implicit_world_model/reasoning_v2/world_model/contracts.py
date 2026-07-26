"""Separated observation and hypothesis-effect world-model contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence

from ..evidence.contracts import EvidenceAddress, EvidenceValue
from ..navigation.contracts import NavigationAction


class ObservationKind(str, Enum):
    EVENT = "event"
    ENTITY_ATTRIBUTE = "entity_attribute"
    STATE = "state"
    STATE_CHANGE = "state_change"
    BRIDGE = "bridge"
    EMPTY = "empty"
    INCONCLUSIVE = "inconclusive"


class Progress(str, Enum):
    ADVANCED = "advanced"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"


class Answerability(str, Enum):
    READY = "ready"
    NOT_READY = "not_ready"
    ABSTAIN = "abstain"


class ContradictionChange(str, Enum):
    OPENED = "opened"
    RESOLVED = "resolved"
    UNCHANGED = "unchanged"


class FrontierChange(str, Enum):
    OPENED = "opened"
    CLOSED = "closed"
    SHIFTED = "shifted"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class BeliefState:
    question: str
    required_roles: tuple[str, ...] = ()
    missing_roles: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    answerability: Answerability = Answerability.NOT_READY
    grounded_role_evidence: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("belief requires a question")
        if not set(self.missing_roles).issubset(
            self.required_roles or self.missing_roles
        ):
            raise ValueError("missing belief roles must be required")


@dataclass(frozen=True)
class ObservationContext:
    acquired_values: tuple[tuple[str, EvidenceValue], ...]
    imagined_prefix: tuple[tuple[str, ObservationDescriptor], ...] = ()


@dataclass(frozen=True)
class ObservationRequest:
    request_id: str
    belief: BeliefState
    action: NavigationAction
    target_address: EvidenceAddress
    context: ObservationContext

    def __post_init__(self) -> None:
        if not self.request_id or self.action.target_id != self.target_address.node_id:
            raise ValueError("observation request target mismatch")


@dataclass(frozen=True)
class ObservationDescriptor:
    kind: ObservationKind
    event_family: str
    entity_facts: tuple[str, ...] = ()
    state_facts: tuple[str, ...] = ()
    state_delta: tuple[str, ...] = ()


def observation_world_key(request: ObservationRequest) -> tuple[object, ...]:
    """Hashable physical-world key, excluding path/hypothesis/request identity."""

    acquired = tuple(
        (
            node_id,
            value.descriptor,
            value.predicate,
            tuple(
                (
                    entity.mention_id,
                    entity.role,
                    entity.entity_type,
                    entity.surface,
                    entity.visual_signature,
                    entity.track_status,
                )
                for entity in value.entities
            ),
            tuple(
                (state.mention_id, state.attribute, state.value, state.polarity)
                for state in value.states
            ),
            (
                None
                if value.state_delta is None
                else (
                    value.state_delta.mention_id,
                    value.state_delta.attribute,
                    value.state_delta.before,
                    value.state_delta.after,
                )
            ),
        )
        for node_id, value in request.context.acquired_values
    )
    return (
        request.action.shared_key,
        request.belief,
        acquired,
        request.context.imagined_prefix,
    )


@dataclass(frozen=True)
class ObservationPrediction:
    request_id: str
    action_id: str
    target_id: str
    descriptor: ObservationDescriptor
    rationale: str = ""
    predicted_only: bool = True

    def __post_init__(self) -> None:
        if not self.predicted_only:
            raise ValueError("world-model observation must remain imagined")


@dataclass(frozen=True)
class HypothesisEffectRequest:
    request_id: str
    path_id: str
    hypothesis: str
    belief: BeliefState
    observation: ObservationPrediction

    def __post_init__(self) -> None:
        if not all((self.request_id, self.path_id, self.hypothesis.strip())):
            raise ValueError("hypothesis-effect request is incomplete")


@dataclass(frozen=True)
class BeliefEffect:
    request_id: str
    progress: Progress
    answerability_after: Answerability
    contradiction_change: ContradictionChange = ContradictionChange.UNCHANGED
    frontier_change: FrontierChange = FrontierChange.UNCHANGED
    resolved_roles: tuple[str, ...] = ()
    opened_roles: tuple[str, ...] = ()
    rationale: str = ""
    predicted_only: bool = True


@dataclass(frozen=True)
class RealObservation:
    action: NavigationAction
    value: EvidenceValue

    def __post_init__(self) -> None:
        if not self.action.reads_evidence or self.action.target_id is None:
            raise ValueError("real observation requires an executed read action")

    @property
    def action_id(self) -> str:
        return self.action.action_id

    @property
    def target_id(self) -> str:
        assert self.action.target_id is not None
        return self.action.target_id


@dataclass(frozen=True)
class GroundedBeliefEffect:
    path_id: str
    belief_after: BeliefState
    direct_same_target: bool
    verified: bool
    rationale: str = ""


class ObservationWorldModel(Protocol):
    model_name: str

    def predict_batch(
        self, requests: Sequence[ObservationRequest]
    ) -> Sequence[ObservationPrediction]: ...


class HypothesisEffectModel(Protocol):
    model_name: str

    def predict_effect_batch(
        self, requests: Sequence[HypothesisEffectRequest]
    ) -> Sequence[BeliefEffect]: ...


class RealEffectCorrector(Protocol):
    model_name: str

    def correct_batch(
        self,
        *,
        path_hypotheses: Mapping[str, str],
        beliefs: Mapping[str, BeliefState],
        acquired_ids_before: tuple[str, ...],
        observation: RealObservation,
    ) -> Sequence[GroundedBeliefEffect]: ...


def project_belief(belief: BeliefState, effect: BeliefEffect) -> BeliefState:
    """Apply an imagined categorical effect to an imagined path only."""

    missing = tuple(
        role for role in belief.missing_roles if role not in effect.resolved_roles
    )
    missing = tuple(dict.fromkeys((*missing, *effect.opened_roles)))
    return BeliefState(
        question=belief.question,
        required_roles=tuple(
            dict.fromkeys((*belief.required_roles, *effect.opened_roles))
        ),
        missing_roles=missing,
        contradictions=belief.contradictions,
        answerability=effect.answerability_after,
        grounded_role_evidence=belief.grounded_role_evidence,
    )
