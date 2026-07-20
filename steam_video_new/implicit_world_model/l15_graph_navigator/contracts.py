"""Preference-only contracts for L1.5 graph navigation.

The public contracts deliberately contain no reward, utility, Q-value, or
trajectory score.  Numeric relation probabilities remain graph/belief
metadata; action selection is an ordinal pairwise decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

from memory_graph.navigation import GraphReadAction, NavigationBeliefState
from memory_graph.types import CausalTemporalOverlay, MemoryNode


class EvidenceRole(str, Enum):
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    IDENTITY = "identity"
    STATE_TRANSITION = "state_transition"
    DEPENDENCY = "dependency"
    BRIDGE = "bridge"
    COUNTEREVIDENCE = "counterevidence"
    VERIFICATION = "verification"
    NONE = "none"


class UncertaintyLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UncertaintyChange(str, Enum):
    DECREASE = "decrease"
    UNCHANGED = "unchanged"
    INCREASE = "increase"


class Answerability(str, Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    ABSTAIN = "abstain"


class RelationGrounding(str, Enum):
    UNSEEN = "unseen"
    PARTIAL = "partial"
    ENDPOINTS_OBSERVED = "endpoints_observed"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"


class PreferenceLabel(str, Enum):
    PREFER_LEFT = "prefer_left"
    TIE = "tie"
    PREFER_RIGHT = "prefer_right"
    INCOMPARABLE = "incomparable"


@dataclass(frozen=True)
class RelationState:
    """Question-conditioned state for one typed graph edge.

    Probabilities and correlation features are priors/audit metadata, never
    action utilities.
    """

    edge_id: str
    src: str
    dst: str
    relation_probabilities: tuple[tuple[str, float], ...]
    posterior_probabilities: tuple[tuple[str, float], ...] = ()
    correlation_features: tuple[tuple[str, float], ...] = ()
    verified_relations: tuple[str, ...] = ()
    factor_sources: tuple[str, ...] = ()
    calibration_status: str = "uncalibrated_prior"
    grounding: RelationGrounding = RelationGrounding.UNSEEN

    def __post_init__(self) -> None:
        for name, value in self.relation_probabilities:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} probability must be in [0, 1]")
        for name, value in self.posterior_probabilities:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} posterior must be in [0, 1]")
        for name, value in self.correlation_features:
            if not math.isfinite(value):
                raise ValueError(f"{name} correlation feature must be finite")


@dataclass(frozen=True)
class BeliefSnapshot:
    """Backend-neutral state exposed to the navigator and world model."""

    belief_id: str
    backend_name: str
    question: str
    acquired_evidence: tuple[str, ...] = ()
    frontier: tuple[str, ...] = ()
    missing_roles: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    relation_states: tuple[RelationState, ...] = ()
    priority_edge_ids: tuple[str, ...] = ()
    blocked_edge_ids: tuple[str, ...] = ()
    uncertainty: UncertaintyLevel = UncertaintyLevel.HIGH
    answerability: Answerability = Answerability.NOT_READY
    remaining_graph_reads: int = 8
    step: int = 0
    backend_ref: str | None = None

    def __post_init__(self) -> None:
        if self.remaining_graph_reads < 0:
            raise ValueError("remaining_graph_reads must be non-negative")

    def navigation_view(self) -> NavigationBeliefState:
        """Adapt to the existing graph-action generator.

        The legacy uncertainty scalar is used only by that data container. It
        is not emitted by the world model and never enters preference ranking.
        """

        uncertainty = {
            UncertaintyLevel.HIGH: 1.0,
            UncertaintyLevel.MEDIUM: 0.5,
            UncertaintyLevel.LOW: 0.1,
        }[self.uncertainty]
        return NavigationBeliefState(
            question=self.question,
            acquired_evidence=self.acquired_evidence,
            frontier=self.frontier,
            missing_roles=self.missing_roles,
            contradictions=self.contradictions,
            residual_uncertainty=uncertainty,
            remaining_graph_reads=self.remaining_graph_reads,
            step=self.step,
        )


@dataclass(frozen=True)
class ObservationDescriptor:
    role: EvidenceRole
    target_ids: tuple[str, ...]
    node_kind: str
    predicted_only: bool = True


@dataclass(frozen=True)
class BeliefDeltaDescriptor:
    resolved_roles: tuple[str, ...] = ()
    relation_updates: tuple[str, ...] = ()
    contradiction_updates: tuple[str, ...] = ()
    uncertainty_change: UncertaintyChange = UncertaintyChange.UNCHANGED
    answerability_after: Answerability = Answerability.NOT_READY
    predicted_only: bool = True


@dataclass(frozen=True)
class PredictedTransition:
    action: GraphReadAction
    observation: ObservationDescriptor
    belief_delta: BeliefDeltaDescriptor


@dataclass(frozen=True)
class TrajectoryPrediction:
    trajectory_id: str
    transitions: tuple[PredictedTransition, ...]

    @property
    def first_action(self) -> GraphReadAction:
        return self.transitions[0].action


@dataclass(frozen=True)
class PairwisePreference:
    left_id: str
    right_id: str
    label: PreferenceLabel
    rationale: str


@dataclass(frozen=True)
class PlanDecision:
    selected_action: GraphReadAction
    selected_trajectory_id: str
    trajectories: tuple[TrajectoryPrediction, ...]
    comparisons: tuple[PairwisePreference, ...]
    undominated_trajectory_ids: tuple[str, ...]
    fallback_policy: str = "stable_structural_order_for_tie_or_incomparable"


@dataclass(frozen=True)
class BeliefUpdateResult:
    belief: BeliefSnapshot
    delta: BeliefDeltaDescriptor
    audit_record: dict[str, object] | None = None


@dataclass(frozen=True)
class NavigationStep:
    belief_before_id: str
    decision: PlanDecision
    observation_ids: tuple[str, ...]
    belief_after_id: str
    realized_belief_delta: BeliefDeltaDescriptor | None = None
    skill_invocation: dict[str, object] | None = None
    belief_update_audit: dict[str, object] | None = None

    def to_l2_record(self) -> dict[str, object]:
        action = self.decision.selected_action
        realized_delta = self.realized_belief_delta
        return {
            "belief_before_id": self.belief_before_id,
            "action": {
                "type": action.action_type.value,
                "source_id": action.source_id,
                "target_ids": list(action.target_ids),
                "relation": action.relation,
            },
            "real_observation_ids": list(self.observation_ids),
            "belief_after_id": self.belief_after_id,
            "selected_trajectory_id": self.decision.selected_trajectory_id,
            "preference_output": "ordinal_only",
            "trajectory_preferences": [
                {
                    "left_id": comparison.left_id,
                    "right_id": comparison.right_id,
                    "label": comparison.label.value,
                    "rationale": comparison.rationale,
                }
                for comparison in self.decision.comparisons
            ],
            "realized_belief_delta": (
                {
                    "resolved_roles": list(realized_delta.resolved_roles),
                    "relation_updates": list(realized_delta.relation_updates),
                    "contradiction_updates": list(
                        realized_delta.contradiction_updates
                    ),
                    "uncertainty_change": realized_delta.uncertainty_change.value,
                    "answerability_after": realized_delta.answerability_after.value,
                    "predicted_only": realized_delta.predicted_only,
                }
                if realized_delta is not None
                else None
            ),
            "skill_invocation": self.skill_invocation,
            "belief_update_audit": self.belief_update_audit,
        }


@dataclass(frozen=True)
class NavigationRun:
    initial_belief_id: str
    final_belief: BeliefSnapshot
    steps: tuple[NavigationStep, ...] = ()
    belief_snapshots: tuple[BeliefSnapshot, ...] = ()

    def to_l2_rollout(self) -> list[dict[str, object]]:
        return [step.to_l2_record() for step in self.steps]


class BeliefBackend(Protocol):
    """Replaceable seam for factorized or future factor-graph inference."""

    def initialize(
        self,
        question: str,
        overlay: CausalTemporalOverlay,
        *,
        seed_evidence: tuple[str, ...] = (),
        missing_roles: tuple[str, ...] | None = None,
        graph_read_budget: int = 8,
    ) -> BeliefSnapshot: ...

    def update(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        observations: list[MemoryNode],
        overlay: CausalTemporalOverlay,
    ) -> BeliefUpdateResult: ...


@dataclass(frozen=True)
class GraphReadExecution:
    observations: tuple[MemoryNode, ...]
    skill_invocation: dict[str, object]


class GraphReadExecutor(Protocol):
    def execute(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> GraphReadExecution: ...


class ObservationBeliefWorldModel(Protocol):
    def predict(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> PredictedTransition: ...


class TrajectoryPreferenceModel(Protocol):
    def compare(
        self,
        left: TrajectoryPrediction,
        right: TrajectoryPrediction,
        belief: BeliefSnapshot,
    ) -> PairwisePreference: ...
