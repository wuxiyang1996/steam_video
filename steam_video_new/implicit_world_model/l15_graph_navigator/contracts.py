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


class HypothesisDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class FrontierChange(str, Enum):
    OPENED = "opened"
    UNCHANGED = "unchanged"
    CLOSED = "closed"


class ContradictionChange(str, Enum):
    OPENED = "opened"
    RESOLVED = "resolved"
    UNCHANGED = "unchanged"


class PathChange(str, Enum):
    OPENED = "opened"
    BLOCKED = "blocked"
    UNCHANGED = "unchanged"


class RecoveryStatus(str, Enum):
    RECOVERED = "recovered"
    STALLED = "stalled"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class HypothesisUpdate:
    edge_id: str
    disposition: HypothesisDisposition


class ReasoningHopType(str, Enum):
    """Semantic next-hop operators for multi-hop reasoning.

    ``GraphReadAction`` remains the execution compatibility contract.  This
    enum makes clear that planning selects evidence/reasoning hops, not
    physical robot actions.
    """

    READ_EVENT = "read_event"
    FOLLOW_TEMPORAL = "follow_temporal"
    FOLLOW_DEPENDENCY = "follow_dependency"
    RESOLVE_IDENTITY = "resolve_identity"
    INSPECT_STATE_DELTA = "inspect_state_delta"
    SEEK_COUNTER_EVIDENCE = "seek_counter_evidence"
    VERIFY_RELATION = "verify_relation"
    STOP_AND_ANSWER = "stop_and_answer"


@dataclass(frozen=True)
class ReasoningHop:
    hop_type: ReasoningHopType
    action: GraphReadAction


@dataclass(frozen=True)
class ReasoningContextBudget:
    max_nodes: int = 24
    max_edges: int = 32
    max_candidate_hops: int = 8
    max_comparisons: int = 32
    recent_hop_window: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("max_nodes", self.max_nodes),
            ("max_edges", self.max_edges),
            ("max_candidate_hops", self.max_candidate_hops),
            ("max_comparisons", self.max_comparisons),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.recent_hop_window < 0:
            raise ValueError("recent_hop_window must be non-negative")


@dataclass(frozen=True)
class ReasoningContextAudit:
    retrieval_mode: str
    retrieved_node_ids: tuple[str, ...]
    dropped_node_ids: tuple[str, ...]
    retrieved_edge_ids: tuple[str, ...]
    dropped_edge_ids: tuple[str, ...]
    retained_candidate_hops: int
    dropped_candidate_hops: int
    comparison_budget: int
    estimated_prompt_tokens: int
    embedding_models_available: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningContext:
    """Bounded, categorical planner input; never contains raw embeddings."""

    question: str
    answerability: Answerability
    missing_roles: tuple[str, ...]
    accepted_hypotheses: tuple[str, ...]
    rejected_hypotheses: tuple[str, ...]
    unresolved_hypotheses: tuple[str, ...]
    contradictions: tuple[str, ...]
    acquired_evidence_refs: tuple[str, ...]
    recent_hops: tuple[str, ...]
    local_node_ids: tuple[str, ...]
    local_edge_ids: tuple[str, ...]
    candidate_hops: tuple[ReasoningHop, ...]
    audit: ReasoningContextAudit


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
    hypothesis_updates: tuple[HypothesisUpdate, ...] = ()
    frontier_change: FrontierChange = FrontierChange.UNCHANGED
    contradiction_change: ContradictionChange = ContradictionChange.UNCHANGED
    path_change: PathChange = PathChange.UNCHANGED
    recovery_status: RecoveryStatus = RecoveryStatus.UNCHANGED
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
    fallback_policy: str = "explicit_abstain_for_non_unique_undominated"
    reasoning_context: ReasoningContext | None = None
    planning_status: str = "selected"
    ambiguity_reason: str | None = None

    @property
    def selected_hop(self) -> ReasoningHop:
        return reasoning_hop_from_action(self.selected_action)


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
            "reasoning_hop": self.decision.selected_hop.hop_type.value,
            "action": {
                "type": action.action_type.value,
                "source_id": action.source_id,
                "target_ids": list(action.target_ids),
                "relation": action.relation,
            },
            "real_observation_ids": list(self.observation_ids),
            "belief_after_id": self.belief_after_id,
            "selected_trajectory_id": self.decision.selected_trajectory_id,
            "planning_status": self.decision.planning_status,
            "ambiguity_reason": self.decision.ambiguity_reason,
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
                    "hypothesis_updates": [
                        {
                            "edge_id": update.edge_id,
                            "disposition": update.disposition.value,
                        }
                        for update in realized_delta.hypothesis_updates
                    ],
                    "frontier_change": realized_delta.frontier_change.value,
                    "contradiction_change": (
                        realized_delta.contradiction_change.value
                    ),
                    "path_change": realized_delta.path_change.value,
                    "recovery_status": realized_delta.recovery_status.value,
                    "uncertainty_change": realized_delta.uncertainty_change.value,
                    "answerability_after": realized_delta.answerability_after.value,
                    "predicted_only": realized_delta.predicted_only,
                }
                if realized_delta is not None
                else None
            ),
            "skill_invocation": self.skill_invocation,
            "belief_update_audit": self.belief_update_audit,
            "reasoning_context_audit": (
                {
                    **reasoning_context_audit_to_dict(
                        self.decision.reasoning_context.audit
                    ),
                    "actual_comparisons": len(self.decision.comparisons),
                    "predicted_trajectories": len(self.decision.trajectories),
                }
                if self.decision.reasoning_context is not None
                else None
            ),
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


def reasoning_hop_from_action(action: GraphReadAction) -> ReasoningHop:
    """Map the legacy execution action to its multi-hop reasoning meaning."""

    from memory_graph.navigation import NavigationActionType

    mapping = {
        NavigationActionType.SEMANTIC: ReasoningHopType.READ_EVENT,
        NavigationActionType.TEMPORAL_BACK: ReasoningHopType.FOLLOW_TEMPORAL,
        NavigationActionType.TEMPORAL_FORWARD: ReasoningHopType.FOLLOW_TEMPORAL,
        NavigationActionType.TRACK_ENTITY: ReasoningHopType.RESOLVE_IDENTITY,
        NavigationActionType.INSPECT_STATE_CHANGE: ReasoningHopType.INSPECT_STATE_DELTA,
        NavigationActionType.FOLLOW_DEPENDENCY: ReasoningHopType.FOLLOW_DEPENDENCY,
        NavigationActionType.CANDIDATE_CAUSE: ReasoningHopType.FOLLOW_DEPENDENCY,
        NavigationActionType.EFFECT: ReasoningHopType.FOLLOW_DEPENDENCY,
        NavigationActionType.FIND_BRIDGE: ReasoningHopType.FOLLOW_DEPENDENCY,
        NavigationActionType.SEARCH_COUNTEREVIDENCE: ReasoningHopType.SEEK_COUNTER_EVIDENCE,
        NavigationActionType.VERIFY: ReasoningHopType.VERIFY_RELATION,
        NavigationActionType.STOP: ReasoningHopType.STOP_AND_ANSWER,
    }
    return ReasoningHop(mapping[action.action_type], action)


def reasoning_context_audit_to_dict(
    audit: ReasoningContextAudit,
) -> dict[str, object]:
    return {
        "retrieval_mode": audit.retrieval_mode,
        "retrieved_node_ids": list(audit.retrieved_node_ids),
        "dropped_node_ids": list(audit.dropped_node_ids),
        "retrieved_edge_ids": list(audit.retrieved_edge_ids),
        "dropped_edge_ids": list(audit.dropped_edge_ids),
        "retained_candidate_hops": audit.retained_candidate_hops,
        "dropped_candidate_hops": audit.dropped_candidate_hops,
        "comparison_budget": audit.comparison_budget,
        "estimated_prompt_tokens": audit.estimated_prompt_tokens,
        "embedding_models_available": list(audit.embedding_models_available),
    }
