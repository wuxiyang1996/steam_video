"""Persistent multi-reasoning-path planner contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from ..navigation.contracts import NavigationAction
from ..world_model.contracts import BeliefEffect, BeliefState, ObservationPrediction


class PathStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CONTRADICTED = "contradicted"
    COMPLETED = "completed"


@dataclass(frozen=True)
class ReasoningPath:
    path_id: str
    hypothesis: str
    belief: BeliefState
    cursor_id: str | None = None
    action_history: tuple[str, ...] = ()
    interpreted_observation_ids: tuple[str, ...] = ()
    parent_path_id: str | None = None
    status: PathStatus = PathStatus.ACTIVE
    pending_first_action_id: str | None = None
    pending_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.path_id or not self.hypothesis.strip():
            raise ValueError("reasoning path requires an ID and hypothesis")
        if self.cursor_id is not None and not self.action_history:
            raise ValueError("path cursor requires an executed action history")
        if self.status is PathStatus.SUSPENDED and not self.pending_first_action_id:
            raise ValueError("suspended path requires a pending first action")


@dataclass(frozen=True)
class SharedEvidenceState:
    acquired_ids: tuple[str, ...] = ()
    remaining_reads: int = 0

    def __post_init__(self) -> None:
        if self.remaining_reads < 0:
            raise ValueError("shared read budget must be non-negative")
        if len(self.acquired_ids) != len(set(self.acquired_ids)):
            raise ValueError("shared evidence IDs must be unique")


@dataclass(frozen=True)
class ReasoningPathForest:
    forest_id: str
    paths: tuple[ReasoningPath, ...]
    shared_evidence: SharedEvidenceState
    step: int = 0
    top_k_applied: bool = False

    def __post_init__(self) -> None:
        if not self.forest_id or not self.paths:
            raise ValueError("reasoning forest must be non-empty")
        if self.top_k_applied:
            raise ValueError("reasoning v2 cannot apply Top-K path pruning")
        ids = [row.path_id for row in self.paths]
        if len(ids) != len(set(ids)):
            raise ValueError("reasoning forest contains duplicate paths")

    @property
    def plannable(self) -> tuple[ReasoningPath, ...]:
        return tuple(
            row
            for row in self.paths
            if row.status in {PathStatus.ACTIVE, PathStatus.SUSPENDED}
        )


@dataclass(frozen=True)
class ImaginedReasoningStep:
    action: NavigationAction
    observation: ObservationPrediction
    effect: BeliefEffect


@dataclass(frozen=True)
class CandidateReasoningTrajectory:
    candidate_id: str
    root_path_id: str
    hypothesis: str
    steps: tuple[ImaginedReasoningStep, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.steps:
            raise ValueError("candidate reasoning trajectory must be non-empty")

    @property
    def first_action(self) -> NavigationAction:
        return self.steps[0].action


@dataclass(frozen=True)
class JointActionTree:
    """All hypothesis-conditioned futures sharing one executable first read."""

    tree_id: str
    first_action: NavigationAction
    trajectories: tuple[CandidateReasoningTrajectory, ...]

    def __post_init__(self) -> None:
        if not self.tree_id or not self.trajectories:
            raise ValueError("joint action tree must be non-empty")
        if any(
            row.first_action.shared_key != self.first_action.shared_key
            for row in self.trajectories
        ):
            raise ValueError("joint tree mixes different physical first actions")

    @property
    def covered_hypotheses(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.hypothesis for row in self.trajectories))


class PreferenceStatus(str, Enum):
    SELECT = "select"
    TIE = "tie"
    INCOMPARABLE = "incomparable"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class TrajectoryPreferenceDecision:
    status: PreferenceStatus
    frontier_candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.status is PreferenceStatus.SELECT:
            if self.selected_candidate_id not in self.frontier_candidate_ids:
                raise ValueError("selected trajectory must belong to the frontier")
        elif self.selected_candidate_id is not None:
            raise ValueError("only a select decision may execute a candidate")


class CategoricalTrajectoryPlannerModel(Protocol):
    model_name: str

    def choose(
        self,
        forest: ReasoningPathForest,
        candidates: Sequence[JointActionTree],
    ) -> TrajectoryPreferenceDecision: ...


@dataclass(frozen=True)
class PlanningDecision:
    candidates: tuple[JointActionTree, ...]
    preference: TrajectoryPreferenceDecision
    selected_action: NavigationAction | None
    observation_prediction_count: int
    effect_prediction_count: int
    horizon: int
    top_k_applied: bool = False

    def __post_init__(self) -> None:
        if self.top_k_applied:
            raise ValueError("reasoning v2 planning cannot apply Top-K")
        if self.horizon not in {1, 2}:
            raise ValueError("reasoning v2 horizon must be one or two")
