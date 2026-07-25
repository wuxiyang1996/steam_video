"""Strict categorical contracts for full retained-graph IWM navigation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

from memory_graph.correlation_overlay import CorrelationEdge
from memory_graph.multichannel_correlation import CandidateNavigationEdge
from memory_graph.soft_correlation import SoftNavigationCorrelation
from memory_graph.types import MemoryNode


class ActionKind(str, Enum):
    START_AT = "start_at"
    TEMPORAL_FORWARD = "temporal_forward"
    TEMPORAL_BACKWARD = "temporal_backward"
    FOLLOW_CORRELATION = "follow_correlation"
    BACKTRACK = "backtrack"
    STOP = "stop"
    ANSWER = "answer"
    ABSTAIN = "abstain"


class AnswerabilityState(str, Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    ABSTAIN = "abstain"


class EvidenceOutcome(str, Enum):
    SUPPORT = "support"
    COUNTEREVIDENCE = "counterevidence"
    IDENTITY_EVIDENCE = "identity_evidence"
    STATE_EVIDENCE = "state_evidence"
    BRIDGE_EVIDENCE = "bridge_evidence"
    EMPTY = "empty"
    INCONCLUSIVE = "inconclusive"


class ProgressChange(str, Enum):
    ADVANCED = "advanced"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"


class ContradictionChange(str, Enum):
    OPENED = "opened"
    RESOLVED = "resolved"
    UNCHANGED = "unchanged"


class FrontierChange(str, Enum):
    OPENED = "opened"
    CLOSED = "closed"
    SHIFTED = "shifted"
    UNCHANGED = "unchanged"


class PreferenceLabel(str, Enum):
    PREFER_LEFT = "prefer_left"
    TIE = "tie"
    PREFER_RIGHT = "prefer_right"
    INCOMPARABLE = "incomparable"


@dataclass(frozen=True)
class NodeKey:
    node_id: str
    node_type: str
    start_s: float
    end_s: float
    semantic_key: str
    embedding_ref: dict[str, Any] | None
    structural_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemporalNavigationEdge:
    edge_id: str
    src: str
    dst: str
    relation: str

    def __post_init__(self) -> None:
        if self.relation not in {"temporal_next", "before", "overlaps", "during"}:
            raise ValueError(f"unsupported temporal relation: {self.relation}")
        if not self.edge_id or not self.src or not self.dst or self.src == self.dst:
            raise ValueError("temporal navigation edge is invalid")


@dataclass(frozen=True)
class RetainedEvidenceGraph:
    graph_id: str
    nodes: tuple[MemoryNode, ...]
    temporal_edges: tuple[TemporalNavigationEdge, ...]
    correlation_edges: tuple[SoftNavigationCorrelation, ...]
    capacity: int
    candidate_edges: tuple[CandidateNavigationEdge, ...] = ()
    verified_relations: tuple[CorrelationEdge, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capacity < 1 or len(self.nodes) > self.capacity:
            raise ValueError("retained graph violates fixed capacity")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("retained graph contains duplicate nodes")
        known = set(node_ids)
        if any(
            edge.src not in known or edge.dst not in known
            for edge in self.temporal_edges
        ):
            raise ValueError("retained graph contains an orphan temporal edge")
        if any(
            edge.src not in known or edge.dst not in known
            for edge in self.correlation_edges
        ):
            raise ValueError("retained graph contains an orphan correlation edge")
        if any(
            edge.src not in known or edge.dst not in known
            for edge in self.candidate_edges
        ):
            raise ValueError("retained graph contains an orphan candidate edge")
        if any(
            edge.src not in known or edge.dst not in known
            for edge in self.verified_relations
        ):
            raise ValueError("retained graph contains an orphan verified relation")

    @property
    def node_by_id(self) -> dict[str, MemoryNode]:
        return {node.node_id: node for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "steam-l1-l1.5-navigation-graph/v0.1",
            "graph_id": self.graph_id,
            "capacity": self.capacity,
            "nodes": [node.to_dict() for node in self.nodes],
            "temporal_edges": [asdict(edge) for edge in self.temporal_edges],
            "correlation_edges": [edge.to_dict() for edge in self.correlation_edges],
            "candidate_edges": [edge.to_dict() for edge in self.candidate_edges],
            "verified_relations": [edge.to_dict() for edge in self.verified_relations],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CursorBeliefState:
    belief_id: str
    question: str
    current_node_id: str | None = None
    localized_entry_node_ids: tuple[str, ...] = ()
    acquired_evidence: tuple[str, ...] = ()
    imagined_evidence: tuple[str, ...] = ()
    cursor_history: tuple[str, ...] = ()
    accepted_relations: tuple[str, ...] = ()
    rejected_relations: tuple[str, ...] = ()
    unresolved_relations: tuple[str, ...] = ()
    required_roles: tuple[str, ...] = ()
    missing_roles: tuple[str, ...] = ()
    grounded_role_evidence: tuple[tuple[str, str], ...] = ()
    contradictions: tuple[str, ...] = ()
    answerability: AnswerabilityState = AnswerabilityState.NOT_READY
    remaining_reads: int = 8
    step: int = 0

    def __post_init__(self) -> None:
        if self.remaining_reads < 0:
            raise ValueError("remaining_reads must be non-negative")
        if len(self.localized_entry_node_ids) != len(
            set(self.localized_entry_node_ids)
        ):
            raise ValueError("localized entry node IDs must be unique")
        if (
            self.current_node_id is not None
            and self.current_node_id not in self.acquired_evidence
        ):
            raise ValueError("current cursor must point to acquired evidence")
        if not set(self.imagined_evidence).issubset(self.acquired_evidence):
            raise ValueError("imagined evidence must be a subset of acquired addresses")
        if len(self.required_roles) != len(set(self.required_roles)):
            raise ValueError("required roles must be unique")
        if not set(self.missing_roles).issubset(
            self.required_roles or self.missing_roles
        ):
            raise ValueError("missing roles must belong to required roles")
        bound_roles: set[str] = set()
        for role, node_id in self.grounded_role_evidence:
            if role in bound_roles:
                raise ValueError("a grounded role may have only one evidence binding")
            if self.required_roles and role not in self.required_roles:
                raise ValueError("grounded role must belong to required roles")
            if (
                node_id not in self.acquired_evidence
                or node_id in self.imagined_evidence
            ):
                raise ValueError("grounded role must cite acquired real evidence")
            bound_roles.add(role)


@dataclass(frozen=True)
class LegalGraphAction:
    action_id: str
    kind: ActionKind
    source_id: str | None = None
    target_id: str | None = None
    edge_id: str | None = None
    relation: str | None = None
    reads_evidence: bool = False

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must not be empty")
        if (
            self.kind
            in {
                ActionKind.START_AT,
                ActionKind.TEMPORAL_FORWARD,
                ActionKind.TEMPORAL_BACKWARD,
                ActionKind.FOLLOW_CORRELATION,
                ActionKind.BACKTRACK,
            }
            and self.target_id is None
        ):
            raise ValueError(f"{self.kind.value} requires a target")


@dataclass(frozen=True)
class NodeModelView:
    key: NodeKey
    acquired: bool
    evidence_value: str | None = None

    def __post_init__(self) -> None:
        if not self.acquired and self.evidence_value is not None:
            raise ValueError("unread target evidence leaked into model input")


@dataclass(frozen=True)
class IWMGraphInput:
    question: str
    current_node_id: str | None
    nodes: tuple[NodeModelView, ...]
    temporal_edges: tuple[TemporalNavigationEdge, ...]
    correlation_edges: tuple[SoftNavigationCorrelation, ...]
    candidate_edges: tuple[CandidateNavigationEdge, ...]
    verified_relations: tuple[CorrelationEdge, ...]
    legal_actions: tuple[LegalGraphAction, ...]
    acquired_evidence: tuple[str, ...]
    missing_roles: tuple[str, ...]
    contradictions: tuple[str, ...]
    answerability: AnswerabilityState
    required_roles: tuple[str, ...] = ()
    grounded_role_evidence: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PredictedObservation:
    target_id: str | None
    outcome: EvidenceOutcome
    descriptor: tuple[str, ...] = ()
    predicted_only: bool = True


@dataclass(frozen=True)
class CategoricalBeliefDelta:
    progress: ProgressChange
    answerability_after: AnswerabilityState
    frontier_change: FrontierChange = FrontierChange.UNCHANGED
    contradiction_change: ContradictionChange = ContradictionChange.UNCHANGED
    resolved_roles: tuple[str, ...] = ()
    opened_roles: tuple[str, ...] = ()
    relation_updates: tuple[str, ...] = ()
    predicted_only: bool = True


@dataclass(frozen=True)
class IWMRequest:
    belief: CursorBeliefState
    graph_input: IWMGraphInput
    action: LegalGraphAction
    parent_action_ids: tuple[str, ...] = ()
    imagined_history: tuple["ImaginedTransition", ...] = ()


@dataclass(frozen=True)
class ImaginedTransition:
    action: LegalGraphAction
    observation: PredictedObservation
    belief_delta: CategoricalBeliefDelta
    structured_patch: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action.target_id != self.observation.target_id:
            raise ValueError("imagined observation target must match its action")
        if not isinstance(self.structured_patch, dict):
            raise ValueError("imagined structured patch must be an object")


@dataclass(frozen=True)
class TrajectoryPrediction:
    trajectory_id: str
    transitions: tuple[ImaginedTransition, ...]

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.transitions:
            raise ValueError("trajectory must contain imagined transitions")

    @property
    def first_action(self) -> LegalGraphAction:
        return self.transitions[0].action


@dataclass(frozen=True)
class TrajectoryPair:
    left: TrajectoryPrediction
    right: TrajectoryPrediction


@dataclass(frozen=True)
class TrajectoryPreference:
    left_id: str
    right_id: str
    label: PreferenceLabel
    rationale: str = ""


class BatchedCategoricalWorldModel(Protocol):
    model_name: str

    def predict_batch(
        self,
        requests: Sequence[IWMRequest],
    ) -> Sequence[ImaginedTransition]: ...


class BatchedTrajectoryPreferenceModel(Protocol):
    model_name: str

    def compare_batch(
        self,
        pairs: Sequence[TrajectoryPair],
        belief: CursorBeliefState,
    ) -> Sequence[TrajectoryPreference]: ...


@dataclass(frozen=True)
class FullGraphPlanDecision:
    selected_action: LegalGraphAction
    planning_status: str
    trajectories: tuple[TrajectoryPrediction, ...]
    preferences: tuple[TrajectoryPreference, ...]
    undominated_trajectory_ids: tuple[str, ...]
    legal_action_count: int
    initial_trajectories: tuple[TrajectoryPrediction, ...] = ()
    top_k_applied: bool = False

    def __post_init__(self) -> None:
        if self.top_k_applied:
            raise ValueError("full-graph main arm cannot apply Top-K pruning")


@dataclass(frozen=True)
class GraphActionExecution:
    action: LegalGraphAction
    observation: MemoryNode | None
    updated_belief: CursorBeliefState
    reread: bool
