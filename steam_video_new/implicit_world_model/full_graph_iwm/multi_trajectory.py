"""Direct categorical IWM planning over persistent competing trajectories."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
from typing import Any, Protocol, Sequence

from memory_graph.types import MemoryNode

from .action_compiler import GraphActionCompiler, execute_graph_action
from .contracts import (
    ActionKind,
    CursorBeliefState,
    LegalGraphAction,
    RetainedEvidenceGraph,
)
from .model_input import build_iwm_graph_input


class TrajectoryStatus(str, Enum):
    ACTIVE = "active"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    SUSPENDED = "suspended"
    MERGED = "merged"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class PoolPreferenceStatus(str, Enum):
    UNIQUE = "unique"
    TIE = "tie"
    INCOMPARABLE = "incomparable"


class EvidenceEffect(str, Enum):
    SUPPORT = "support"
    COUNTEREVIDENCE = "counterevidence"
    INCONCLUSIVE = "inconclusive"
    COMPLETE = "complete"


EXPANDABLE_STATUSES = {
    TrajectoryStatus.ACTIVE,
    TrajectoryStatus.SUPPORTED,
    TrajectoryStatus.INCONCLUSIVE,
}


@dataclass(frozen=True)
class ReasoningTrajectory:
    trajectory_id: str
    hypothesis: str
    belief: CursorBeliefState
    parent_trajectory_id: str | None = None
    action_history: tuple[str, ...] = ()
    shared_observation_ids: tuple[str, ...] = ()
    status: TrajectoryStatus = TrajectoryStatus.ACTIVE
    merged_into: str | None = None
    lifecycle_rationale: str = ""

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.hypothesis.strip():
            raise ValueError("trajectory ID and hypothesis must be non-empty")
        if self.status is TrajectoryStatus.MERGED and not self.merged_into:
            raise ValueError("merged trajectory requires merged_into")
        if self.status is not TrajectoryStatus.MERGED and self.merged_into is not None:
            raise ValueError("only merged trajectories may set merged_into")
        if not set(self.shared_observation_ids).issubset(self.belief.acquired_evidence):
            raise ValueError("shared observations must be acquired real evidence")


@dataclass(frozen=True)
class TrajectoryPool:
    pool_id: str
    trajectories: tuple[ReasoningTrajectory, ...]
    top_k_applied: bool = False

    def __post_init__(self) -> None:
        if not self.pool_id or not self.trajectories:
            raise ValueError("trajectory pool must be non-empty")
        if self.top_k_applied:
            raise ValueError("multi-trajectory main method cannot apply Top-K")
        ids = [row.trajectory_id for row in self.trajectories]
        if len(ids) != len(set(ids)):
            raise ValueError("trajectory IDs must be unique")
        expandable = self.expandable
        if expandable:
            acquired = set(expandable[0].belief.acquired_evidence)
            remaining = expandable[0].belief.remaining_reads
            if any(
                set(row.belief.acquired_evidence) != acquired
                or row.belief.remaining_reads != remaining
                for row in expandable[1:]
            ):
                raise ValueError(
                    "expandable trajectories must share real evidence and read budget"
                )

    @property
    def expandable(self) -> tuple[ReasoningTrajectory, ...]:
        return tuple(
            row for row in self.trajectories if row.status in EXPANDABLE_STATUSES
        )


@dataclass(frozen=True)
class TrajectoryExpansion:
    expansion_id: str
    trajectory_id: str
    action: LegalGraphAction


@dataclass(frozen=True)
class PredictedLifecycle:
    trajectory_id: str
    status: TrajectoryStatus
    rationale: str = ""
    predicted_only: bool = True

    def __post_init__(self) -> None:
        if self.status in {
            TrajectoryStatus.CONTRADICTED,
            TrajectoryStatus.COMPLETED,
            TrajectoryStatus.ABANDONED,
            TrajectoryStatus.MERGED,
        }:
            raise ValueError(
                "imagined lifecycle cannot terminate a persistent trajectory"
            )


@dataclass(frozen=True)
class MultiTrajectoryIWMDecision:
    status: PoolPreferenceStatus
    preferred_expansion_ids: tuple[str, ...]
    lifecycle_predictions: tuple[PredictedLifecycle, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class MultiTrajectoryPlanDecision:
    selected_action: LegalGraphAction
    planning_status: str
    expansions: tuple[TrajectoryExpansion, ...]
    preferred_expansion_ids: tuple[str, ...]
    lifecycle_predictions: tuple[PredictedLifecycle, ...]
    legal_expansion_count: int
    top_k_applied: bool = False
    imagined_paths: tuple[Any, ...] = ()
    preferred_path_ids: tuple[str, ...] = ()
    planning_horizon: int = 0
    preference_audit: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.top_k_applied:
            raise ValueError("multi-trajectory planner cannot apply Top-K")
        if self.planning_horizon not in {0, 1, 2}:
            raise ValueError(
                "multi-trajectory planning horizon must be zero, one, or two"
            )


@dataclass(frozen=True)
class TrajectoryEvidenceAssessment:
    trajectory_id: str
    effect: EvidenceEffect
    rationale: str = ""


@dataclass(frozen=True)
class MultiTrajectoryExecution:
    action: LegalGraphAction
    observation: MemoryNode | None
    pool: TrajectoryPool
    assessments: tuple[TrajectoryEvidenceAssessment, ...]


@dataclass(frozen=True)
class MultiTrajectoryTraceStep:
    pool_before: TrajectoryPool
    decision: MultiTrajectoryPlanDecision
    observation_id: str | None
    assessments: tuple[TrajectoryEvidenceAssessment, ...]
    pool_after: TrajectoryPool


@dataclass(frozen=True)
class MultiTrajectoryRunTrace:
    initial_pool: TrajectoryPool
    final_pool: TrajectoryPool
    steps: tuple[MultiTrajectoryTraceStep, ...]
    termination: str


class DirectMultiTrajectoryIWM(Protocol):
    model_name: str

    def prefer(
        self,
        pool: TrajectoryPool,
        expansions: Sequence[TrajectoryExpansion],
        graph: RetainedEvidenceGraph,
    ) -> MultiTrajectoryIWMDecision: ...


class RealBeliefUpdater(Protocol):
    def update(
        self, belief: CursorBeliefState, observation: MemoryNode
    ) -> CursorBeliefState: ...


class ActionAwareRealBeliefUpdater(Protocol):
    def update_after_action(
        self,
        trajectory_id: str,
        previous_belief: CursorBeliefState,
        structurally_updated_belief: CursorBeliefState,
        action: LegalGraphAction,
        observation: MemoryNode,
        graph: RetainedEvidenceGraph,
    ) -> CursorBeliefState: ...


class HypothesisAwareRealBeliefUpdater(Protocol):
    def update_for_trajectory(
        self,
        trajectory_id: str,
        hypothesis: str,
        previous_belief: CursorBeliefState,
        structurally_updated_belief: CursorBeliefState,
        action: LegalGraphAction,
        observation: MemoryNode,
        graph: RetainedEvidenceGraph,
    ) -> CursorBeliefState: ...


class TrajectoryEvidenceAssessor(Protocol):
    def assess_batch(
        self,
        trajectories: Sequence[ReasoningTrajectory],
        observation: MemoryNode,
        corrected_beliefs: Sequence[CursorBeliefState],
    ) -> Sequence[TrajectoryEvidenceAssessment]: ...


class MultiTrajectoryIWMPlanner:
    """Compile every trajectory expansion and defer preference to the IWM."""

    def __init__(
        self,
        iwm: DirectMultiTrajectoryIWM,
        *,
        action_compiler: GraphActionCompiler | None = None,
    ) -> None:
        self.iwm = iwm
        self.action_compiler = action_compiler or GraphActionCompiler()

    def plan(
        self,
        pool: TrajectoryPool,
        graph: RetainedEvidenceGraph,
    ) -> MultiTrajectoryPlanDecision:
        expansions = self._compile_expansions(pool, graph)
        if not expansions:
            raise RuntimeError("trajectory pool has no legal expansions")
        imagined = self.iwm.prefer(pool, expansions, graph)
        known = {row.expansion_id: row for row in expansions}
        preferred = imagined.preferred_expansion_ids
        if len(preferred) != len(set(preferred)) or not set(preferred) <= set(known):
            raise ValueError("IWM returned invalid preferred expansion IDs")
        if imagined.status is PoolPreferenceStatus.UNIQUE and len(preferred) != 1:
            raise ValueError("unique IWM preference requires one expansion")
        if imagined.status is PoolPreferenceStatus.TIE and len(preferred) < 2:
            raise ValueError("tied IWM preference requires multiple expansions")
        if imagined.status is PoolPreferenceStatus.INCOMPARABLE and preferred:
            raise ValueError("incomparable IWM preference cannot select expansions")
        lifecycle_ids = [row.trajectory_id for row in imagined.lifecycle_predictions]
        if len(lifecycle_ids) != len(set(lifecycle_ids)) or not set(lifecycle_ids) <= {
            row.trajectory_id for row in pool.trajectories
        }:
            raise ValueError("IWM returned invalid lifecycle predictions")

        preferred_actions = {
            shared_action_key(known[expansion_id].action): known[expansion_id].action
            for expansion_id in preferred
        }
        if len(preferred_actions) == 1:
            selected = next(iter(preferred_actions.values()))
            status = (
                "selected_unique_expansion"
                if len(preferred) == 1
                else "selected_shared_action_across_trajectories"
            )
        else:
            selected = _shared_abstain(expansions)
            status = (
                "abstain_incomparable_trajectory_expansions"
                if not preferred
                else "abstain_distinct_preferred_actions"
            )
        return MultiTrajectoryPlanDecision(
            selected_action=selected,
            planning_status=status,
            expansions=expansions,
            preferred_expansion_ids=preferred,
            lifecycle_predictions=imagined.lifecycle_predictions,
            legal_expansion_count=len(expansions),
            top_k_applied=False,
        )

    def _compile_expansions(
        self,
        pool: TrajectoryPool,
        graph: RetainedEvidenceGraph,
    ) -> tuple[TrajectoryExpansion, ...]:
        rows: list[TrajectoryExpansion] = []
        for trajectory in pool.expandable:
            for action in self.action_compiler.compile(trajectory.belief, graph):
                digest = hashlib.sha256(
                    f"{trajectory.trajectory_id}\x1f{action.action_id}".encode("utf-8")
                ).hexdigest()[:20]
                rows.append(
                    TrajectoryExpansion(
                        expansion_id=f"expansion:{digest}",
                        trajectory_id=trajectory.trajectory_id,
                        action=action,
                    )
                )
        return tuple(rows)


class GPTOSSMultiTrajectoryIWM:
    """Directly prefer hypothesis-conditioned expansions with categorical output."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))

    def prefer(
        self,
        pool: TrajectoryPool,
        expansions: Sequence[TrajectoryExpansion],
        graph: RetainedEvidenceGraph,
    ) -> MultiTrajectoryIWMDecision:
        trajectory_by_id = {row.trajectory_id: row for row in pool.expandable}
        aliases = {
            row.expansion_id: _alphabetic_alias("candidate", index)
            for index, row in enumerate(expansions)
        }
        trajectory_aliases = {
            row.trajectory_id: _alphabetic_alias("trajectory", index)
            for index, row in enumerate(pool.expandable)
        }
        action_sets = {
            trajectory_id: tuple(
                row.action for row in expansions if row.trajectory_id == trajectory_id
            )
            for trajectory_id in trajectory_by_id
        }
        graph_inputs = {
            trajectory_id: build_iwm_graph_input(
                trajectory.belief,
                graph,
                action_sets[trajectory_id],
            )
            for trajectory_id, trajectory in trajectory_by_id.items()
        }
        payload = {
            "question": pool.expandable[0].belief.question,
            "trajectory_pool": {
                trajectory_aliases[trajectory_id]: _trajectory_payload(
                    trajectory,
                    graph_inputs[trajectory_id],
                )
                for trajectory_id, trajectory in trajectory_by_id.items()
            },
            "candidate_expansions": {
                aliases[row.expansion_id]: _expansion_payload(
                    row,
                    trajectory_aliases[row.trajectory_id],
                    graph_inputs[row.trajectory_id],
                )
                for row in expansions
            },
            "allowed_preference_status": [
                value.value for value in PoolPreferenceStatus
            ],
            "allowed_lifecycle": [
                TrajectoryStatus.ACTIVE.value,
                TrajectoryStatus.SUPPORTED.value,
                TrajectoryStatus.INCONCLUSIVE.value,
                TrajectoryStatus.SUSPENDED.value,
            ],
            "required_output": {
                "only_keys": [
                    "status",
                    "preferred",
                    "trajectory_lifecycle",
                    "rationale",
                ],
                "preferred": "exact candidate aliases",
                "trajectory_lifecycle": {
                    alias: {
                        "status": "one allowed non-terminal lifecycle",
                        "rationale": "short categorical rationale",
                    }
                    for alias in trajectory_aliases.values()
                },
            },
            "required_contract": {
                "jointly_compare_all_candidates": True,
                "temporal_semantic_correlation_and_belief_context_present": True,
                "lifecycle_is_imagined_and_non_terminal": True,
                "no_top_k_or_beam": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        for attempt in range(2):
            result = self.client.complete_json(
                task=(
                    (
                        "Jointly compare every legal expansion of every competing "
                        "reasoning trajectory. Return categorical preference and "
                        "non-terminal imagined lifecycle suggestions. Prefer multiple "
                        "aliases when tied; preserve incomparability. Do not rank with "
                        "numbers."
                    )
                    if attempt == 0
                    else (
                        "Repair the response to exactly match the categorical schema, "
                        "cover every trajectory, and use no numeric values."
                    )
                ),
                payload=payload,
            )
            try:
                return _parse_multi_trajectory_iwm_result(
                    result,
                    aliases=aliases,
                    trajectory_aliases=trajectory_aliases,
                )
            except ValueError:
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable multi-trajectory IWM repair state")


def _parse_multi_trajectory_iwm_result(
    result: dict[str, Any],
    *,
    aliases: dict[str, str],
    trajectory_aliases: dict[str, str],
) -> MultiTrajectoryIWMDecision:
    if set(result) != {
        "status",
        "preferred",
        "trajectory_lifecycle",
        "rationale",
    } or _contains_number(result):
        raise ValueError("multi-trajectory IWM output schema is invalid")
    status = PoolPreferenceStatus(str(result.get("status") or ""))
    preferred_aliases = _string_list(result.get("preferred"), "preferred")
    if len(preferred_aliases) != len(set(preferred_aliases)) or not set(
        preferred_aliases
    ) <= set(aliases.values()):
        raise ValueError("multi-trajectory IWM returned unknown candidate aliases")
    if status is PoolPreferenceStatus.UNIQUE and len(preferred_aliases) != 1:
        raise ValueError("unique preference requires one candidate")
    if status is PoolPreferenceStatus.TIE and len(preferred_aliases) < 2:
        raise ValueError("tie preference requires multiple candidates")
    if status is PoolPreferenceStatus.INCOMPARABLE and preferred_aliases:
        raise ValueError("incomparable preference cannot select a candidate")
    by_alias = {alias: expansion_id for expansion_id, alias in aliases.items()}
    lifecycle_payload = result.get("trajectory_lifecycle")
    if not isinstance(lifecycle_payload, dict) or set(lifecycle_payload) != set(
        trajectory_aliases.values()
    ):
        raise ValueError("multi-trajectory lifecycle coverage mismatch")
    trajectory_by_alias = {
        alias: trajectory_id for trajectory_id, alias in trajectory_aliases.items()
    }
    lifecycle: list[PredictedLifecycle] = []
    for alias, row in lifecycle_payload.items():
        if not isinstance(row, dict) or set(row) != {"status", "rationale"}:
            raise ValueError("multi-trajectory lifecycle fields are invalid")
        lifecycle.append(
            PredictedLifecycle(
                trajectory_id=trajectory_by_alias[alias],
                status=TrajectoryStatus(str(row.get("status") or "")),
                rationale=str(row.get("rationale") or ""),
            )
        )
    return MultiTrajectoryIWMDecision(
        status=status,
        preferred_expansion_ids=tuple(by_alias[alias] for alias in preferred_aliases),
        lifecycle_predictions=tuple(lifecycle),
        rationale=str(result.get("rationale") or ""),
    )


class GPTOSSRealTrajectoryEvidenceAssessor:
    """Broadcast one real observation to all hypotheses in one categorical call."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))

    def assess_batch(
        self,
        trajectories: Sequence[ReasoningTrajectory],
        observation: MemoryNode,
        corrected_beliefs: Sequence[CursorBeliefState],
    ) -> Sequence[TrajectoryEvidenceAssessment]:
        if len(trajectories) != len(corrected_beliefs):
            raise ValueError("trajectory and corrected-belief coverage mismatch")
        aliases = {
            row.trajectory_id: _alphabetic_alias("trajectory", index)
            for index, row in enumerate(trajectories)
        }
        payload = {
            "question": corrected_beliefs[0].question,
            "executed_real_observation": {
                "text": observation.text,
                "predicate": observation.metadata.get("predicate"),
                "action_kind": observation.metadata.get("action_kind"),
                "participants": observation.metadata.get("participants") or [],
                "states": observation.metadata.get("states") or [],
                "state_change": observation.metadata.get("state_change"),
            },
            "competing_trajectories": {
                aliases[trajectory.trajectory_id]: {
                    "hypothesis": trajectory.hypothesis,
                    "missing_roles_after_belief_correction": list(belief.missing_roles),
                    "contradictions_after_belief_correction": list(
                        belief.contradictions
                    ),
                    "answerability_after_belief_correction": (
                        belief.answerability.value
                    ),
                }
                for trajectory, belief in zip(trajectories, corrected_beliefs)
            },
            "allowed_effects": [value.value for value in EvidenceEffect],
            "required_output": {
                "only_key": "assessments",
                "assessments": {
                    alias: {
                        "effect": "one allowed categorical effect",
                        "rationale": "short evidence-grounded rationale",
                    }
                    for alias in aliases.values()
                },
            },
            "required_contract": {
                "observation_is_executed_and_real": True,
                "assess_every_trajectory_independently": True,
                "no_hidden_clue_or_answer_label": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        for attempt in range(2):
            result = self.client.complete_json(
                task=(
                    (
                        "Using only the executed real observation, categorically assess "
                        "its effect on every competing reasoning hypothesis."
                    )
                    if attempt == 0
                    else (
                        "Repair the response to exactly cover every trajectory with one "
                        "categorical effect and rationale, using no numeric values."
                    )
                ),
                payload=payload,
            )
            try:
                if set(result) != {"assessments"} or _contains_number(result):
                    raise ValueError("trajectory assessment schema is invalid")
                rows = result.get("assessments")
                if not isinstance(rows, dict) or set(rows) != set(aliases.values()):
                    raise ValueError("trajectory assessment coverage mismatch")
                trajectory_by_alias = {
                    alias: trajectory_id for trajectory_id, alias in aliases.items()
                }
                parsed: list[TrajectoryEvidenceAssessment] = []
                for alias, row in rows.items():
                    if not isinstance(row, dict) or set(row) != {
                        "effect",
                        "rationale",
                    }:
                        raise ValueError("trajectory assessment fields are invalid")
                    parsed.append(
                        TrajectoryEvidenceAssessment(
                            trajectory_id=trajectory_by_alias[alias],
                            effect=EvidenceEffect(str(row.get("effect") or "")),
                            rationale=str(row.get("rationale") or ""),
                        )
                    )
                parsed_by_id = {row.trajectory_id: row for row in parsed}
                return tuple(parsed_by_id[row.trajectory_id] for row in trajectories)
            except ValueError:
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable real trajectory assessment repair state")


def initialize_trajectory_pool(
    belief: CursorBeliefState,
    hypotheses: Sequence[str],
    *,
    pool_id: str = "trajectory_pool:initial",
) -> TrajectoryPool:
    if not hypotheses or len(hypotheses) != len(set(hypotheses)):
        raise ValueError("initial hypotheses must be non-empty and unique")
    trajectories = tuple(
        ReasoningTrajectory(
            trajectory_id=_trajectory_id(pool_id, hypothesis),
            hypothesis=hypothesis,
            belief=replace(
                belief,
                belief_id=f"{belief.belief_id}:hypothesis:{index}",
            ),
        )
        for index, hypothesis in enumerate(hypotheses)
    )
    return TrajectoryPool(pool_id=pool_id, trajectories=trajectories)


def branch_trajectory_pool(
    pool: TrajectoryPool,
    parent_trajectory_id: str,
    hypotheses: Sequence[str],
) -> TrajectoryPool:
    """Add every supplied grounded interpretation; never select a fixed-K subset."""

    parent = next(
        (row for row in pool.trajectories if row.trajectory_id == parent_trajectory_id),
        None,
    )
    if parent is None:
        raise ValueError("trajectory branch parent is absent")
    if not hypotheses or len(hypotheses) != len(set(hypotheses)):
        raise ValueError("branch hypotheses must be non-empty and unique")
    existing_hypotheses = {row.hypothesis for row in pool.trajectories}
    if existing_hypotheses.intersection(hypotheses):
        raise ValueError("branch hypothesis already exists in the pool")
    children = tuple(
        ReasoningTrajectory(
            trajectory_id=_trajectory_id(
                f"{pool.pool_id}\x1f{parent_trajectory_id}", hypothesis
            ),
            parent_trajectory_id=parent_trajectory_id,
            hypothesis=hypothesis,
            belief=replace(
                parent.belief,
                belief_id=f"{parent.belief.belief_id}:branch:{index}",
            ),
            shared_observation_ids=parent.shared_observation_ids,
        )
        for index, hypothesis in enumerate(hypotheses)
    )
    return consolidate_trajectory_pool(
        TrajectoryPool(
            pool_id=f"{pool.pool_id}:branched",
            trajectories=(*pool.trajectories, *children),
            top_k_applied=False,
        )
    )


def execute_shared_trajectory_action(
    pool: TrajectoryPool,
    decision: MultiTrajectoryPlanDecision,
    graph: RetainedEvidenceGraph,
    *,
    belief_updater: RealBeliefUpdater | None = None,
    assessor: TrajectoryEvidenceAssessor | None = None,
) -> MultiTrajectoryExecution:
    action = decision.selected_action
    if action.kind in {ActionKind.STOP, ActionKind.ANSWER, ActionKind.ABSTAIN}:
        return MultiTrajectoryExecution(action, None, pool, ())
    selected_expansions = {
        row.expansion_id: row
        for row in decision.expansions
        if row.expansion_id in set(decision.preferred_expansion_ids)
    }
    sources = [
        row
        for row in selected_expansions.values()
        if row.action.action_id == action.action_id
    ]
    if not sources:
        raise ValueError("selected action has no preferred trajectory expansion")
    source_id = sources[0].trajectory_id
    trajectory_by_id = {row.trajectory_id: row for row in pool.trajectories}
    source = trajectory_by_id[source_id]
    execution = execute_graph_action(action, source.belief, graph)
    observation = execution.observation
    if observation is None:
        moved_rows = tuple(
            replace(
                trajectory,
                belief=execution.updated_belief,
                action_history=(*trajectory.action_history, action.action_id),
            )
            if trajectory.trajectory_id == source_id
            else trajectory
            for trajectory in pool.trajectories
        )
        return MultiTrajectoryExecution(
            action=action,
            observation=None,
            pool=consolidate_trajectory_pool(
                TrajectoryPool(
                    pool_id=f"{pool.pool_id}:cursor-step",
                    trajectories=moved_rows,
                )
            ),
            assessments=(),
        )
    pending: list[
        tuple[
            ReasoningTrajectory,
            CursorBeliefState,
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = []
    untouched: list[ReasoningTrajectory] = []
    for trajectory in pool.trajectories:
        if trajectory.status not in EXPANDABLE_STATUSES:
            untouched.append(trajectory)
            continue
        if trajectory.trajectory_id == source_id:
            belief = execution.updated_belief
            action_history = (*trajectory.action_history, action.action_id)
        else:
            belief = _share_execution_state(trajectory.belief, action)
            action_history = trajectory.action_history
        shared_observations = trajectory.shared_observation_ids
        shared_observations = tuple(
            dict.fromkeys((*shared_observations, observation.node_id))
        )
        if belief_updater is not None:
            hypothesis_aware = getattr(
                belief_updater, "update_for_trajectory", None
            )
            action_aware = getattr(belief_updater, "update_after_action", None)
            if callable(hypothesis_aware):
                belief = hypothesis_aware(
                    trajectory.trajectory_id,
                    trajectory.hypothesis,
                    trajectory.belief,
                    belief,
                    action,
                    observation,
                    graph,
                )
            elif callable(action_aware):
                belief = action_aware(
                    trajectory.trajectory_id,
                    trajectory.belief,
                    belief,
                    action,
                    observation,
                    graph,
                )
            else:
                belief = belief_updater.update(belief, observation)
        pending.append((trajectory, belief, action_history, shared_observations))

    if assessor is not None:
        assessments = tuple(
            assessor.assess_batch(
                tuple(row[0] for row in pending),
                observation,
                tuple(row[1] for row in pending),
            )
        )
    else:
        assessments = tuple(
            TrajectoryEvidenceAssessment(
                row[0].trajectory_id,
                EvidenceEffect.INCONCLUSIVE,
                "no categorical trajectory assessor supplied",
            )
            for row in pending
        )
    expected_ids = {row[0].trajectory_id for row in pending}
    if (
        len(assessments) != len(pending)
        or {row.trajectory_id for row in assessments} != expected_ids
    ):
        raise ValueError("trajectory assessment coverage mismatch")
    assessment_by_id = {row.trajectory_id: row for row in assessments}
    updated_by_id = {row.trajectory_id: row for row in untouched}
    for trajectory, belief, action_history, shared_observations in pending:
        assessment = assessment_by_id[trajectory.trajectory_id]
        updated_by_id[trajectory.trajectory_id] = replace(
            trajectory,
            belief=belief,
            action_history=action_history,
            shared_observation_ids=shared_observations,
            status=_status_after_evidence(assessment.effect),
            lifecycle_rationale=assessment.rationale,
        )
    updated_rows = [updated_by_id[row.trajectory_id] for row in pool.trajectories]
    updated_pool = consolidate_trajectory_pool(
        TrajectoryPool(
            pool_id=f"{pool.pool_id}:step:{_shared_step(updated_rows)}",
            trajectories=tuple(updated_rows),
        )
    )
    return MultiTrajectoryExecution(
        action=action,
        observation=observation,
        pool=updated_pool,
        assessments=assessments,
    )


def run_multi_trajectory_closed_loop(
    pool: TrajectoryPool,
    graph: RetainedEvidenceGraph,
    planner: MultiTrajectoryIWMPlanner,
    *,
    max_decisions: int = 16,
    belief_updater: RealBeliefUpdater | None = None,
    assessor: TrajectoryEvidenceAssessor | None = None,
) -> MultiTrajectoryRunTrace:
    """Execute one shared real action per decision and preserve the full pool."""

    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    initial = pool
    steps: list[MultiTrajectoryTraceStep] = []
    seen: set[tuple[Any, ...]] = set()
    termination = "decision_limit"
    for _ in range(max_decisions):
        if not pool.expandable:
            termination = "no_expandable_trajectories"
            break
        decision = planner.plan(pool, graph)
        state = (
            tuple(
                (
                    row.trajectory_id,
                    row.status.value,
                    row.belief.current_node_id,
                    row.belief.acquired_evidence,
                )
                for row in pool.trajectories
            ),
            decision.selected_action.action_id,
        )
        if state in seen:
            termination = "repeated_nonprogress_state"
            break
        seen.add(state)
        if decision.selected_action.kind in {
            ActionKind.STOP,
            ActionKind.ANSWER,
            ActionKind.ABSTAIN,
        }:
            termination = decision.selected_action.kind.value
            steps.append(
                MultiTrajectoryTraceStep(
                    pool_before=pool,
                    decision=decision,
                    observation_id=None,
                    assessments=(),
                    pool_after=pool,
                )
            )
            break
        before = pool
        execution = execute_shared_trajectory_action(
            pool,
            decision,
            graph,
            belief_updater=belief_updater,
            assessor=assessor,
        )
        pool = execution.pool
        steps.append(
            MultiTrajectoryTraceStep(
                pool_before=before,
                decision=decision,
                observation_id=(
                    execution.observation.node_id
                    if execution.observation is not None
                    else None
                ),
                assessments=execution.assessments,
                pool_after=pool,
            )
        )
        if pool.expandable and pool.expandable[0].belief.remaining_reads == 0:
            termination = "read_budget_exhausted"
            break
    return MultiTrajectoryRunTrace(
        initial_pool=initial,
        final_pool=pool,
        steps=tuple(steps),
        termination=termination,
    )


def consolidate_trajectory_pool(pool: TrajectoryPool) -> TrajectoryPool:
    """Merge exact equivalents only; never truncate the pool by a score or K."""

    representatives: dict[tuple[Any, ...], str] = {}
    rows: list[ReasoningTrajectory] = []
    for trajectory in pool.trajectories:
        if trajectory.status not in EXPANDABLE_STATUSES:
            rows.append(trajectory)
            continue
        signature = _trajectory_signature(trajectory)
        representative = representatives.get(signature)
        if representative is None:
            representatives[signature] = trajectory.trajectory_id
            rows.append(trajectory)
        else:
            rows.append(
                replace(
                    trajectory,
                    status=TrajectoryStatus.MERGED,
                    merged_into=representative,
                    lifecycle_rationale="exact_structural_equivalence",
                )
            )
    return TrajectoryPool(pool.pool_id, tuple(rows), top_k_applied=False)


def shared_action_key(action: LegalGraphAction) -> tuple[str, ...]:
    """Identify one real evidence read across trajectory-specific graph routes."""

    if action.reads_evidence and action.target_id is not None:
        return ("read_evidence", action.target_id)
    return (
        action.kind.value,
        action.source_id or "",
        action.target_id or "",
        action.edge_id or "",
    )


def _share_execution_state(
    belief: CursorBeliefState,
    action: LegalGraphAction,
) -> CursorBeliefState:
    if not action.reads_evidence or action.target_id is None:
        return belief
    acquired = tuple(dict.fromkeys((*belief.acquired_evidence, action.target_id)))
    imagined = tuple(
        node_id for node_id in belief.imagined_evidence if node_id != action.target_id
    )
    return replace(
        belief,
        belief_id=f"{belief.belief_id}:shared-read:{action.action_id}",
        acquired_evidence=acquired,
        imagined_evidence=imagined,
        remaining_reads=max(0, belief.remaining_reads - 1),
        step=belief.step + 1,
    )


def _status_after_evidence(effect: EvidenceEffect) -> TrajectoryStatus:
    return {
        EvidenceEffect.SUPPORT: TrajectoryStatus.SUPPORTED,
        EvidenceEffect.COUNTEREVIDENCE: TrajectoryStatus.CONTRADICTED,
        EvidenceEffect.INCONCLUSIVE: TrajectoryStatus.INCONCLUSIVE,
        EvidenceEffect.COMPLETE: TrajectoryStatus.COMPLETED,
    }[effect]


def _trajectory_signature(trajectory: ReasoningTrajectory) -> tuple[Any, ...]:
    belief = trajectory.belief
    return (
        trajectory.hypothesis,
        belief.current_node_id,
        belief.missing_roles,
        belief.contradictions,
        belief.accepted_relations,
        belief.rejected_relations,
        trajectory.action_history,
    )


def _trajectory_payload(
    trajectory: ReasoningTrajectory,
    graph_input: Any,
) -> dict[str, Any]:
    acquired_values = [
        {
            "semantic_key": view.key.semantic_key,
            "evidence_value": view.evidence_value,
        }
        for view in graph_input.nodes
        if view.acquired and view.evidence_value is not None
    ]
    return {
        "hypothesis": trajectory.hypothesis,
        "status": trajectory.status.value,
        "current_semantic_key": _semantic_key(
            graph_input, trajectory.belief.current_node_id
        ),
        "required_roles": list(trajectory.belief.required_roles),
        "missing_roles": list(trajectory.belief.missing_roles),
        "grounded_role_evidence": [
            {
                "role": role,
                "evidence_semantic_key": _semantic_key(graph_input, node_id),
            }
            for role, node_id in trajectory.belief.grounded_role_evidence
        ],
        "contradictions": list(trajectory.belief.contradictions),
        "acquired_real_evidence": acquired_values,
        "has_prior_reasoning_steps": bool(trajectory.action_history),
    }


def _expansion_payload(
    expansion: TrajectoryExpansion,
    trajectory_alias: str,
    graph_input: Any,
) -> dict[str, Any]:
    action = expansion.action
    target_view = next(
        (view for view in graph_input.nodes if view.key.node_id == action.target_id),
        None,
    )
    return {
        "trajectory": trajectory_alias,
        "action_kind": action.kind.value,
        "source_semantic_key": _semantic_key(graph_input, action.source_id),
        "target_semantic_key": (
            target_view.key.semantic_key if target_view is not None else None
        ),
        "target_node_type": (
            target_view.key.node_type if target_view is not None else None
        ),
        "target_time_span": (
            {
                "start_s": target_view.key.start_s,
                "end_s": target_view.key.end_s,
            }
            if target_view is not None
            else None
        ),
        "target_structural_tags": (
            list(target_view.key.structural_tags) if target_view is not None else []
        ),
        "embedding_available": (
            target_view.key.embedding_ref is not None
            if target_view is not None
            else False
        ),
        "temporal_or_correlation_relation": action.relation,
        "reads_real_evidence": action.reads_evidence,
    }


def _semantic_key(graph_input: Any, node_id: str | None) -> str | None:
    if node_id is None:
        return None
    return next(
        (
            view.key.semantic_key
            for view in graph_input.nodes
            if view.key.node_id == node_id
        ),
        None,
    )


def _shared_abstain(
    expansions: Sequence[TrajectoryExpansion],
) -> LegalGraphAction:
    return next(
        row.action for row in expansions if row.action.kind is ActionKind.ABSTAIN
    )


def _trajectory_id(pool_id: str, hypothesis: str) -> str:
    digest = hashlib.sha256(f"{pool_id}\x1f{hypothesis}".encode("utf-8")).hexdigest()[
        :20
    ]
    return f"reasoning_trajectory:{digest}"


def _alphabetic_alias(prefix: str, index: int) -> str:
    value = index
    suffix = ""
    while True:
        value, remainder = divmod(value, 26)
        suffix = chr(ord("a") + remainder) + suffix
        if value == 0:
            return f"{prefix}_{suffix}"
        value -= 1


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise ValueError(f"{field} must be a string list")
    return tuple(value)


def _contains_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_contains_number(row) for row in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_number(row) for row in value)
    return False


def _shared_step(rows: Sequence[ReasoningTrajectory]) -> int:
    expandable = [row.belief.step for row in rows if row.status in EXPANDABLE_STATUSES]
    return max(expandable, default=0)
