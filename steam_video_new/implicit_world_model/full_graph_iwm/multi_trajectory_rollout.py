"""Complete-coverage categorical rollout planning for competing hypotheses.

The module deliberately separates imagined transitions from the persistent
``TrajectoryPool``. Every legal hypothesis-conditioned chain is predicted,
every horizon-one/two chain participates in the categorical partial order, and
only the first action of a uniquely preferred first-hop set may be executed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from itertools import combinations
import json
from typing import Any, Protocol, Sequence

from .action_compiler import GraphActionCompiler
from .contracts import (
    ActionKind,
    AnswerabilityState,
    CategoricalBeliefDelta,
    ContradictionChange,
    CursorBeliefState,
    EvidenceOutcome,
    FrontierChange,
    ImaginedTransition,
    LegalGraphAction,
    PredictedObservation,
    PreferenceLabel,
    ProgressChange,
    RetainedEvidenceGraph,
)
from .model_input import build_iwm_graph_input
from .multi_trajectory import (
    MultiTrajectoryPlanDecision,
    TrajectoryExpansion,
    TrajectoryPool,
    shared_action_key,
)
from .planner import project_imagined_belief


TERMINAL_ACTIONS = {ActionKind.STOP, ActionKind.ANSWER, ActionKind.ABSTAIN}


@dataclass(frozen=True)
class HypothesisExpansionRequest:
    """One legal action under one real or imagined hypothesis state."""

    request_id: str
    trajectory_id: str
    hypothesis: str
    belief: CursorBeliefState
    action: LegalGraphAction
    imagined_history: tuple[ImaginedTransition, ...] = ()
    trajectory_contexts: tuple[tuple[str, str, CursorBeliefState], ...] = ()


@dataclass(frozen=True)
class ImaginedHypothesisPath:
    """A raw short chain or a joint first-hop tree across competing paths."""

    path_id: str
    trajectory_id: str
    hypothesis: str
    transitions: tuple[ImaginedTransition, ...]
    continuations: tuple[ImaginedTransition, ...] = ()
    conditioned_outcomes: tuple["HypothesisConditionedOutcome", ...] = ()

    def __post_init__(self) -> None:
        if not self.path_id or not self.trajectory_id or not self.hypothesis.strip():
            raise ValueError("imagined path identity must be non-empty")
        if not self.transitions or len(self.transitions) > 2:
            raise ValueError("imagined path must contain one or two root transitions")
        if self.continuations:
            raise ValueError(
                "canonical per-chain rollout stores continuations as distinct paths"
            )
        if self.conditioned_outcomes:
            root_key = shared_action_key(self.transitions[0].action)
            if any(
                not row.transitions
                or len(row.transitions) > 2
                or shared_action_key(row.transitions[0].action) != root_key
                for row in self.conditioned_outcomes
            ):
                raise ValueError(
                    "joint-tree outcomes must share one legal first-hop action"
                )

    @property
    def first_action(self) -> LegalGraphAction:
        return self.transitions[0].action


@dataclass(frozen=True)
class HypothesisConditionedOutcome:
    trajectory_id: str
    hypothesis: str
    transitions: tuple[ImaginedTransition, ...]


@dataclass(frozen=True)
class HypothesisPathPair:
    comparison_id: str
    left: ImaginedHypothesisPath
    right: ImaginedHypothesisPath


@dataclass(frozen=True)
class HypothesisPathPreference:
    comparison_id: str
    left_id: str
    right_id: str
    label: PreferenceLabel
    rationale: str = ""


class CategoricalHypothesisWorldModel(Protocol):
    model_name: str

    def predict_batch(
        self,
        requests: Sequence[HypothesisExpansionRequest],
        graph: RetainedEvidenceGraph,
    ) -> Sequence[ImaginedTransition]: ...


class CategoricalHypothesisPathPreference(Protocol):
    model_name: str

    def compare_batch(
        self,
        pool: TrajectoryPool,
        pairs: Sequence[HypothesisPathPair],
        graph: RetainedEvidenceGraph,
        *,
        include_imagined_transitions: bool,
    ) -> Sequence[HypothesisPathPreference]: ...


class CategoricalHypothesisSetwisePreference(Protocol):
    model_name: str

    def select_frontier(
        self,
        pool: TrajectoryPool,
        paths: Sequence[ImaginedHypothesisPath],
        graph: RetainedEvidenceGraph,
        *,
        include_imagined_transitions: bool,
    ) -> tuple[str, ...]: ...


class CategoricalEvidenceActionScheduler(Protocol):
    """Schedule one real read while preserving every surviving trajectory."""

    model_name: str

    def select_evidence_action(
        self,
        pool: TrajectoryPool,
        paths: Sequence[ImaginedHypothesisPath],
        graph: RetainedEvidenceGraph,
        *,
        include_imagined_transitions: bool,
    ) -> str | None: ...


class MultiTrajectoryRolloutPlanner:
    """Enumerate all short paths and select from their categorical partial order."""

    def __init__(
        self,
        world_model: CategoricalHypothesisWorldModel,
        preference_model: CategoricalHypothesisPathPreference,
        *,
        horizon: int = 2,
        action_compiler: GraphActionCompiler | None = None,
        max_complete_pairs: int | None = None,
        setwise_preference_model: (
            CategoricalHypothesisSetwisePreference | None
        ) = None,
        evidence_scheduler: CategoricalEvidenceActionScheduler | None = None,
    ) -> None:
        if horizon not in {1, 2}:
            raise ValueError("multi-trajectory rollout horizon must be one or two")
        self.world_model = world_model
        self.preference_model = preference_model
        self.horizon = horizon
        self.action_compiler = action_compiler or GraphActionCompiler()
        if max_complete_pairs is not None and max_complete_pairs < 1:
            raise ValueError("max_complete_pairs must be positive when supplied")
        self.max_complete_pairs = max_complete_pairs
        self.setwise_preference_model = setwise_preference_model
        self.evidence_scheduler = evidence_scheduler
        self.last_complete_coverage_audit: dict[str, Any] = {}

    def plan(
        self,
        pool: TrajectoryPool,
        graph: RetainedEvidenceGraph,
    ) -> MultiTrajectoryPlanDecision:
        all_requests = compile_hypothesis_requests(pool, graph, self.action_compiler)
        requests = _active_planning_requests(all_requests)
        if not all_requests or not requests:
            raise RuntimeError("trajectory pool has no legal expansions")
        first_predictions = self._predict_checked(requests, graph)
        paths: list[ImaginedHypothesisPath] = []
        second_requests: list[HypothesisExpansionRequest] = []
        parent_by_request: dict[
            str, tuple[HypothesisExpansionRequest, ImaginedTransition]
        ] = {}
        for request, prediction in zip(requests, first_predictions):
            if self.horizon == 1 or prediction.action.kind in TERMINAL_ACTIONS:
                paths.append(_path(request, (prediction,)))
                continue
            imagined_belief = project_imagined_belief(
                request.belief,
                prediction,
            )
            children = _active_planning_requests(
                _compile_trajectory_requests(
                    request.trajectory_id,
                    request.hypothesis,
                    imagined_belief,
                    graph,
                    self.action_compiler,
                    imagined_history=(prediction,),
                )
            )
            for child in children:
                second_requests.append(child)
                parent_by_request[child.request_id] = (request, prediction)
        if second_requests:
            second_predictions = self._predict_checked(tuple(second_requests), graph)
            for request, prediction in zip(second_requests, second_predictions):
                root, first = parent_by_request[request.request_id]
                paths.append(_path(root, (first, prediction)))
        if not paths:
            raise RuntimeError("world model produced no imagined paths")

        ordered_paths = _joint_reasoning_chains(paths)
        expansions = tuple(
            TrajectoryExpansion(
                expansion_id=request.request_id,
                trajectory_id=request.trajectory_id,
                action=request.action,
            )
            for request in requests
        )
        pair_count = len(ordered_paths) * (len(ordered_paths) - 1) // 2
        if (
            self.setwise_preference_model is None
            and self.max_complete_pairs is not None
            and pair_count > self.max_complete_pairs
        ):
            self.last_complete_coverage_audit = {
                "horizon": self.horizon,
                "first_expansion_count": len(requests),
                "second_expansion_count": len(second_requests),
                "imagined_path_count": len(ordered_paths),
                "exhaustive_pair_count": pair_count,
                "comparison_count": 0,
                "complete_coverage": False,
                "failure": "complete_comparison_budget_exceeded",
                "top_k_applied": False,
                "canonical_path_order": True,
            }
            return MultiTrajectoryPlanDecision(
                selected_action=_abstain_action(all_requests),
                planning_status="rollout_abstain_complete_comparison_budget_exceeded",
                expansions=expansions,
                preferred_expansion_ids=(),
                lifecycle_predictions=(),
                legal_expansion_count=len(expansions),
                top_k_applied=False,
                imagined_paths=ordered_paths,
                preferred_path_ids=(),
                planning_horizon=self.horizon,
                preference_audit=self.last_complete_coverage_audit,
            )
        if self.setwise_preference_model is not None:
            pairs: tuple[HypothesisPathPair, ...] = ()
            preferences: tuple[HypothesisPathPreference, ...] = ()
            preferred = self.setwise_preference_model.select_frontier(
                pool,
                ordered_paths,
                graph,
                include_imagined_transitions=True,
            )
        else:
            pairs = tuple(
                _pair(left, right) for left, right in combinations(ordered_paths, 2)
            )
            preferences = tuple(
                self.preference_model.compare_batch(
                    pool,
                    pairs,
                    graph,
                    include_imagined_transitions=True,
                )
            )
            preferred = _undominated_paths(ordered_paths, pairs, preferences)
        known_path_ids = {path.path_id for path in ordered_paths}
        if (
            len(preferred) != len(set(preferred))
            or not set(preferred) <= known_path_ids
        ):
            raise ValueError("preference frontier contains invalid chain IDs")
        first_hops = {
            shared_action_key(path.first_action): path.first_action
            for path in ordered_paths
            if path.path_id in set(preferred)
        }
        scheduler_used = False
        scheduler_status = "not_needed"
        if len(first_hops) == 1:
            selected = next(iter(first_hops.values()))
            status = "rollout_selected_unique_preferred_first_hop"
        elif first_hops:
            scheduler = getattr(self.evidence_scheduler, "select_evidence_action", None)
            scheduler_used = callable(scheduler)
            if not scheduler_used:
                selected = _abstain_action(all_requests)
                status = "rollout_abstain_non_unique_first_hops"
                scheduler_status = "not_configured"
                scheduled_path_id = None
            else:
                scheduled_path_id = scheduler(
                    pool,
                    tuple(
                        path for path in ordered_paths if path.path_id in set(preferred)
                    ),
                    graph,
                    include_imagined_transitions=True,
                )
            scheduled_path = next(
                (
                    path
                    for path in ordered_paths
                    if path.path_id == scheduled_path_id
                    and path.path_id in set(preferred)
                    and path.first_action.kind not in TERMINAL_ACTIONS
                ),
                None,
            )
            if not scheduler_used:
                pass
            elif scheduled_path is not None:
                selected = scheduled_path.first_action
                status = "rollout_scheduled_discriminating_first_hop"
                scheduler_status = "selected"
            else:
                selected = _abstain_action(all_requests)
                status = "rollout_abstain_scheduler_inconclusive"
                scheduler_status = "inconclusive"
        else:
            selected = _abstain_action(all_requests)
            status = "rollout_abstain_preference_cycle"
        preferred_first_request_ids = tuple(
            request.request_id
            for request in requests
            if shared_action_key(request.action) in first_hops
        )
        self.last_complete_coverage_audit = {
            "horizon": self.horizon,
            "first_expansion_count": len(requests),
            "second_expansion_count": len(second_requests),
            "imagined_path_count": len(ordered_paths),
            "hypothesis_conditioned_outcome_count": len(paths),
            "path_representation": (
                "one_joint_first_hop_tree_with_all_hypothesis_conditioned_chains"
            ),
            "exhaustive_pair_count": len(pairs),
            "comparison_count": (
                1 if self.setwise_preference_model is not None else len(preferences)
            ),
            "complete_coverage": (
                True
                if self.setwise_preference_model is not None
                else len(preferences) == len(pairs)
            ),
            "preference_mode": (
                "setwise_full_frontier"
                if self.setwise_preference_model is not None
                else "exhaustive_pairwise"
            ),
            "top_k_applied": False,
            "canonical_path_order": True,
            "evidence_scheduler_used": scheduler_used,
            "evidence_scheduler_status": scheduler_status,
            "surviving_trajectory_paths_preserved": True,
        }
        return MultiTrajectoryPlanDecision(
            selected_action=selected,
            planning_status=status,
            expansions=expansions,
            preferred_expansion_ids=preferred_first_request_ids,
            lifecycle_predictions=(),
            legal_expansion_count=len(expansions),
            top_k_applied=False,
            imagined_paths=ordered_paths,
            preferred_path_ids=preferred,
            planning_horizon=self.horizon,
            preference_audit=self.last_complete_coverage_audit,
        )

    def _predict_checked(
        self,
        requests: Sequence[HypothesisExpansionRequest],
        graph: RetainedEvidenceGraph,
    ) -> tuple[ImaginedTransition, ...]:
        predictions = tuple(self.world_model.predict_batch(requests, graph))
        if len(predictions) != len(requests):
            raise ValueError("world model prediction coverage mismatch")
        for request, prediction in zip(requests, predictions):
            if prediction.action != request.action:
                raise ValueError("world model prediction changed its legal action")
            if not prediction.observation.predicted_only:
                raise ValueError("imagined observation was marked real")
            if not prediction.belief_delta.predicted_only:
                raise ValueError("imagined belief delta was marked real")
            if not request.action.reads_evidence and prediction != _neutral_transition(
                request
            ):
                raise ValueError(
                    "no-read action must have an invariant neutral transition"
                )
        return predictions


class ReactiveMultiTrajectoryPlanner:
    """Matched no-WM arm: compare current legal actions without transitions."""

    def __init__(
        self,
        preference_model: CategoricalHypothesisPathPreference,
        *,
        action_compiler: GraphActionCompiler | None = None,
        setwise_preference_model: (
            CategoricalHypothesisSetwisePreference | None
        ) = None,
        evidence_scheduler: CategoricalEvidenceActionScheduler | None = None,
    ) -> None:
        self.preference_model = preference_model
        self.action_compiler = action_compiler or GraphActionCompiler()
        self.setwise_preference_model = setwise_preference_model
        self.evidence_scheduler = evidence_scheduler
        self.last_complete_coverage_audit: dict[str, Any] = {}

    def plan(
        self, pool: TrajectoryPool, graph: RetainedEvidenceGraph
    ) -> MultiTrajectoryPlanDecision:
        all_requests = compile_hypothesis_requests(pool, graph, self.action_compiler)
        requests = _active_planning_requests(all_requests)
        raw_paths = tuple(
            _path(request, (_neutral_transition(request),)) for request in requests
        )
        paths = _joint_reasoning_chains(raw_paths)
        if self.setwise_preference_model is not None:
            pairs: tuple[HypothesisPathPair, ...] = ()
            preferences: tuple[HypothesisPathPreference, ...] = ()
            preferred = self.setwise_preference_model.select_frontier(
                pool,
                paths,
                graph,
                include_imagined_transitions=False,
            )
        else:
            pairs = tuple(_pair(left, right) for left, right in combinations(paths, 2))
            preferences = tuple(
                self.preference_model.compare_batch(
                    pool,
                    pairs,
                    graph,
                    include_imagined_transitions=False,
                )
            )
            preferred = _undominated_paths(paths, pairs, preferences)
        first_hops = {
            shared_action_key(path.first_action): path.first_action
            for path in paths
            if path.path_id in set(preferred)
        }
        scheduler_used = False
        scheduler_status = "not_needed"
        if len(first_hops) == 1:
            selected = next(iter(first_hops.values()))
            status = "reactive_selected_unique_preferred_first_hop"
        elif first_hops:
            scheduler = getattr(self.evidence_scheduler, "select_evidence_action", None)
            scheduler_used = callable(scheduler)
            if not scheduler_used:
                selected = _abstain_action(all_requests)
                status = "reactive_abstain_non_unique_first_hops"
                scheduler_status = "not_configured"
                scheduled_path_id = None
            else:
                scheduled_path_id = scheduler(
                    pool,
                    tuple(path for path in paths if path.path_id in set(preferred)),
                    graph,
                    include_imagined_transitions=False,
                )
            scheduled_path = next(
                (
                    path
                    for path in paths
                    if path.path_id == scheduled_path_id
                    and path.path_id in set(preferred)
                    and path.first_action.kind not in TERMINAL_ACTIONS
                ),
                None,
            )
            if not scheduler_used:
                pass
            elif scheduled_path is not None:
                selected = scheduled_path.first_action
                status = "reactive_scheduled_discriminating_first_hop"
                scheduler_status = "selected"
            else:
                selected = _abstain_action(all_requests)
                status = "reactive_abstain_scheduler_inconclusive"
                scheduler_status = "inconclusive"
        else:
            selected = _abstain_action(all_requests)
            status = "reactive_abstain_empty_frontier"
        expansions = tuple(
            TrajectoryExpansion(row.request_id, row.trajectory_id, row.action)
            for row in requests
        )
        self.last_complete_coverage_audit = {
            "horizon": 0,
            "first_expansion_count": len(requests),
            "imagined_path_count": len(paths),
            "hypothesis_conditioned_outcome_count": len(raw_paths),
            "exhaustive_pair_count": len(pairs),
            "comparison_count": (
                1 if self.setwise_preference_model is not None else len(preferences)
            ),
            "complete_coverage": (
                True
                if self.setwise_preference_model is not None
                else len(preferences) == len(pairs)
            ),
            "preference_mode": (
                "setwise_full_frontier"
                if self.setwise_preference_model is not None
                else "exhaustive_pairwise"
            ),
            "top_k_applied": False,
            "world_model_predictions_visible": False,
            "evidence_scheduler_used": scheduler_used,
            "evidence_scheduler_status": scheduler_status,
            "surviving_trajectory_paths_preserved": True,
        }
        return MultiTrajectoryPlanDecision(
            selected_action=selected,
            planning_status=status,
            expansions=expansions,
            preferred_expansion_ids=tuple(
                row.request_id
                for row in requests
                if shared_action_key(row.action) in first_hops
            ),
            lifecycle_predictions=(),
            legal_expansion_count=len(expansions),
            top_k_applied=False,
            imagined_paths=(),
            preferred_path_ids=preferred,
            planning_horizon=0,
            preference_audit=self.last_complete_coverage_audit,
        )


class GPTOSSCategoricalMultiTrajectoryModel:
    """Categorical transition and exhaustive pairwise preference adapter."""

    def __init__(
        self,
        client: Any,
        *,
        transition_batch_size: int = 1,
        comparison_batch_size: int = 24,
    ) -> None:
        if transition_batch_size < 1 or comparison_batch_size < 1:
            raise ValueError("transport batch sizes must be positive")
        self.client = client
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))
        self.transition_batch_size = transition_batch_size
        self.comparison_batch_size = comparison_batch_size
        self.transport_audits: list[dict[str, Any]] = []
        self._transition_sequence_normalizations = 0
        self._comparison_sequence_normalizations = 0
        self._transition_adaptive_splits = 0

    def predict_batch(
        self,
        requests: Sequence[HypothesisExpansionRequest],
        graph: RetainedEvidenceGraph,
    ) -> Sequence[ImaginedTransition]:
        normalization_start = self._transition_sequence_normalizations
        split_start = self._transition_adaptive_splits
        predictions: list[ImaginedTransition] = []
        model_requests = tuple(row for row in requests if row.action.reads_evidence)
        groups = _group_transition_requests(model_requests)
        for start in range(0, len(groups), self.transition_batch_size):
            batch = tuple(groups[start : start + self.transition_batch_size])
            predictions.extend(self._predict_with_adaptive_split(batch, graph))
        prediction_by_request = {
            request.request_id: prediction
            for request, prediction in zip(
                (request for group in groups for request in group),
                predictions,
            )
        }
        ordered_predictions = tuple(
            prediction_by_request[request.request_id]
            if request.action.reads_evidence
            else _neutral_transition(request)
            for request in requests
        )
        self.transport_audits.append(
            {
                "operation": "categorical_transition",
                "item_count": len(requests),
                "model_predicted_item_count": len(model_requests),
                "neutral_no_read_item_count": len(requests) - len(model_requests),
                "shared_action_group_count": len(groups),
                "batch_count": _batch_count(len(groups), self.transition_batch_size),
                "transport_batch_unit": "shared_action_group",
                "complete_coverage": len(ordered_predictions) == len(requests),
                "ordered_sequence_normalization_count": (
                    self._transition_sequence_normalizations - normalization_start
                ),
                "adaptive_transport_split_count": (
                    self._transition_adaptive_splits - split_start
                ),
                "top_k_applied": False,
            }
        )
        return ordered_predictions

    def _predict_with_adaptive_split(
        self,
        groups: tuple[tuple[HypothesisExpansionRequest, ...], ...],
        graph: RetainedEvidenceGraph,
    ) -> tuple[ImaginedTransition, ...]:
        """Split only failed transport batches; preserve every request and group."""

        try:
            return self._predict_one_batch(groups, graph)
        except (RuntimeError, ValueError):
            if len(groups) <= 1:
                raise
            midpoint = len(groups) // 2
            self._transition_adaptive_splits += 1
            return (
                *self._predict_with_adaptive_split(groups[:midpoint], graph),
                *self._predict_with_adaptive_split(groups[midpoint:], graph),
            )

    def select_evidence_action(
        self,
        pool: TrajectoryPool,
        paths: Sequence[ImaginedHypothesisPath],
        graph: RetainedEvidenceGraph,
        *,
        include_imagined_transitions: bool,
    ) -> str | None:
        """Categorically schedule one observation without deleting any path."""

        ordered = tuple(
            path
            for path in sorted(paths, key=lambda row: row.path_id)
            if path.first_action.kind not in TERMINAL_ACTIONS
        )
        if not ordered:
            return None
        aliases = {
            _alias("evidence_action", index): path for index, path in enumerate(ordered)
        }
        payload = {
            "question": pool.trajectories[0].belief.question,
            "acquired_real_evidence": _acquired_evidence_payload(pool, graph),
            "surviving_reasoning_paths": {
                alias: _path_payload(
                    path,
                    graph,
                    include_imagined_transitions=include_imagined_transitions,
                )
                for alias, path in aliases.items()
            },
            "allowed_status": ["select", "incomparable"],
            "required_output": {
                "only_keys": ["status", "selected", "rationale"],
                "selected": (
                    "one exact evidence_action alias when select; "
                    "none when genuinely incomparable"
                ),
            },
            "contract": {
                "schedule_one_real_observation_only": True,
                "preserve_every_surviving_reasoning_path": True,
                "prefer_hypothesis_discrimination_and_future_recoverability": True,
                "do_not_treat_frontier_size_as_progress": True,
                "no_candidate_pruning_or_top_k": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        task = (
            "The reasoning-path frontier remains alive. Select the single real "
            "evidence action whose predicted observation best distinguishes the "
            "surviving hypotheses or unlocks their next grounded reasoning step. "
            "This schedules one read only and must not delete any path. Return "
            "incomparable only when the supplied outcomes provide no categorical "
            "basis for choosing. Emit no numbers."
        )
        for attempt in range(2):
            result = self.client.complete_json(
                task=(task if attempt == 0 else _SCHEDULER_REPAIR_TASK),
                payload=payload,
            )
            try:
                if (
                    not isinstance(result, dict)
                    or set(result) != {"status", "selected", "rationale"}
                    or _contains_number(result)
                ):
                    raise ValueError("evidence scheduler response schema is invalid")
                status = str(result.get("status") or "")
                selected = result.get("selected")
                if status == "select":
                    if not isinstance(selected, str) or selected not in aliases:
                        raise ValueError(
                            "evidence scheduler selected an unknown action"
                        )
                    selected_path_id: str | None = aliases[selected].path_id
                elif status == "incomparable":
                    if selected not in {None, "none"}:
                        raise ValueError(
                            "incomparable scheduler result must not select"
                        )
                    selected_path_id = None
                else:
                    raise ValueError("evidence scheduler status is invalid")
                self.transport_audits.append(
                    {
                        "operation": "categorical_evidence_scheduler",
                        "item_count": len(ordered),
                        "batch_count": 1,
                        "complete_coverage": True,
                        "top_k_applied": False,
                        "surviving_paths_preserved": True,
                        "imagined_transitions_visible": include_imagined_transitions,
                    }
                )
                return selected_path_id
            except (TypeError, ValueError) as exc:
                if "finish_reason=length" in str(exc):
                    raise
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable evidence scheduler repair state")

    def _predict_one_batch(
        self,
        groups: tuple[tuple[HypothesisExpansionRequest, ...], ...],
        graph: RetainedEvidenceGraph,
    ) -> tuple[ImaginedTransition, ...]:
        requests = tuple(request for group in groups for request in group)
        aliases = {
            _alias("prediction", index): row for index, row in enumerate(requests)
        }
        alias_by_request = {
            request.request_id: alias for alias, request in aliases.items()
        }
        grouped_payload: dict[str, Any] = {}
        for group_index, group in enumerate(groups):
            representative = group[0]
            grouped_payload[_alias("action_group", group_index)] = {
                "shared_action": _transition_action_payload(
                    representative.action,
                    representative.belief,
                    graph,
                ),
                "conditioned_requests": {
                    alias_by_request[request.request_id]: (
                        _transition_condition_payload(request, graph)
                    )
                    for request in group
                },
            }
        payload = {
            "question": requests[0].belief.question,
            "acquired_real_evidence": _transition_acquired_evidence_payload(
                requests[0].belief,
                graph,
            ),
            "shared_action_groups": grouped_payload,
            "allowed": {
                "outcome": [row.value for row in EvidenceOutcome],
                "progress": [row.value for row in ProgressChange],
                "answerability_after": [row.value for row in AnswerabilityState],
                "frontier_change": [row.value for row in FrontierChange],
                "contradiction_change": [row.value for row in ContradictionChange],
            },
            "required_output": {
                "only_key": "predictions",
                "one_row_per_conditioned_request": list(aliases),
                "fields": [
                    "outcome",
                    "progress",
                    "answerability_after",
                    "frontier_change",
                    "contradiction_change",
                    "resolved_roles",
                    "opened_roles",
                    "relation_updates",
                    "rationale",
                ],
                "resolved_roles": (
                    "zero or more exact strings copied from that conditioned "
                    "request's belief.missing_roles"
                ),
            },
            "contract": {
                "every_conditioned_request_is_independent": True,
                "shared_fields_apply_to_every_conditioned_request_in_group": True,
                "predicted_only": True,
                "target_descriptor_is_backend_bound": True,
                "target_semantic_key_is_an_address_level_observation_clue": True,
                "missing_acquired_evidence_is_not_an_empty_outcome_signal": True,
                "resolved_roles_must_be_exact_missing_role_copies": True,
                "do_not_rephrase_or_invent_resolved_roles": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        task = (
            "Predict one categorical observation and belief delta for every "
            "conditioned request inside every shared legal-action group. Return one "
            "flat prediction row for every conditioned-request alias. A read action "
            "asks what inspecting its target semantic address is likely to reveal; "
            "use that address clue, the question, the hypothesis, and any imagined "
            "prefix to predict its future effect. An empty acquired-evidence list "
            "only means that no real read has happened yet and is not evidence that "
            "the target read will be empty. Do not select an action and do not emit "
            "numbers."
        )
        for attempt in range(2):
            result = self.client.complete_json(
                task=(task if attempt == 0 else _TRANSITION_REPAIR_TASK),
                payload=payload,
            )
            try:
                if not isinstance(result, dict) or _contains_number(result):
                    raise ValueError(
                        "categorical transition response schema is invalid"
                    )
                if set(result) == {"predictions"}:
                    raw_predictions = result.get("predictions")
                elif set(result) == set(aliases):
                    raw_predictions = result
                else:
                    raise ValueError(
                        "categorical transition response schema is invalid"
                    )
                rows, normalized = _normalize_alias_rows(
                    raw_predictions,
                    tuple(aliases),
                    alias_fields=("action", "alias", "outcome_id", "outcome_alias"),
                    context="categorical transition",
                )
                self._transition_sequence_normalizations += int(normalized)
                return tuple(
                    _parse_transition(aliases[alias], rows[alias], graph)
                    for alias in aliases
                )
            except (TypeError, ValueError) as exc:
                if "finish_reason=length" in str(exc):
                    raise
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable categorical transition repair state")

    def compare_batch(
        self,
        pool: TrajectoryPool,
        pairs: Sequence[HypothesisPathPair],
        graph: RetainedEvidenceGraph,
        *,
        include_imagined_transitions: bool,
    ) -> Sequence[HypothesisPathPreference]:
        normalization_start = self._comparison_sequence_normalizations
        comparisons: list[HypothesisPathPreference] = []
        ordered = tuple(sorted(pairs, key=lambda row: row.comparison_id))
        for start in range(0, len(ordered), self.comparison_batch_size):
            comparisons.extend(
                self._compare_one_batch(
                    pool,
                    ordered[start : start + self.comparison_batch_size],
                    graph,
                    include_imagined_transitions=include_imagined_transitions,
                )
            )
        by_id = {row.comparison_id: row for row in comparisons}
        result = tuple(by_id[row.comparison_id] for row in pairs)
        self.transport_audits.append(
            {
                "operation": "categorical_path_preference",
                "item_count": len(pairs),
                "batch_count": _batch_count(len(pairs), self.comparison_batch_size),
                "complete_coverage": len(result) == len(pairs),
                "ordered_sequence_normalization_count": (
                    self._comparison_sequence_normalizations - normalization_start
                ),
                "canonical_transport_order": True,
                "top_k_applied": False,
                "imagined_transitions_visible": include_imagined_transitions,
            }
        )
        return result

    def select_frontier(
        self,
        pool: TrajectoryPool,
        paths: Sequence[ImaginedHypothesisPath],
        graph: RetainedEvidenceGraph,
        *,
        include_imagined_transitions: bool,
    ) -> tuple[str, ...]:
        """Jointly compare every chain and return the categorical frontier."""

        ordered = tuple(sorted(paths, key=lambda row: row.path_id))
        aliases = {_alias("chain", index): path for index, path in enumerate(ordered)}
        payload = {
            "question": pool.trajectories[0].belief.question,
            "acquired_real_evidence": _acquired_evidence_payload(pool, graph),
            "candidate_chains": {
                alias: _path_payload(
                    path,
                    graph,
                    include_imagined_transitions=include_imagined_transitions,
                )
                for alias, path in aliases.items()
            },
            "allowed_status": ["unique", "tie", "incomparable"],
            "required_output": {
                "only_keys": ["status", "preferred", "rationale"],
                "preferred": "exact chain aliases from candidate_chains",
            },
            "contract": {
                "jointly_compare_every_candidate_chain": True,
                "return_complete_undominated_frontier": True,
                "ties_and_incomparability_are_valid": True,
                "no_candidate_pruning_or_top_k": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        task = (
            "Jointly compare every complete reasoning chain using its predicted "
            "future belief effects. Return the complete categorical preferred "
            "frontier, preserving ties and incomparability. Do not rank with numbers."
            if include_imagined_transitions
            else (
                "Jointly compare every current-belief reasoning action without "
                "imagined transitions. Return the complete categorical preferred "
                "frontier, preserving ties and incomparability."
            )
        )
        for attempt in range(2):
            result = self.client.complete_json(
                task=(task if attempt == 0 else _REPAIR_TASK),
                payload=payload,
            )
            try:
                if (
                    not isinstance(result, dict)
                    or set(result) != {"status", "preferred", "rationale"}
                    or _contains_number(result)
                ):
                    raise ValueError("setwise preference response schema is invalid")
                status = str(result.get("status") or "")
                preferred_aliases = _strings(result.get("preferred"), "preferred")
                if not set(preferred_aliases) <= set(aliases):
                    raise ValueError("setwise preference returned an unknown chain")
                if status == "unique" and len(preferred_aliases) != 1:
                    raise ValueError("unique setwise preference requires one chain")
                if status == "tie" and len(preferred_aliases) < 2:
                    raise ValueError("tied setwise preference requires multiple chains")
                if status == "incomparable" and len(preferred_aliases) < 2:
                    raise ValueError(
                        "incomparable setwise preference requires the complete "
                        "multi-chain frontier"
                    )
                if status not in {"unique", "tie", "incomparable"}:
                    raise ValueError("setwise preference status is invalid")
                by_alias = {alias: path.path_id for alias, path in aliases.items()}
                selected = tuple(by_alias[alias] for alias in preferred_aliases)
                self.transport_audits.append(
                    {
                        "operation": "categorical_setwise_frontier",
                        "item_count": len(ordered),
                        "batch_count": 1,
                        "complete_coverage": True,
                        "top_k_applied": False,
                        "imagined_transitions_visible": (include_imagined_transitions),
                    }
                )
                return selected
            except (TypeError, ValueError):
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable setwise preference repair state")

    def _compare_one_batch(
        self,
        pool: TrajectoryPool,
        pairs: Sequence[HypothesisPathPair],
        graph: RetainedEvidenceGraph,
        *,
        include_imagined_transitions: bool,
    ) -> tuple[HypothesisPathPreference, ...]:
        aliases = {_alias("comparison", index): row for index, row in enumerate(pairs)}
        payload = {
            "question": pool.trajectories[0].belief.question,
            "acquired_real_evidence": _acquired_evidence_payload(pool, graph),
            "comparisons": {
                alias: {
                    "left": _path_payload(
                        pair.left,
                        graph,
                        include_imagined_transitions=include_imagined_transitions,
                    ),
                    "right": _path_payload(
                        pair.right,
                        graph,
                        include_imagined_transitions=include_imagined_transitions,
                    ),
                }
                for alias, pair in aliases.items()
            },
            "allowed_labels": [row.value for row in PreferenceLabel],
            "required_output": {
                "only_key": "comparisons",
                "one_row_per_comparison": list(aliases),
                "fields": ["label", "rationale"],
            },
            "contract": {
                "pairwise_categorical_preference_only": True,
                "ties_and_incomparability_are_valid": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        task = (
            "Compare each pair using its predicted future belief effects and return "
            "only categorical preferences."
            if include_imagined_transitions
            else "Compare each pair as a reactive current-belief policy. Do not infer "
            "or use imagined future transitions."
        )
        for attempt in range(2):
            result = self.client.complete_json(
                task=(task if attempt == 0 else _REPAIR_TASK),
                payload=payload,
            )
            try:
                if (
                    not isinstance(result, dict)
                    or set(result) != {"comparisons"}
                    or _contains_number(result)
                ):
                    raise ValueError(
                        "categorical preference response schema is invalid"
                    )
                rows, normalized = _normalize_alias_rows(
                    result.get("comparisons"),
                    tuple(aliases),
                    alias_fields=("comparison", "comparison_id"),
                    context="categorical preference",
                )
                self._comparison_sequence_normalizations += int(normalized)
                parsed: list[HypothesisPathPreference] = []
                for alias, pair in aliases.items():
                    row = rows[alias]
                    if not isinstance(row, dict) or set(row) != {"label", "rationale"}:
                        raise ValueError("categorical preference row schema is invalid")
                    parsed.append(
                        HypothesisPathPreference(
                            comparison_id=pair.comparison_id,
                            left_id=pair.left.path_id,
                            right_id=pair.right.path_id,
                            label=PreferenceLabel(str(row.get("label") or "")),
                            rationale=str(row.get("rationale") or ""),
                        )
                    )
                return tuple(parsed)
            except (TypeError, ValueError):
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable categorical preference repair state")


class ShuffledHypothesisWorldModel:
    """Deterministically rotate categorical consequences while retaining actions."""

    def __init__(self, delegate: CategoricalHypothesisWorldModel) -> None:
        self.delegate = delegate
        self.model_name = f"shuffled:{delegate.model_name}"
        self.audits: list[dict[str, Any]] = []

    def predict_batch(
        self,
        requests: Sequence[HypothesisExpansionRequest],
        graph: RetainedEvidenceGraph,
    ) -> Sequence[ImaginedTransition]:
        predictions = tuple(self.delegate.predict_batch(requests, graph))
        read_indices = [
            index
            for index, request in enumerate(requests)
            if request.action.reads_evidence
        ]
        if len(read_indices) < 2:
            return predictions
        donors = [predictions[index] for index in read_indices]
        rotated = donors[1:] + donors[:1]
        donor_by_index = dict(zip(read_indices, rotated))
        shuffled_rows: list[ImaginedTransition] = []
        for index, (request, prediction) in enumerate(zip(requests, predictions)):
            donor = donor_by_index.get(index)
            if donor is None:
                # Terminal/cursor-only controls are protocol invariants, not
                # learned consequences, and therefore are never intervened on.
                shuffled_rows.append(prediction)
                continue
            shuffled_rows.append(
                ImaginedTransition(
                    action=request.action,
                    observation=replace(
                        donor.observation,
                        target_id=request.action.target_id,
                        descriptor=_target_descriptor(request.action, graph),
                    ),
                    belief_delta=donor.belief_delta,
                    structured_patch=donor.structured_patch,
                )
            )
        shuffled = tuple(shuffled_rows)
        self.audits.append(
            {
                "request_count": len(requests),
                "intervened_read_count": len(read_indices),
                "preserved_no_read_count": len(requests) - len(read_indices),
                "permutation": "deterministic_left_rotation",
                "top_k_applied": False,
            }
        )
        return shuffled


def compile_hypothesis_requests(
    pool: TrajectoryPool,
    graph: RetainedEvidenceGraph,
    compiler: GraphActionCompiler | None = None,
) -> tuple[HypothesisExpansionRequest, ...]:
    compiler = compiler or GraphActionCompiler()
    return tuple(
        sorted(
            (
                _request(
                    trajectory_id=trajectory.trajectory_id,
                    hypothesis=trajectory.hypothesis,
                    belief=trajectory.belief,
                    action=action,
                )
                for trajectory in pool.expandable
                for action in compiler.compile(trajectory.belief, graph)
            ),
            key=lambda row: row.request_id,
        )
    )


def _active_planning_requests(
    requests: Sequence[HypothesisExpansionRequest],
) -> tuple[HypothesisExpansionRequest, ...]:
    """Separate evidence acquisition from terminal control decisions.

    A ready trajectory may answer. Otherwise all available navigation actions
    remain legal and terminal controls do not compete with evidence reads.
    Abstention is retained only as a fail-closed controller outcome.
    """

    grouped: dict[str, list[HypothesisExpansionRequest]] = {}
    for request in requests:
        grouped.setdefault(request.trajectory_id, []).append(request)
    selected: list[HypothesisExpansionRequest] = []
    for rows in grouped.values():
        belief = rows[0].belief
        if belief.answerability is AnswerabilityState.READY:
            answers = [row for row in rows if row.action.kind is ActionKind.ANSWER]
            if answers:
                selected.extend(answers)
                continue
        navigation = [row for row in rows if row.action.kind not in TERMINAL_ACTIONS]
        if navigation:
            selected.extend(navigation)
            continue
        selected.extend(row for row in rows if row.action.kind is ActionKind.ABSTAIN)
    return tuple(sorted(selected, key=lambda row: row.request_id))


def _compile_trajectory_requests(
    trajectory_id: str,
    hypothesis: str,
    belief: CursorBeliefState,
    graph: RetainedEvidenceGraph,
    compiler: GraphActionCompiler,
    *,
    imagined_history: tuple[ImaginedTransition, ...],
) -> tuple[HypothesisExpansionRequest, ...]:
    return tuple(
        sorted(
            (
                _request(
                    trajectory_id=trajectory_id,
                    hypothesis=hypothesis,
                    belief=belief,
                    action=action,
                    imagined_history=imagined_history,
                )
                for action in compiler.compile(belief, graph)
            ),
            key=lambda row: row.request_id,
        )
    )


def audit_permutation_invariance(
    paths: Sequence[ImaginedHypothesisPath],
    pairs: Sequence[HypothesisPathPair],
    preferences: Sequence[HypothesisPathPreference],
) -> dict[str, Any]:
    """Mechanically verify that aggregation ignores candidate input order."""

    forward = _undominated_paths(tuple(paths), tuple(pairs), tuple(preferences))
    reverse = _undominated_paths(
        tuple(reversed(paths)), tuple(reversed(pairs)), tuple(reversed(preferences))
    )
    return {
        "preferred_path_set_equal": set(forward) == set(reverse),
        "forward_preferred": list(forward),
        "reverse_preferred": list(reverse),
        "top_k_applied": False,
    }


def _request(
    *,
    trajectory_id: str,
    hypothesis: str,
    belief: CursorBeliefState,
    action: LegalGraphAction,
    imagined_history: tuple[ImaginedTransition, ...] = (),
    trajectory_contexts: tuple[tuple[str, str, CursorBeliefState], ...] = (),
) -> HypothesisExpansionRequest:
    payload = "\x1f".join(
        (
            trajectory_id,
            *(row.action.action_id for row in imagined_history),
            action.action_id,
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return HypothesisExpansionRequest(
        request_id=f"hypothesis-expansion:{digest}",
        trajectory_id=trajectory_id,
        hypothesis=hypothesis,
        belief=belief,
        action=action,
        imagined_history=imagined_history,
        trajectory_contexts=trajectory_contexts,
    )


def _path(
    root: HypothesisExpansionRequest,
    transitions: tuple[ImaginedTransition, ...],
    *,
    continuations: tuple[ImaginedTransition, ...] = (),
) -> ImaginedHypothesisPath:
    digest = hashlib.sha256(
        "\x1e".join(
            (root.trajectory_id, *(row.action.action_id for row in transitions))
        ).encode("utf-8")
    ).hexdigest()[:20]
    return ImaginedHypothesisPath(
        path_id=f"hypothesis-path:{digest}",
        trajectory_id=root.trajectory_id,
        hypothesis=root.hypothesis,
        transitions=transitions,
        continuations=tuple(
            sorted(continuations, key=lambda row: row.action.action_id)
        ),
    )


def _joint_reasoning_chains(
    paths: Sequence[ImaginedHypothesisPath],
) -> tuple[ImaginedHypothesisPath, ...]:
    """Group all hypothesis-conditioned chains under their shared first hop.

    Persistent hypotheses may retain different graph cursors after a shared
    read. Grouping by exact two-hop sequence then creates partial "joint"
    candidates. A Planner decision executes only the first real hop, so the
    correct comparison unit is the complete first-hop action tree containing
    every predicted one-/two-hop continuation for every hypothesis.
    """

    grouped: dict[tuple[str, ...], list[ImaginedHypothesisPath]] = {}
    for path in paths:
        key = shared_action_key(path.first_action)
        grouped.setdefault(key, []).append(path)
    result: list[ImaginedHypothesisPath] = []
    for first_hop_key, members in grouped.items():
        ordered_members = tuple(
            sorted(
                members,
                key=lambda row: (
                    row.trajectory_id,
                    tuple(value.action.action_id for value in row.transitions),
                ),
            )
        )
        representative = ordered_members[0]
        digest = hashlib.sha256("\x1e".join(first_hop_key).encode("utf-8")).hexdigest()[
            :20
        ]
        result.append(
            ImaginedHypothesisPath(
                path_id=f"joint-first-hop-tree:{digest}",
                trajectory_id="trajectory-pool",
                hypothesis="competing trajectory pool",
                transitions=(representative.transitions[0],),
                conditioned_outcomes=tuple(
                    HypothesisConditionedOutcome(
                        trajectory_id=member.trajectory_id,
                        hypothesis=member.hypothesis,
                        transitions=member.transitions,
                    )
                    for member in ordered_members
                ),
            )
        )
    return tuple(sorted(result, key=lambda row: row.path_id))


def _pair(
    left: ImaginedHypothesisPath, right: ImaginedHypothesisPath
) -> HypothesisPathPair:
    digest = hashlib.sha256(
        f"{left.path_id}\x1f{right.path_id}".encode("utf-8")
    ).hexdigest()[:20]
    return HypothesisPathPair(f"path-comparison:{digest}", left, right)


def _undominated_paths(
    paths: Sequence[ImaginedHypothesisPath],
    pairs: Sequence[HypothesisPathPair],
    preferences: Sequence[HypothesisPathPreference],
) -> tuple[str, ...]:
    if len(pairs) != len(preferences):
        raise ValueError("complete pairwise preference coverage is required")
    dominated: set[str] = set()
    for pair, preference in zip(pairs, preferences):
        if (
            preference.comparison_id != pair.comparison_id
            or preference.left_id != pair.left.path_id
            or preference.right_id != pair.right.path_id
        ):
            raise ValueError("preference does not match its canonical path pair")
        if preference.label is PreferenceLabel.PREFER_LEFT:
            dominated.add(pair.right.path_id)
        elif preference.label is PreferenceLabel.PREFER_RIGHT:
            dominated.add(pair.left.path_id)
    return tuple(
        sorted(path.path_id for path in paths if path.path_id not in dominated)
    )


def _transition_request_payload(
    request: HypothesisExpansionRequest,
    graph: RetainedEvidenceGraph,
) -> dict[str, Any]:
    return {
        "question": request.belief.question,
        "hypothesis": request.hypothesis,
        "belief": _transition_belief_payload(request),
        "acquired_real_evidence": _transition_acquired_evidence_payload(
            request.belief,
            graph,
        ),
        "action": _transition_action_payload(
            request.action,
            request.belief,
            graph,
        ),
        "imagined_prefix": [
            _transition_descriptor(row, graph) for row in request.imagined_history
        ],
    }


def _group_transition_requests(
    requests: Sequence[HypothesisExpansionRequest],
) -> tuple[tuple[HypothesisExpansionRequest, ...], ...]:
    """Losslessly group exact legal action sequences across hypotheses."""

    grouped: dict[tuple[str, ...], list[HypothesisExpansionRequest]] = {}
    request_ids: set[str] = set()
    for request in requests:
        if request.request_id in request_ids:
            raise ValueError("duplicate hypothesis expansion request")
        request_ids.add(request.request_id)
        key = (
            *(row.action.action_id for row in request.imagined_history),
            request.action.action_id,
        )
        members = grouped.setdefault(key, [])
        if members and (
            members[0].action != request.action
            or tuple(row.action for row in members[0].imagined_history)
            != tuple(row.action for row in request.imagined_history)
        ):
            raise ValueError("action sequence identity collision")
        members.append(request)
    return tuple(tuple(members) for members in grouped.values())


def _transition_acquired_evidence_payload(
    belief: CursorBeliefState,
    graph: RetainedEvidenceGraph,
) -> list[dict[str, Any]]:
    graph_input = build_iwm_graph_input(belief, graph, ())
    return [
        {
            "semantic_key": row.key.semantic_key,
            "evidence_value": row.evidence_value,
        }
        for row in graph_input.nodes
        if row.acquired and row.evidence_value is not None
    ]


def _transition_action_payload(
    action: LegalGraphAction,
    belief: CursorBeliefState,
    graph: RetainedEvidenceGraph,
) -> dict[str, Any]:
    graph_input = build_iwm_graph_input(belief, graph, (action,))
    target = next(
        (row for row in graph_input.nodes if row.key.node_id == action.target_id),
        None,
    )
    return {
        "kind": action.kind.value,
        "relation": action.relation,
        "target_semantic_key": target.key.semantic_key if target else None,
        "target_structural_tags": list(target.key.structural_tags) if target else [],
        "target_time_span": (
            {"start_s": target.key.start_s, "end_s": target.key.end_s}
            if target
            else None
        ),
    }


def _transition_condition_payload(
    request: HypothesisExpansionRequest,
    graph: RetainedEvidenceGraph,
) -> dict[str, Any]:
    return {
        "hypothesis": request.hypothesis,
        "belief": _transition_belief_payload(request),
        "imagined_prefix": [
            _transition_descriptor(row, graph) for row in request.imagined_history
        ],
    }


def _transition_belief_payload(
    request: HypothesisExpansionRequest,
) -> dict[str, Any]:
    return {
        "missing_roles": list(request.belief.missing_roles),
        "contradictions": list(request.belief.contradictions),
        "answerability": request.belief.answerability.value,
    }


def _parse_transition(
    request: HypothesisExpansionRequest,
    row: Any,
    graph: RetainedEvidenceGraph,
) -> ImaginedTransition:
    expected = {
        "outcome",
        "progress",
        "answerability_after",
        "frontier_change",
        "contradiction_change",
        "resolved_roles",
        "opened_roles",
        "relation_updates",
        "rationale",
    }
    if not isinstance(row, dict) or set(row) != expected or _contains_number(row):
        raise ValueError("categorical transition row schema is invalid")
    resolved = _strings(row.get("resolved_roles"), "resolved_roles")
    known_roles = set(request.belief.required_roles or request.belief.missing_roles)
    if not set(resolved).issubset(known_roles):
        raise ValueError("imagined transition resolved an unknown role")
    currently_missing = set(request.belief.missing_roles)
    resolved = tuple(role for role in resolved if role in currently_missing)
    opened = _strings(row.get("opened_roles"), "opened_roles")
    relation_updates = _strings(row.get("relation_updates"), "relation_updates")
    return ImaginedTransition(
        action=request.action,
        observation=PredictedObservation(
            target_id=request.action.target_id,
            outcome=EvidenceOutcome(str(row.get("outcome") or "")),
            descriptor=_target_descriptor(request.action, graph),
        ),
        belief_delta=CategoricalBeliefDelta(
            progress=ProgressChange(str(row.get("progress") or "")),
            answerability_after=AnswerabilityState(
                str(row.get("answerability_after") or "")
            ),
            frontier_change=FrontierChange(str(row.get("frontier_change") or "")),
            contradiction_change=ContradictionChange(
                str(row.get("contradiction_change") or "")
            ),
            resolved_roles=resolved,
            opened_roles=opened,
            relation_updates=relation_updates,
        ),
    )


def _path_payload(
    path: ImaginedHypothesisPath,
    graph: RetainedEvidenceGraph,
    *,
    include_imagined_transitions: bool,
) -> dict[str, Any]:
    result = {
        "actions": [
            {
                "kind": row.action.kind.value,
                "relation": row.action.relation,
                "target_semantic_key": _target_descriptor(row.action, graph),
            }
            for row in path.transitions
        ],
    }
    if include_imagined_transitions:
        result["hypothesis_conditioned_outcome_classes"] = _lossless_outcome_classes(
            path, graph
        )
        result["outcome_grouping_contract"] = (
            "exact_descriptor_equivalence_only; every conditioned member retained"
        )
    else:
        result["competing_hypotheses"] = [
            outcome.hypothesis for outcome in path.conditioned_outcomes
        ]
    return result


def _lossless_outcome_classes(
    path: ImaginedHypothesisPath,
    graph: RetainedEvidenceGraph,
) -> dict[str, Any]:
    """Deduplicate exact descriptors while retaining every hypothesis member."""

    grouped: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    for outcome in path.conditioned_outcomes:
        descriptors = [
            _transition_descriptor(row, graph) for row in outcome.transitions
        ]
        key = json.dumps(
            descriptors,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if key not in grouped:
            grouped[key] = {
                "imagined_transitions": descriptors,
                "conditioned_members": [],
            }
            ordered_keys.append(key)
        grouped[key]["conditioned_members"].append({"hypothesis": outcome.hypothesis})
    return {
        _alias("outcome_class", index): grouped[key]
        for index, key in enumerate(ordered_keys)
    }


def _transition_descriptor(
    transition: ImaginedTransition,
    graph: RetainedEvidenceGraph,
) -> dict[str, Any]:
    delta = transition.belief_delta
    return {
        "target_semantic_key": _target_descriptor(transition.action, graph),
        "outcome": transition.observation.outcome.value,
        "progress": delta.progress.value,
        "answerability_after": delta.answerability_after.value,
        "frontier_change": delta.frontier_change.value,
        "contradiction_change": delta.contradiction_change.value,
        "resolved_roles": list(delta.resolved_roles),
        "opened_roles": list(delta.opened_roles),
        "relation_updates": list(delta.relation_updates),
        "structured_belief_event_patch": transition.structured_patch,
    }


def _acquired_evidence_payload(
    pool: TrajectoryPool,
    graph: RetainedEvidenceGraph,
) -> list[dict[str, str]]:
    acquired = (
        set(pool.expandable[0].belief.acquired_evidence) if pool.expandable else set()
    )
    return [
        {
            "semantic_key": str(
                node.metadata.get("predicate") or node.text or node.node_id
            ),
            "evidence_value": str(node.text or node.metadata.get("predicate") or ""),
        }
        for node in graph.nodes
        if node.node_id in acquired
    ]


def _neutral_transition(request: HypothesisExpansionRequest) -> ImaginedTransition:
    return ImaginedTransition(
        action=request.action,
        observation=PredictedObservation(
            target_id=request.action.target_id,
            outcome=EvidenceOutcome.INCONCLUSIVE,
        ),
        belief_delta=CategoricalBeliefDelta(
            progress=ProgressChange.UNCHANGED,
            answerability_after=request.belief.answerability,
        ),
    )


def _target_descriptor(
    action: LegalGraphAction,
    graph: RetainedEvidenceGraph,
) -> tuple[str, ...]:
    if not action.reads_evidence or action.target_id is None:
        return ()
    node = graph.node_by_id.get(action.target_id)
    if node is None:
        raise ValueError("action target is absent from retained graph")
    return (str(node.metadata.get("predicate") or node.text or node.node_id),)


def _abstain_action(
    requests: Sequence[HypothesisExpansionRequest],
) -> LegalGraphAction:
    return next(
        request.action
        for request in requests
        if request.action.kind is ActionKind.ABSTAIN
    )


_REPAIR_TASK = (
    "Repair the JSON to exactly match the requested categorical schema, cover "
    "every alias, and emit no numeric values."
)

_TRANSITION_REPAIR_TASK = (
    "Repair the JSON to exactly match the requested categorical transition "
    "schema and cover every prediction alias. For each conditioned request, "
    "resolved_roles may contain only exact strings copied verbatim from that "
    "request's belief.missing_roles; do not rephrase, shorten, or invent role "
    "names. Emit no numeric values."
)

_SCHEDULER_REPAIR_TASK = (
    "Repair the JSON to exactly contain status, selected, and rationale. Status "
    "must be select with one exact evidence_action alias, or incomparable with "
    "selected set to none. Emit no numeric values."
)


def _contains_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_contains_number(row) for row in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_number(row) for row in value)
    return False


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None or value == "none":
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise ValueError(f"{name} must be a string list")
    rows = tuple(value)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{name} must not contain duplicates")
    return rows


def _normalize_alias_rows(
    value: Any,
    aliases: tuple[str, ...],
    *,
    alias_fields: tuple[str, ...],
    context: str,
) -> tuple[dict[str, Any], bool]:
    """Accept a complete alias map or an exact ordered transport sequence."""

    if isinstance(value, dict):
        if set(value) != set(aliases):
            raise ValueError(f"{context} response coverage mismatch")
        raw_rows = tuple((alias, value[alias]) for alias in aliases)
        ordered_sequence = False
    elif isinstance(value, list) and len(value) == len(aliases):
        raw_rows = tuple(zip(aliases, value))
        ordered_sequence = True
    else:
        raise ValueError(f"{context} response coverage mismatch")
    rows: dict[str, Any] = {}
    for alias, raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError(f"{context} ordered row is not an object")
        row = dict(raw)
        supplied = [row.pop(field) for field in alias_fields if field in row]
        if supplied and any(
            not isinstance(candidate, str) or candidate != alias
            for candidate in supplied
        ):
            raise ValueError(f"{context} ordered row alias mismatch")
        rows[alias] = row
    return rows, ordered_sequence


def _alias(prefix: str, index: int) -> str:
    value = index
    letters = ""
    while True:
        letters = chr(ord("a") + value % 26) + letters
        value = value // 26 - 1
        if value < 0:
            break
    return f"{prefix}_{letters}"


def _batch_count(size: int, width: int) -> int:
    return (size + width - 1) // width if size else 0
