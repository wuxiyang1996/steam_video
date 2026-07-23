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
    """A complete short chain with every matching hypothesis-conditioned outcome."""

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
            trajectory_ids = [
                row.trajectory_id for row in self.conditioned_outcomes
            ]
            if len(trajectory_ids) != len(set(trajectory_ids)):
                raise ValueError(
                    "a joint chain may contain one outcome per trajectory"
                )
            if any(
                tuple(value.action.action_id for value in row.transitions)
                != tuple(value.action.action_id for value in self.transitions)
                for row in self.conditioned_outcomes
            ):
                raise ValueError(
                    "joint-chain outcomes must share one legal action sequence"
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
        self.last_complete_coverage_audit: dict[str, Any] = {}

    def plan(
        self,
        pool: TrajectoryPool,
        graph: RetainedEvidenceGraph,
    ) -> MultiTrajectoryPlanDecision:
        requests = compile_hypothesis_requests(pool, graph, self.action_compiler)
        if not requests:
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
            for child in _compile_trajectory_requests(
                request.trajectory_id,
                request.hypothesis,
                imagined_belief,
                graph,
                self.action_compiler,
                imagined_history=(prediction,),
            ):
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
        if self.max_complete_pairs is not None and pair_count > self.max_complete_pairs:
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
                selected_action=_abstain_action(requests),
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
        first_hops = {
            shared_action_key(path.first_action): path.first_action
            for path in ordered_paths
            if path.path_id in set(preferred)
        }
        if len(first_hops) == 1:
            selected = next(iter(first_hops.values()))
            status = "rollout_selected_unique_preferred_first_hop"
        else:
            selected = _abstain_action(requests)
            status = (
                "rollout_abstain_preference_cycle"
                if not first_hops
                else "rollout_abstain_non_unique_first_hops"
            )
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
            "path_representation": "one_hypothesis_conditioned_reasoning_chain",
            "exhaustive_pair_count": len(pairs),
            "comparison_count": len(preferences),
            "complete_coverage": len(preferences) == len(pairs),
            "top_k_applied": False,
            "canonical_path_order": True,
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
        return predictions


class ReactiveMultiTrajectoryPlanner:
    """Matched no-WM arm: compare current legal actions without transitions."""

    def __init__(
        self,
        preference_model: CategoricalHypothesisPathPreference,
        *,
        action_compiler: GraphActionCompiler | None = None,
    ) -> None:
        self.preference_model = preference_model
        self.action_compiler = action_compiler or GraphActionCompiler()
        self.last_complete_coverage_audit: dict[str, Any] = {}

    def plan(
        self, pool: TrajectoryPool, graph: RetainedEvidenceGraph
    ) -> MultiTrajectoryPlanDecision:
        requests = compile_hypothesis_requests(pool, graph, self.action_compiler)
        raw_paths = tuple(
            _path(request, (_neutral_transition(request),)) for request in requests
        )
        paths = _joint_reasoning_chains(raw_paths)
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
        selected = (
            next(iter(first_hops.values()))
            if len(first_hops) == 1
            else _abstain_action(requests)
        )
        status = (
            "reactive_selected_unique_preferred_first_hop"
            if len(first_hops) == 1
            else "reactive_abstain_non_unique_first_hops"
        )
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
            "comparison_count": len(preferences),
            "complete_coverage": len(preferences) == len(pairs),
            "top_k_applied": False,
            "world_model_predictions_visible": False,
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
        transition_batch_size: int = 32,
        comparison_batch_size: int = 24,
    ) -> None:
        if transition_batch_size < 1 or comparison_batch_size < 1:
            raise ValueError("transport batch sizes must be positive")
        self.client = client
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))
        self.transition_batch_size = transition_batch_size
        self.comparison_batch_size = comparison_batch_size
        self.transport_audits: list[dict[str, Any]] = []

    def predict_batch(
        self,
        requests: Sequence[HypothesisExpansionRequest],
        graph: RetainedEvidenceGraph,
    ) -> Sequence[ImaginedTransition]:
        predictions: list[ImaginedTransition] = []
        for start in range(0, len(requests), self.transition_batch_size):
            batch = tuple(requests[start : start + self.transition_batch_size])
            predictions.extend(self._predict_one_batch(batch, graph))
        self.transport_audits.append(
            {
                "operation": "categorical_transition",
                "item_count": len(requests),
                "batch_count": _batch_count(len(requests), self.transition_batch_size),
                "complete_coverage": len(predictions) == len(requests),
                "top_k_applied": False,
            }
        )
        return tuple(predictions)

    def _predict_one_batch(
        self,
        requests: tuple[HypothesisExpansionRequest, ...],
        graph: RetainedEvidenceGraph,
    ) -> tuple[ImaginedTransition, ...]:
        aliases = {_alias("action", index): row for index, row in enumerate(requests)}
        payload = {
            "requests": {
                alias: _transition_request_payload(request, graph)
                for alias, request in aliases.items()
            },
            "allowed": {
                "outcome": [row.value for row in EvidenceOutcome],
                "progress": [row.value for row in ProgressChange],
                "answerability_after": [row.value for row in AnswerabilityState],
                "frontier_change": [row.value for row in FrontierChange],
                "contradiction_change": [row.value for row in ContradictionChange],
            },
            "required_output": {
                "only_key": "predictions",
                "one_row_per_request": list(aliases),
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
            },
            "contract": {
                "predicted_only": True,
                "target_descriptor_is_backend_bound": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        task = (
            "Predict one categorical observation and belief delta for every "
            "hypothesis-conditioned legal action. Do not select an action and do not "
            "emit numbers."
        )
        for attempt in range(2):
            result = self.client.complete_json(
                task=(task if attempt == 0 else _REPAIR_TASK),
                payload=payload,
            )
            try:
                if (
                    not isinstance(result, dict)
                    or set(result) != {"predictions"}
                    or _contains_number(result)
                ):
                    raise ValueError(
                        "categorical transition response schema is invalid"
                    )
                rows = result.get("predictions")
                if not isinstance(rows, dict) or set(rows) != set(aliases):
                    raise ValueError(
                        "categorical transition response coverage mismatch"
                    )
                return tuple(
                    _parse_transition(aliases[alias], rows[alias], graph)
                    for alias in aliases
                )
            except (TypeError, ValueError):
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
                "canonical_transport_order": True,
                "top_k_applied": False,
                "imagined_transitions_visible": include_imagined_transitions,
            }
        )
        return result

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
                rows = result.get("comparisons")
                if not isinstance(rows, dict) or set(rows) != set(aliases):
                    raise ValueError(
                        "categorical preference response coverage mismatch"
                    )
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
        if len(predictions) < 2:
            return predictions
        donors = predictions[1:] + predictions[:1]
        shuffled = tuple(
            ImaginedTransition(
                action=request.action,
                observation=replace(
                    donor.observation,
                    target_id=request.action.target_id,
                    descriptor=_target_descriptor(request.action, graph),
                ),
                belief_delta=donor.belief_delta,
            )
            for request, donor in zip(requests, donors)
        )
        self.audits.append(
            {
                "request_count": len(requests),
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
    """Group identical legal action sequences without dropping any hypothesis."""

    grouped: dict[tuple[str, ...], list[ImaginedHypothesisPath]] = {}
    for path in paths:
        key = tuple(row.action.action_id for row in path.transitions)
        grouped.setdefault(key, []).append(path)
    result: list[ImaginedHypothesisPath] = []
    for action_ids, members in grouped.items():
        ordered_members = tuple(sorted(members, key=lambda row: row.trajectory_id))
        representative = ordered_members[0]
        digest = hashlib.sha256(
            "\x1e".join(action_ids).encode("utf-8")
        ).hexdigest()[:20]
        result.append(
            ImaginedHypothesisPath(
                path_id=f"joint-reasoning-chain:{digest}",
                trajectory_id="trajectory-pool",
                hypothesis="competing trajectory pool",
                transitions=representative.transitions,
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
    actions = (request.action,)
    graph_input = build_iwm_graph_input(request.belief, graph, actions)
    target = next(
        (
            row
            for row in graph_input.nodes
            if row.key.node_id == request.action.target_id
        ),
        None,
    )
    return {
        "question": request.belief.question,
        "hypothesis": request.hypothesis,
        "belief": {
            "missing_roles": list(request.belief.missing_roles),
            "contradictions": list(request.belief.contradictions),
            "answerability": request.belief.answerability.value,
            "acquired_real_evidence": [
                {
                    "semantic_key": row.key.semantic_key,
                    "evidence_value": row.evidence_value,
                }
                for row in graph_input.nodes
                if row.acquired
            ],
        },
        "action": {
            "kind": request.action.kind.value,
            "relation": request.action.relation,
            "target_semantic_key": target.key.semantic_key if target else None,
            "target_structural_tags": list(target.key.structural_tags)
            if target
            else [],
            "target_time_span": (
                {"start_s": target.key.start_s, "end_s": target.key.end_s}
                if target
                else None
            ),
        },
        "imagined_prefix": [
            _transition_descriptor(row, graph) for row in request.imagined_history
        ],
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
    allowed_missing_roles = set(request.belief.missing_roles)
    if not set(resolved).issubset(allowed_missing_roles):
        raise ValueError("imagined transition resolved an unknown role")
    opened = _strings(row.get("opened_roles"), "opened_roles")
    known_relations = _known_relation_ids(graph)
    relation_updates = _strings(row.get("relation_updates"), "relation_updates")
    if not set(relation_updates).issubset(known_relations):
        raise ValueError("imagined transition referenced an unknown relation")
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
        result["hypothesis_conditioned_outcomes"] = [
            {
                "hypothesis": outcome.hypothesis,
                "imagined_transitions": [
                    _transition_descriptor(row, graph)
                    for row in outcome.transitions
                ],
            }
            for outcome in path.conditioned_outcomes
        ]
    else:
        result["competing_hypotheses"] = [
            outcome.hypothesis for outcome in path.conditioned_outcomes
        ]
    return result


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


def _known_relation_ids(graph: RetainedEvidenceGraph) -> set[str]:
    return {
        row.edge_id
        for rows in (
            graph.temporal_edges,
            graph.correlation_edges,
            graph.candidate_edges,
            graph.verified_relations,
        )
        for row in rows
    }


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
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise ValueError(f"{name} must be a string list")
    rows = tuple(value)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{name} must not contain duplicates")
    return rows


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
