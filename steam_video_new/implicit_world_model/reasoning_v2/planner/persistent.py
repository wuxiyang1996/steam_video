"""Persistent multi-path planning over L1/L1.5 evidence memory."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
from typing import Sequence

from ..evidence.contracts import EvidenceMemory
from ..navigation.actions import compile_legal_actions
from ..navigation.contracts import (
    NavigationAction,
    NavigationGraph,
)
from ..world_model.contracts import (
    Answerability,
    BeliefEffect,
    BeliefState,
    HypothesisEffectModel,
    HypothesisEffectRequest,
    ObservationContext,
    ObservationDescriptor,
    ObservationPrediction,
    ObservationRequest,
    ObservationWorldModel,
    RealEffectCorrector,
    RealObservation,
    observation_world_key,
    project_belief,
)
from ..world_model.validation import validate_observation_predictions
from .contracts import (
    CandidateReasoningTrajectory,
    CategoricalTrajectoryPlannerModel,
    ImaginedReasoningStep,
    JointActionTree,
    PathStatus,
    PlanningDecision,
    ReasoningPath,
    ReasoningPathForest,
    SharedEvidenceState,
)


class PersistentMultiPathPlanner:
    """Imagine all local paths, choose once, and retain unexecuted alternatives."""

    def __init__(
        self,
        observation_model: ObservationWorldModel,
        effect_model: HypothesisEffectModel,
        preference_model: CategoricalTrajectoryPlannerModel,
        *,
        horizon: int = 2,
    ) -> None:
        if horizon not in {1, 2}:
            raise ValueError("planner horizon must be one or two")
        self.observation_model = observation_model
        self.effect_model = effect_model
        self.preference_model = preference_model
        self.horizon = horizon

    def plan(
        self,
        forest: ReasoningPathForest,
        memory: EvidenceMemory,
        navigation: NavigationGraph,
    ) -> PlanningDecision:
        if forest.shared_evidence.remaining_reads == 0:
            return _empty_decision(self.horizon)
        first = _first_expansions(forest, memory, navigation)
        if not first:
            return _empty_decision(self.horizon)
        first_predictions, first_prediction_count = _predict_observations(
            self.observation_model,
            tuple(row.observation_request for row in first),
        )
        observation_by_id = {row.request_id: row for row in first_predictions}
        first_effect_requests = tuple(
            HypothesisEffectRequest(
                request_id=f"effect:{row.expansion_id}",
                path_id=row.path.path_id,
                hypothesis=row.path.hypothesis,
                belief=row.path.belief,
                observation=observation_by_id[row.observation_request.request_id],
            )
            for row in first
        )
        first_effects = tuple(
            self.effect_model.predict_effect_batch(first_effect_requests)
        )
        _validate_effects(first_effect_requests, first_effects)
        first_effect_by_id = {row.request_id: row for row in first_effects}

        candidates: list[CandidateReasoningTrajectory] = []
        second_units: list[_SecondExpansion] = []
        for row in first:
            observation = observation_by_id[row.observation_request.request_id]
            effect = first_effect_by_id[f"effect:{row.expansion_id}"]
            first_step = ImaginedReasoningStep(row.action, observation, effect)
            if self.horizon == 1:
                candidates.append(_candidate(row.path, (first_step,)))
                continue
            projected = replace(
                row.path,
                belief=project_belief(row.path.belief, effect),
                cursor_id=row.action.target_id,
                action_history=(*row.path.action_history, row.action.action_id),
                status=PathStatus.ACTIVE,
                pending_first_action_id=None,
                pending_candidate_ids=(),
            )
            imagined_acquired = tuple(
                dict.fromkeys(
                    (*forest.shared_evidence.acquired_ids, str(row.action.target_id))
                )
            )
            second_actions = tuple(
                action
                for action in compile_legal_actions(
                    navigation,
                    cursor_id=projected.cursor_id,
                    acquired_ids=imagined_acquired,
                )
                if action.reads_evidence
            )
            if not second_actions:
                candidates.append(_candidate(row.path, (first_step,)))
                continue
            for action in second_actions:
                request = _observation_request(
                    forest,
                    memory,
                    action,
                    suffix=f"{row.expansion_id}:second:{action.action_id}",
                    imagined_prefix=((observation.target_id, observation.descriptor),),
                )
                second_units.append(
                    _SecondExpansion(row.path, first_step, projected, action, request)
                )
        second_prediction_count = 0
        second_effect_count = 0
        if second_units:
            second_predictions, second_prediction_count = _predict_observations(
                self.observation_model,
                tuple(row.observation_request for row in second_units),
            )
            second_prediction_by_id = {
                row.request_id: row for row in second_predictions
            }
            effect_requests = tuple(
                HypothesisEffectRequest(
                    request_id=f"effect:second:{row.observation_request.request_id}",
                    path_id=row.root.path_id,
                    hypothesis=row.root.hypothesis,
                    belief=row.projected.belief,
                    observation=second_prediction_by_id[
                        row.observation_request.request_id
                    ],
                )
                for row in second_units
            )
            effects = tuple(self.effect_model.predict_effect_batch(effect_requests))
            _validate_effects(effect_requests, effects)
            second_effect_count = len(effects)
            effect_by_id = {row.request_id: row for row in effects}
            for row in second_units:
                observation = second_prediction_by_id[
                    row.observation_request.request_id
                ]
                effect = effect_by_id[
                    f"effect:second:{row.observation_request.request_id}"
                ]
                candidates.append(
                    _candidate(
                        row.root,
                        (
                            row.first_step,
                            ImaginedReasoningStep(row.action, observation, effect),
                        ),
                    )
                )
        ordered = _joint_action_trees(tuple(candidates))
        _validate_joint_coverage(forest, ordered)
        preference = self.preference_model.choose(forest, ordered)
        known = {row.tree_id: row for row in ordered}
        if not set(preference.frontier_candidate_ids).issubset(known):
            raise ValueError("planner preference contains an unknown trajectory")
        selected = (
            known[preference.selected_candidate_id].first_action
            if preference.selected_candidate_id is not None
            else None
        )
        return PlanningDecision(
            candidates=ordered,
            preference=preference,
            selected_action=selected,
            observation_prediction_count=first_prediction_count
            + second_prediction_count,
            effect_prediction_count=len(first_effects) + second_effect_count,
            horizon=self.horizon,
        )


def apply_real_read(
    forest: ReasoningPathForest,
    decision: PlanningDecision,
    memory: EvidenceMemory,
    *,
    corrector: RealEffectCorrector | None = None,
) -> ReasoningPathForest:
    """Execute one read while retaining every unselected reasoning alternative."""

    action = decision.selected_action
    if action is None or not action.reads_evidence or action.target_id is None:
        return forest
    if forest.shared_evidence.remaining_reads == 0:
        raise ValueError("cannot execute a read after budget exhaustion")
    record = memory.read(action.target_id)
    by_root: dict[str, list[CandidateReasoningTrajectory]] = defaultdict(list)
    for tree in decision.candidates:
        for candidate in tree.trajectories:
            by_root[candidate.root_path_id].append(candidate)
    root_by_id = {row.path_id: row for row in forest.paths}
    next_paths: list[ReasoningPath] = []
    for root_id, candidates in by_root.items():
        root = root_by_id[root_id]
        by_first: dict[str, list[CandidateReasoningTrajectory]] = defaultdict(list)
        for candidate in candidates:
            by_first[candidate.first_action.action_id].append(candidate)
        for first_action_id, alternatives in sorted(by_first.items()):
            first_action = alternatives[0].first_action
            selected = first_action.shared_key == action.shared_key
            identity = f"{root.path_id}\x1f{first_action_id}\x1f{forest.step + 1}"
            child_id = (
                f"reasoning-path:{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
            )
            next_paths.append(
                ReasoningPath(
                    path_id=child_id,
                    hypothesis=root.hypothesis,
                    belief=root.belief,
                    cursor_id=(action.target_id if selected else root.cursor_id),
                    action_history=(
                        (*root.action_history, action.action_id)
                        if selected
                        else root.action_history
                    ),
                    interpreted_observation_ids=(
                        (*root.interpreted_observation_ids, action.target_id)
                        if selected
                        else root.interpreted_observation_ids
                    ),
                    parent_path_id=root.path_id,
                    status=PathStatus.ACTIVE if selected else PathStatus.SUSPENDED,
                    pending_first_action_id=None if selected else first_action_id,
                    pending_candidate_ids=tuple(
                        row.candidate_id for row in alternatives
                    ),
                )
            )
    planned_roots = set(by_root)
    next_paths.extend(row for row in forest.paths if row.path_id not in planned_roots)
    shared = SharedEvidenceState(
        acquired_ids=tuple(
            dict.fromkeys((*forest.shared_evidence.acquired_ids, action.target_id))
        ),
        remaining_reads=forest.shared_evidence.remaining_reads - 1,
    )
    if corrector is not None:
        real = RealObservation(action, record.value)
        effects = tuple(
            corrector.correct_batch(
                path_hypotheses={row.path_id: row.hypothesis for row in next_paths},
                beliefs={row.path_id: row.belief for row in next_paths},
                acquired_ids_before=forest.shared_evidence.acquired_ids,
                observation=real,
            )
        )
        effect_by_id = {row.path_id: row for row in effects}
        if set(effect_by_id) != {row.path_id for row in next_paths}:
            raise ValueError("real effect correction coverage mismatch")
        next_paths = [
            replace(
                row,
                belief=(
                    effect_by_id[row.path_id].belief_after
                    if effect_by_id[row.path_id].verified
                    and effect_by_id[row.path_id].direct_same_target
                    else row.belief
                ),
            )
            for row in next_paths
        ]
    return ReasoningPathForest(
        forest_id=f"{forest.forest_id}:step:{forest.step + 1}",
        paths=_exact_path_deduplicate(tuple(next_paths)),
        shared_evidence=shared,
        step=forest.step + 1,
        top_k_applied=False,
    )


def initialize_forest(
    *,
    forest_id: str,
    belief: BeliefState,
    hypotheses: Sequence[str],
    read_budget: int,
) -> ReasoningPathForest:
    if not hypotheses or len(hypotheses) != len(set(hypotheses)):
        raise ValueError("initial reasoning hypotheses must be unique")
    paths = tuple(
        ReasoningPath(
            path_id=f"reasoning-path:{hashlib.sha256(f'{forest_id}:{index}:{value}'.encode()).hexdigest()[:20]}",
            hypothesis=value,
            belief=belief,
        )
        for index, value in enumerate(hypotheses)
    )
    return ReasoningPathForest(
        forest_id=forest_id,
        paths=paths,
        shared_evidence=SharedEvidenceState(remaining_reads=read_budget),
    )


@dataclass(frozen=True)
class _FirstExpansion:
    expansion_id: str
    path: ReasoningPath
    action: NavigationAction
    observation_request: ObservationRequest


@dataclass(frozen=True)
class _SecondExpansion:
    root: ReasoningPath
    first_step: ImaginedReasoningStep
    projected: ReasoningPath
    action: NavigationAction
    observation_request: ObservationRequest


def _first_expansions(
    forest: ReasoningPathForest,
    memory: EvidenceMemory,
    navigation: NavigationGraph,
) -> tuple[_FirstExpansion, ...]:
    rows: list[_FirstExpansion] = []
    for path in forest.plannable:
        actions = compile_legal_actions(
            navigation,
            cursor_id=path.cursor_id,
            acquired_ids=forest.shared_evidence.acquired_ids,
        )
        for action in actions:
            if not action.reads_evidence:
                continue
            expansion_id = _digest(path.path_id, action.action_id)
            rows.append(
                _FirstExpansion(
                    expansion_id,
                    path,
                    action,
                    _observation_request(forest, memory, action, suffix=expansion_id),
                )
            )
    return tuple(rows)


def _observation_request(
    forest: ReasoningPathForest,
    memory: EvidenceMemory,
    action: NavigationAction,
    *,
    suffix: str,
    imagined_prefix: tuple[tuple[str, ObservationDescriptor], ...] = (),
) -> ObservationRequest:
    assert action.target_id is not None
    record = memory.read(action.target_id)
    acquired = tuple(
        (node_id, memory.read(node_id).value)
        for node_id in forest.shared_evidence.acquired_ids
    )
    return ObservationRequest(
        request_id=f"observation:{suffix}",
        belief=_shared_belief(forest),
        action=action,
        target_address=record.address,
        context=ObservationContext(acquired, imagined_prefix),
    )


def _shared_belief(forest: ReasoningPathForest) -> BeliefState:
    paths = forest.plannable
    first = paths[0].belief
    required = tuple(
        dict.fromkeys(role for row in paths for role in row.belief.required_roles)
    )
    missing = tuple(
        dict.fromkeys(role for row in paths for role in row.belief.missing_roles)
    )
    contradictions = tuple(
        dict.fromkeys(value for row in paths for value in row.belief.contradictions)
    )
    answerability = (
        Answerability.READY
        if paths
        and all(row.belief.answerability is Answerability.READY for row in paths)
        else Answerability.NOT_READY
    )
    return BeliefState(
        question=first.question,
        required_roles=required,
        missing_roles=missing,
        contradictions=contradictions,
        answerability=answerability,
    )


def _predict_observations(
    model: ObservationWorldModel,
    requests: tuple[ObservationRequest, ...],
) -> tuple[tuple[ObservationPrediction, ...], int]:
    # Deduplicate the physical observation request across hypotheses and paths.
    representative: dict[tuple[object, ...], ObservationRequest] = {}
    aliases: dict[str, str] = {}
    for request in requests:
        key = observation_world_key(request)
        canonical = representative.setdefault(key, request)
        aliases[request.request_id] = canonical.request_id
    unique = tuple(representative.values())
    predicted = validate_observation_predictions(unique, model.predict_batch(unique))
    by_id = {row.request_id: row for row in predicted}
    return (
        tuple(
            replace(
                by_id[aliases[row.request_id]],
                request_id=row.request_id,
                action_id=row.action.action_id,
                target_id=str(row.action.target_id),
            )
            for row in requests
        ),
        len(unique),
    )


def _validate_effects(
    requests: tuple[HypothesisEffectRequest, ...],
    effects: tuple[BeliefEffect, ...],
) -> None:
    if len(requests) != len(effects):
        raise ValueError("hypothesis-effect prediction coverage mismatch")
    if {row.request_id for row in requests} != {row.request_id for row in effects}:
        raise ValueError("hypothesis-effect prediction IDs mismatch")


def _candidate(root, steps):
    identity = "\x1f".join((root.path_id, *(step.action.action_id for step in steps)))
    return CandidateReasoningTrajectory(
        candidate_id=f"candidate-trajectory:{hashlib.sha256(identity.encode()).hexdigest()[:20]}",
        root_path_id=root.path_id,
        hypothesis=root.hypothesis,
        steps=steps,
    )


def _joint_action_trees(
    candidates: tuple[CandidateReasoningTrajectory, ...],
) -> tuple[JointActionTree, ...]:
    grouped: dict[tuple[str, ...], list[CandidateReasoningTrajectory]] = defaultdict(
        list
    )
    for candidate in candidates:
        grouped[candidate.first_action.shared_key].append(candidate)
    trees = []
    for shared_key, trajectories in grouped.items():
        ordered = tuple(sorted(trajectories, key=lambda row: row.candidate_id))
        identity = "\x1f".join((*shared_key, *(row.candidate_id for row in ordered)))
        trees.append(
            JointActionTree(
                tree_id=(
                    "joint-action-tree:"
                    f"{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
                ),
                first_action=ordered[0].first_action,
                trajectories=ordered,
            )
        )
    return tuple(sorted(trees, key=lambda row: row.tree_id))


def _validate_joint_coverage(
    forest: ReasoningPathForest,
    trees: tuple[JointActionTree, ...],
) -> None:
    required = {path.hypothesis for path in forest.plannable}
    incomplete = [
        tree.tree_id for tree in trees if set(tree.covered_hypotheses) != required
    ]
    if incomplete:
        raise ValueError(
            f"joint action tree lacks hypothesis-conditioned outcomes: {incomplete}"
        )


def _digest(*values):
    return hashlib.sha256("\x1f".join(values).encode()).hexdigest()[:20]


def _exact_path_deduplicate(paths):
    seen = set()
    result = []
    for path in paths:
        signature = (
            path.hypothesis,
            path.belief,
            path.cursor_id,
            path.action_history,
            path.interpreted_observation_ids,
            path.status,
            path.pending_first_action_id,
        )
        if signature not in seen:
            seen.add(signature)
            result.append(path)
    return tuple(result)


def _empty_decision(horizon):
    from .contracts import PreferenceStatus, TrajectoryPreferenceDecision

    return PlanningDecision(
        candidates=(),
        preference=TrajectoryPreferenceDecision(
            PreferenceStatus.ABSTAIN, (), None, "no executable evidence action"
        ),
        selected_action=None,
        observation_prediction_count=0,
        effect_prediction_count=0,
        horizon=horizon,
    )
