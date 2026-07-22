"""No-Top-K horizon-1/2 planning driven only by categorical IWM outputs."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from itertools import combinations
from typing import Protocol, Sequence

from .action_compiler import GraphActionCompiler
from .contracts import (
    ActionKind,
    BatchedCategoricalWorldModel,
    BatchedTrajectoryPreferenceModel,
    ContradictionChange,
    CursorBeliefState,
    FullGraphPlanDecision,
    IWMRequest,
    ImaginedTransition,
    LegalGraphAction,
    PreferenceLabel,
    RetainedEvidenceGraph,
    TrajectoryPair,
    TrajectoryPrediction,
)


class SetwisePreferenceModel(Protocol):
    def select(
        self,
        trajectories: Sequence[TrajectoryPrediction],
        belief: CursorBeliefState,
    ) -> tuple[str, ...]: ...
from .model_input import build_iwm_graph_input


class FullGraphIWMPlanner:
    """Compare complete short trajectories without heuristic action ranking."""

    def __init__(
        self,
        world_model: BatchedCategoricalWorldModel,
        preference_model: BatchedTrajectoryPreferenceModel,
        *,
        horizon: int = 2,
        action_compiler: GraphActionCompiler | None = None,
        max_trajectory_pairs: int | None = None,
        setwise_preference_model: SetwisePreferenceModel | None = None,
        execute_stable_ties: bool = False,
    ) -> None:
        if horizon not in {1, 2}:
            raise ValueError("full-graph planner supports horizon one or two")
        self.world_model = world_model
        self.preference_model = preference_model
        self.horizon = horizon
        self.action_compiler = action_compiler or GraphActionCompiler()
        if max_trajectory_pairs is not None and max_trajectory_pairs < 1:
            raise ValueError("max_trajectory_pairs must be positive when supplied")
        self.max_trajectory_pairs = max_trajectory_pairs
        self.setwise_preference_model = setwise_preference_model
        self.execute_stable_ties = execute_stable_ties

    def plan(
        self,
        belief: CursorBeliefState,
        graph: RetainedEvidenceGraph,
    ) -> FullGraphPlanDecision:
        first_actions = self.action_compiler.compile(belief, graph)
        graph_input = build_iwm_graph_input(belief, graph, first_actions)
        first_requests = tuple(
            IWMRequest(belief=belief, graph_input=graph_input, action=action)
            for action in first_actions
        )
        first_predictions = self._predict_checked(first_requests)

        trajectories: list[TrajectoryPrediction] = []
        second_requests: list[IWMRequest] = []
        second_parents: list[ImaginedTransition] = []
        for first in first_predictions:
            if self.horizon == 1 or _is_terminal(first.action):
                trajectories.append(_trajectory((first,)))
                continue
            imagined_belief = _project_imagined_belief(belief, first)
            second_actions = self.action_compiler.compile(imagined_belief, graph)
            second_input = build_iwm_graph_input(
                imagined_belief,
                graph,
                second_actions,
            )
            for second_action in second_actions:
                second_requests.append(
                    IWMRequest(
                        belief=imagined_belief,
                        graph_input=second_input,
                        action=second_action,
                        parent_action_ids=(first.action.action_id,),
                        imagined_history=(first,),
                    )
                )
                second_parents.append(first)

        if second_requests:
            second_predictions = self._predict_checked(tuple(second_requests))
            trajectories.extend(
                _trajectory((first, second))
                for first, second in zip(second_parents, second_predictions)
            )
        if not trajectories:
            raise RuntimeError("full-graph action compiler produced no trajectory")

        if self.setwise_preference_model is not None:
            preferred = self.setwise_preference_model.select(trajectories, belief)
            preferred_set = set(preferred)
            known = {trajectory.trajectory_id for trajectory in trajectories}
            if not preferred_set <= known or len(preferred) != len(preferred_set):
                raise ValueError("setwise preference returned invalid trajectory IDs")
            first_hops = {
                trajectory.first_action.action_id: trajectory.first_action
                for trajectory in trajectories
                if trajectory.trajectory_id in preferred_set
            }
            if len(first_hops) == 1:
                selected = next(iter(first_hops.values()))
                status = "setwise_selected_unique_preferred_first_hop"
            elif first_hops and self.execute_stable_ties:
                ordered_preferred = [
                    action for action in first_actions if action.action_id in first_hops
                ]
                selected = next(
                    (action for action in ordered_preferred if action.reads_evidence),
                    ordered_preferred[0],
                )
                status = "setwise_selected_stable_tie_break"
            else:
                selected = next(
                    action for action in first_actions if action.kind is ActionKind.ABSTAIN
                )
                status = (
                    "setwise_abstain_incomparable"
                    if not first_hops
                    else "setwise_abstain_tied_first_hops"
                )
            return FullGraphPlanDecision(
                selected_action=selected,
                planning_status=status,
                trajectories=tuple(trajectories),
                preferences=(),
                undominated_trajectory_ids=preferred,
                legal_action_count=len(first_actions),
                top_k_applied=False,
            )

        pair_count = len(trajectories) * (len(trajectories) - 1) // 2
        if (
            self.max_trajectory_pairs is not None
            and pair_count > self.max_trajectory_pairs
        ):
            selected = next(
                action for action in first_actions if action.kind is ActionKind.ABSTAIN
            )
            return FullGraphPlanDecision(
                selected_action=selected,
                planning_status="abstain_exhaustive_comparison_budget_exceeded",
                trajectories=tuple(trajectories),
                preferences=(),
                undominated_trajectory_ids=tuple(
                    trajectory.trajectory_id for trajectory in trajectories
                ),
                legal_action_count=len(first_actions),
                top_k_applied=False,
            )
        pairs = tuple(
            TrajectoryPair(left, right) for left, right in combinations(trajectories, 2)
        )
        preferences = tuple(self.preference_model.compare_batch(pairs, belief))
        if len(preferences) != len(pairs):
            raise ValueError(
                "preference model must return one result per trajectory pair"
            )
        dominated: set[str] = set()
        for pair, preference in zip(pairs, preferences):
            if (
                preference.left_id != pair.left.trajectory_id
                or preference.right_id != pair.right.trajectory_id
            ):
                raise ValueError("preference result does not match its trajectory pair")
            if preference.label is PreferenceLabel.PREFER_LEFT:
                dominated.add(pair.right.trajectory_id)
            elif preference.label is PreferenceLabel.PREFER_RIGHT:
                dominated.add(pair.left.trajectory_id)

        undominated = tuple(
            trajectory.trajectory_id
            for trajectory in trajectories
            if trajectory.trajectory_id not in dominated
        )
        undominated_set = set(undominated)
        first_hops = {
            trajectory.first_action.action_id: trajectory.first_action
            for trajectory in trajectories
            if trajectory.trajectory_id in undominated_set
        }
        if len(first_hops) == 1:
            selected = next(iter(first_hops.values()))
            status = "selected_unique_undominated_first_hop"
        else:
            selected = next(
                action for action in first_actions if action.kind is ActionKind.ABSTAIN
            )
            status = (
                "abstain_preference_cycle"
                if not first_hops
                else "abstain_non_unique_partial_order"
            )
        return FullGraphPlanDecision(
            selected_action=selected,
            planning_status=status,
            trajectories=tuple(trajectories),
            preferences=preferences,
            undominated_trajectory_ids=undominated,
            legal_action_count=len(first_actions),
            top_k_applied=False,
        )

    def _predict_checked(
        self,
        requests: tuple[IWMRequest, ...],
    ) -> tuple[ImaginedTransition, ...]:
        predictions = tuple(self.world_model.predict_batch(requests))
        if len(predictions) != len(requests):
            raise ValueError("world model must return one transition per request")
        for request, prediction in zip(requests, predictions):
            if prediction.action != request.action:
                raise ValueError("world-model transition does not match its action")
            if not prediction.observation.predicted_only:
                raise ValueError("world-model observation must remain predicted-only")
            if not prediction.belief_delta.predicted_only:
                raise ValueError("world-model belief delta must remain predicted-only")
        return predictions


def _project_imagined_belief(
    belief: CursorBeliefState,
    transition: ImaginedTransition,
) -> CursorBeliefState:
    action = transition.action
    acquired = belief.acquired_evidence
    imagined = belief.imagined_evidence
    current = belief.current_node_id
    history = belief.cursor_history
    remaining = belief.remaining_reads
    if action.target_id is not None and not _is_terminal(action):
        current = action.target_id
        if belief.current_node_id is not None:
            history = (*history, belief.current_node_id)
        if action.reads_evidence:
            acquired = tuple(dict.fromkeys((*acquired, action.target_id)))
            imagined = tuple(dict.fromkeys((*imagined, action.target_id)))
            remaining = max(0, remaining - 1)
    missing = [
        role
        for role in belief.missing_roles
        if role not in set(transition.belief_delta.resolved_roles)
    ]
    missing.extend(
        role for role in transition.belief_delta.opened_roles if role not in missing
    )
    contradictions = belief.contradictions
    if transition.belief_delta.contradiction_change is ContradictionChange.RESOLVED:
        contradictions = ()
    elif transition.belief_delta.contradiction_change is ContradictionChange.OPENED:
        contradictions = tuple(
            dict.fromkeys((*contradictions, "imagined_contradiction"))
        )
    return replace(
        belief,
        belief_id=f"{belief.belief_id}:imagined:{action.action_id}",
        current_node_id=current,
        acquired_evidence=acquired,
        imagined_evidence=imagined,
        cursor_history=history,
        missing_roles=tuple(missing),
        contradictions=contradictions,
        answerability=transition.belief_delta.answerability_after,
        remaining_reads=remaining,
        step=belief.step + 1,
    )


def _trajectory(
    transitions: tuple[ImaginedTransition, ...],
) -> TrajectoryPrediction:
    payload = "\x1e".join(value.action.action_id for value in transitions)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return TrajectoryPrediction(f"trajectory:{digest}", transitions)


def _is_terminal(action: LegalGraphAction) -> bool:
    return action.kind in {ActionKind.STOP, ActionKind.ANSWER, ActionKind.ABSTAIN}
