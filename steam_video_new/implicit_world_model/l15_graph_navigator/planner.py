"""Model-predictive graph navigation with ordinal trajectory preferences."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

from memory_graph.navigation import (
    GraphReadAction,
    NavigationActionType,
    execute_real_graph_read,
    propose_navigation_actions,
)
from memory_graph.types import CausalTemporalOverlay

from .contracts import (
    BeliefBackend,
    BeliefSnapshot,
    GraphReadExecution,
    GraphReadExecutor,
    NavigationRun,
    NavigationStep,
    ObservationBeliefWorldModel,
    PairwisePreference,
    PlanDecision,
    PredictedTransition,
    PreferenceLabel,
    TrajectoryPrediction,
    TrajectoryPreferenceModel,
    UncertaintyChange,
    UncertaintyLevel,
)


class PreferenceOnlyPlanner:
    """Expand short graph trajectories and select via a partial order."""

    def __init__(
        self,
        world_model: ObservationBeliefWorldModel,
        preference_model: TrajectoryPreferenceModel,
        *,
        horizon: int = 2,
        max_second_actions: int = 4,
    ) -> None:
        if horizon not in {1, 2}:
            raise ValueError("only horizon 1 or 2 is supported")
        if max_second_actions <= 0:
            raise ValueError("max_second_actions must be positive")
        self.world_model = world_model
        self.preference_model = preference_model
        self.horizon = horizon
        self.max_second_actions = max_second_actions

    def plan(
        self,
        belief: BeliefSnapshot,
        overlay: CausalTemporalOverlay,
    ) -> PlanDecision:
        trajectories = self._expand_trajectories(belief, overlay)
        if not trajectories:
            raise RuntimeError("the graph action generator returned no trajectories")

        comparisons: list[PairwisePreference] = []
        dominated: set[str] = set()
        for left, right in combinations(trajectories, 2):
            comparison = self.preference_model.compare(left, right, belief)
            comparisons.append(comparison)
            if comparison.label is PreferenceLabel.PREFER_LEFT:
                dominated.add(right.trajectory_id)
            elif comparison.label is PreferenceLabel.PREFER_RIGHT:
                dominated.add(left.trajectory_id)

        undominated = tuple(
            trajectory.trajectory_id
            for trajectory in trajectories
            if trajectory.trajectory_id not in dominated
        )
        selected = next(
            (
                trajectory
                for trajectory in trajectories
                if trajectory.trajectory_id in set(undominated)
            ),
            trajectories[0],
        )
        return PlanDecision(
            selected_action=selected.first_action,
            selected_trajectory_id=selected.trajectory_id,
            trajectories=tuple(trajectories),
            comparisons=tuple(comparisons),
            undominated_trajectory_ids=undominated,
        )

    def _expand_trajectories(
        self,
        belief: BeliefSnapshot,
        overlay: CausalTemporalOverlay,
    ) -> list[TrajectoryPrediction]:
        actions = _ordered_actions(
            propose_navigation_actions(belief.navigation_view(), overlay)
        )
        trajectories: list[TrajectoryPrediction] = []
        sequence = 0
        for first_action in actions:
            first = self.world_model.predict(belief, first_action, overlay)
            if (
                self.horizon == 1
                or first_action.action_type is NavigationActionType.STOP
            ):
                trajectories.append(
                    TrajectoryPrediction(f"trajectory:{sequence}", (first,))
                )
                sequence += 1
                continue

            imagined_belief = _project_imagined_belief(belief, first)
            second_actions = _ordered_actions(
                propose_navigation_actions(imagined_belief.navigation_view(), overlay)
            )[: self.max_second_actions]
            if not second_actions:
                trajectories.append(
                    TrajectoryPrediction(f"trajectory:{sequence}", (first,))
                )
                sequence += 1
                continue
            for second_action in second_actions:
                second = self.world_model.predict(
                    imagined_belief,
                    second_action,
                    overlay,
                )
                trajectories.append(
                    TrajectoryPrediction(
                        f"trajectory:{sequence}",
                        (first, second),
                    )
                )
                sequence += 1
        return trajectories


class ClosedLoopNavigator:
    """Execute one real read, update through the backend, and replan."""

    def __init__(
        self,
        backend: BeliefBackend,
        planner: PreferenceOnlyPlanner,
        executor: GraphReadExecutor | None = None,
    ) -> None:
        self.backend = backend
        self.planner = planner
        self.executor = executor or PersistedGraphReadExecutor()

    def run(
        self,
        belief: BeliefSnapshot,
        overlay: CausalTemporalOverlay,
        *,
        max_steps: int | None = None,
    ) -> NavigationRun:
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        initial_belief_id = belief.belief_id
        steps: list[NavigationStep] = []
        snapshots = [belief]
        while belief.remaining_graph_reads > 0:
            if max_steps is not None and len(steps) >= max_steps:
                break
            decision = self.planner.plan(belief, overlay)
            action = decision.selected_action
            if action.action_type is NavigationActionType.STOP:
                steps.append(
                    NavigationStep(
                        belief_before_id=belief.belief_id,
                        decision=decision,
                        observation_ids=(),
                        belief_after_id=belief.belief_id,
                        realized_belief_delta=None,
                        skill_invocation=None,
                    )
                )
                break
            execution = self.executor.execute(belief, action, overlay)
            observations = list(execution.observations)
            update = self.backend.update(belief, action, observations, overlay)
            steps.append(
                NavigationStep(
                    belief_before_id=belief.belief_id,
                    decision=decision,
                    observation_ids=tuple(node.node_id for node in observations),
                    belief_after_id=update.belief.belief_id,
                    realized_belief_delta=update.delta,
                    skill_invocation=execution.skill_invocation,
                )
            )
            belief = update.belief
            snapshots.append(belief)
        return NavigationRun(
            initial_belief_id,
            belief,
            tuple(steps),
            tuple(snapshots),
        )


class PersistedGraphReadExecutor:
    """Default executor used when the Video_Skills runtime is unavailable."""

    def execute(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> GraphReadExecution:
        observations = tuple(execute_real_graph_read(action, overlay))
        return GraphReadExecution(
            observations=observations,
            skill_invocation={
                "skill_id": "persisted_graph_read",
                "args": {
                    "action_type": action.action_type.value,
                    "source_id": action.source_id,
                    "target_ids": list(action.target_ids),
                    "relation": action.relation,
                },
                "outputs": {
                    "real_observation_ids": [node.node_id for node in observations],
                },
                "status": "executed" if observations else "insufficient",
            },
        )


def _project_imagined_belief(
    belief: BeliefSnapshot,
    transition: PredictedTransition,
) -> BeliefSnapshot:
    """Create a planning-only shadow state; never persist it as evidence."""

    target_ids = transition.observation.target_ids
    acquired = tuple(dict.fromkeys(belief.acquired_evidence + target_ids))
    resolved = set(transition.belief_delta.resolved_roles)
    missing = tuple(role for role in belief.missing_roles if role not in resolved)
    uncertainty = belief.uncertainty
    if transition.belief_delta.uncertainty_change is UncertaintyChange.DECREASE:
        uncertainty = {
            UncertaintyLevel.HIGH: UncertaintyLevel.MEDIUM,
            UncertaintyLevel.MEDIUM: UncertaintyLevel.LOW,
            UncertaintyLevel.LOW: UncertaintyLevel.LOW,
        }[uncertainty]
    return replace(
        belief,
        belief_id=f"{belief.belief_id}:imagined",
        acquired_evidence=acquired,
        frontier=target_ids or belief.frontier,
        missing_roles=missing,
        uncertainty=uncertainty,
        answerability=transition.belief_delta.answerability_after,
        step=belief.step + 1,
    )


def _ordered_actions(actions: list[GraphReadAction]) -> list[GraphReadAction]:
    """Stable structural fallback, deliberately not a learned utility score."""

    return sorted(
        actions,
        key=lambda action: (
            action.action_type is NavigationActionType.STOP,
            action.action_type.value,
            action.source_id or "",
            action.target_ids,
            action.relation or "",
        ),
    )
