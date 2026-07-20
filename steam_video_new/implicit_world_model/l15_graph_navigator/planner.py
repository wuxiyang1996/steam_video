"""Model-predictive graph navigation with ordinal trajectory preferences."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from itertools import combinations
from typing import Mapping

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
    ContradictionChange,
    FrontierChange,
    GraphReadExecution,
    GraphReadExecutor,
    NavigationRun,
    NavigationStep,
    ObservationBeliefWorldModel,
    PairwisePreference,
    HypothesisDisposition,
    PlanDecision,
    PredictedTransition,
    PreferenceLabel,
    PathChange,
    RelationGrounding,
    TrajectoryPrediction,
    TrajectoryPreferenceModel,
    UncertaintyChange,
    UncertaintyLevel,
)
from .context import ReasoningContextBuilder
from .interventions import TransitionIntervention, intervene_trajectories
from .realized import derive_realized_belief_delta


class PreferenceOnlyPlanner:
    """Expand short graph trajectories and select via a partial order."""

    def __init__(
        self,
        world_model: ObservationBeliefWorldModel,
        preference_model: TrajectoryPreferenceModel,
        *,
        horizon: int = 2,
        max_second_actions: int = 4,
        context_builder: ReasoningContextBuilder | None = None,
        transition_intervention: TransitionIntervention = TransitionIntervention.NORMAL,
    ) -> None:
        if horizon not in {1, 2}:
            raise ValueError("only horizon 1 or 2 is supported")
        if max_second_actions <= 0:
            raise ValueError("max_second_actions must be positive")
        self.world_model = world_model
        self.preference_model = preference_model
        self.horizon = horizon
        self.max_second_actions = max_second_actions
        self.context_builder = context_builder or ReasoningContextBuilder()
        self.transition_intervention = transition_intervention

    def plan(
        self,
        belief: BeliefSnapshot,
        overlay: CausalTemporalOverlay,
        *,
        recent_hops: tuple[str, ...] = (),
        embedding_scores: Mapping[str, float] | None = None,
    ) -> PlanDecision:
        actions = guided_navigation_actions(belief, overlay)
        built = self.context_builder.build(
            belief,
            overlay,
            actions,
            recent_hops=recent_hops,
            embedding_scores=embedding_scores,
        )
        trajectories = self._expand_trajectories(
            built.belief,
            built.overlay,
            built.actions,
            max_trajectories=_max_fully_compared_trajectories(
                built.context.audit.comparison_budget
            ),
        )
        if not trajectories:
            raise RuntimeError("the graph action generator returned no trajectories")
        trajectories = list(
            intervene_trajectories(
                tuple(trajectories), built.belief, self.transition_intervention
            )
        )

        comparisons: list[PairwisePreference] = []
        dominated: set[str] = set()
        for left, candidate in combinations(trajectories, 2):
            comparison = self.preference_model.compare(
                left,
                candidate,
                built.belief,
            )
            comparisons.append(comparison)
            if comparison.label is PreferenceLabel.PREFER_LEFT:
                dominated.add(candidate.trajectory_id)
            elif comparison.label is PreferenceLabel.PREFER_RIGHT:
                dominated.add(left.trajectory_id)

        undominated = tuple(
            trajectory.trajectory_id
            for trajectory in trajectories
            if trajectory.trajectory_id not in dominated
        )
        unique = [
            trajectory
            for trajectory in trajectories
            if trajectory.trajectory_id in set(undominated)
        ]
        first_hops = {
            _action_digest(trajectory.first_action): trajectory.first_action
            for trajectory in unique
        }
        if len(first_hops) == 1:
            selected_action = next(iter(first_hops.values()))
            selected_trajectory_id = min(
                trajectory.trajectory_id for trajectory in unique
            )
            planning_status = (
                "selected"
                if len(unique) == 1
                else "selected_equivalent_first_hop"
            )
            ambiguity_reason = None
        else:
            selected_action = GraphReadAction(
                NavigationActionType.STOP,
                rationale="planning_abstain: non-unique undominated trajectories",
            )
            selected_trajectory_id = "planning:abstain"
            planning_status = "abstain"
            ambiguity_reason = (
                "no_undominated_trajectory"
                if not unique
                else "multiple_undominated_trajectories"
            )
        return PlanDecision(
            selected_action=selected_action,
            selected_trajectory_id=selected_trajectory_id,
            trajectories=tuple(trajectories),
            comparisons=tuple(comparisons),
            undominated_trajectory_ids=undominated,
            fallback_policy="explicit_abstain_for_non_unique_undominated",
            reasoning_context=built.context,
            planning_status=planning_status,
            ambiguity_reason=ambiguity_reason,
        )

    def _expand_trajectories(
        self,
        belief: BeliefSnapshot,
        overlay: CausalTemporalOverlay,
        actions: tuple[GraphReadAction, ...],
        *,
        max_trajectories: int,
    ) -> list[TrajectoryPrediction]:
        trajectories: list[TrajectoryPrediction] = []
        for first_action in actions:
            if len(trajectories) >= max_trajectories:
                break
            first = self.world_model.predict(belief, first_action, overlay)
            if (
                self.horizon == 1
                or first_action.action_type is NavigationActionType.STOP
            ):
                trajectories.append(
                    TrajectoryPrediction(_trajectory_id((first,)), (first,))
                )
                continue

            imagined_belief = _project_imagined_belief(belief, first)
            second_actions = guided_navigation_actions(
                imagined_belief,
                overlay,
            )[: self.max_second_actions]
            if not second_actions:
                trajectories.append(
                    TrajectoryPrediction(_trajectory_id((first,)), (first,))
                )
                continue
            for second_action in second_actions:
                if len(trajectories) >= max_trajectories:
                    break
                second = self.world_model.predict(
                    imagined_belief,
                    second_action,
                    overlay,
                )
                trajectories.append(
                    TrajectoryPrediction(
                        _trajectory_id((first, second)),
                        (first, second),
                    )
                )
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
        recent_hops: list[str] = []
        while belief.remaining_graph_reads > 0:
            if max_steps is not None and len(steps) >= max_steps:
                break
            decision = self.planner.plan(
                belief,
                overlay,
                recent_hops=tuple(recent_hops),
            )
            action = decision.selected_action
            recent_hops.append(decision.selected_hop.hop_type.value)
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
            execution_update = getattr(self.backend, "update_from_execution", None)
            update = (
                execution_update(belief, action, execution, overlay)
                if callable(execution_update)
                else self.backend.update(belief, action, observations, overlay)
            )
            steps.append(
                NavigationStep(
                    belief_before_id=belief.belief_id,
                    decision=decision,
                    observation_ids=tuple(node.node_id for node in observations),
                    belief_after_id=update.belief.belief_id,
                    realized_belief_delta=derive_realized_belief_delta(
                        belief, update.belief
                    ),
                    skill_invocation=execution.skill_invocation,
                    belief_update_audit=update.audit_record,
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
        action_key = "\x1f".join(
            (
                belief.belief_id,
                action.action_type.value,
                action.source_id or "",
                *action.target_ids,
                action.relation or "",
            )
        )
        node_id = f"persisted-read:{hashlib.sha256(action_key.encode('utf-8')).hexdigest()[:24]}"
        evidence_refs = tuple(
            dict.fromkeys(
                ref
                for node in observations
                for ref in node.source_segments
            )
        )
        return GraphReadExecution(
            observations=observations,
            skill_invocation={
                "node_id": node_id,
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
                "evidence_refs": list(evidence_refs),
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
    hypothesis_updates = {
        update.edge_id: update.disposition
        for update in transition.belief_delta.hypothesis_updates
    }
    relation_states = tuple(
        replace(
            state,
            grounding=(
                RelationGrounding.VERIFIED
                if hypothesis_updates[state.edge_id]
                is HypothesisDisposition.ACCEPTED
                else RelationGrounding.CONTRADICTED
                if hypothesis_updates[state.edge_id]
                is HypothesisDisposition.REJECTED
                else state.grounding
            ),
        )
        if state.edge_id in hypothesis_updates
        else state
        for state in belief.relation_states
    )
    contradictions = list(belief.contradictions)
    contradiction_updates = set(transition.belief_delta.contradiction_updates)
    if (
        transition.belief_delta.contradiction_change
        is ContradictionChange.RESOLVED
    ):
        contradictions = [
            value for value in contradictions if value not in contradiction_updates
        ]
    elif (
        transition.belief_delta.contradiction_change
        is ContradictionChange.OPENED
    ):
        contradictions = list(
            dict.fromkeys((*contradictions, *transition.belief_delta.contradiction_updates))
        )
    priority = set(belief.priority_edge_ids)
    blocked = set(belief.blocked_edge_ids)
    for edge_id, disposition in hypothesis_updates.items():
        if disposition is HypothesisDisposition.ACCEPTED:
            priority.discard(edge_id)
            blocked.discard(edge_id)
        elif disposition is HypothesisDisposition.REJECTED:
            priority.discard(edge_id)
            blocked.add(edge_id)
        else:
            priority.add(edge_id)
    changed_edges = set(transition.belief_delta.relation_updates)
    if transition.belief_delta.path_change is PathChange.BLOCKED:
        blocked.update(changed_edges)
    elif transition.belief_delta.path_change is PathChange.OPENED:
        blocked.difference_update(changed_edges)
    frontier = belief.frontier
    if transition.belief_delta.frontier_change is FrontierChange.OPENED:
        frontier = target_ids or belief.frontier
    elif transition.belief_delta.frontier_change is FrontierChange.CLOSED:
        frontier = ()
    return replace(
        belief,
        belief_id=f"{belief.belief_id}:imagined",
        acquired_evidence=acquired,
        frontier=frontier,
        missing_roles=missing,
        contradictions=tuple(contradictions),
        relation_states=relation_states,
        priority_edge_ids=tuple(sorted(priority)),
        blocked_edge_ids=tuple(sorted(blocked)),
        uncertainty=uncertainty,
        answerability=transition.belief_delta.answerability_after,
        step=belief.step + 1,
    )


def guided_navigation_actions(
    belief: BeliefSnapshot,
    overlay: CausalTemporalOverlay,
) -> list[GraphReadAction]:
    """Use factor-graph directives to order/filter legal exploration actions."""

    actions = propose_navigation_actions(belief.navigation_view(), overlay)
    blocked = set(belief.blocked_edge_ids)
    acquired = set(belief.acquired_evidence)
    edges = {
        edge.edge_id: edge
        for edge in overlay.relations + overlay.l1_structural_relations
    }
    for edge_id in tuple(dict.fromkeys(belief.priority_edge_ids + belief.blocked_edge_ids)):
        edge = edges.get(edge_id)
        if edge is None:
            continue
        targets = _verification_targets(edge, overlay, acquired)
        if not targets:
            continue
        relation = next(iter(sorted(edge.relation_probabilities)), None)
        actions.append(
            GraphReadAction(
                action_type=NavigationActionType.VERIFY,
                source_id=edge.src,
                target_ids=targets,
                relation=relation,
                rationale="factor belief requests a grounded provenance reread",
            )
        )
    kept: list[tuple[GraphReadAction, set[str]]] = []
    for action in actions:
        edge_ids = _action_edge_ids(action, overlay)
        if (
            edge_ids
            and edge_ids <= blocked
            and action.action_type
            not in {
                NavigationActionType.SEARCH_COUNTEREVIDENCE,
                NavigationActionType.VERIFY,
            }
        ):
            continue
        kept.append((action, edge_ids))

    ordered = sorted(
        (action for action, _ in kept),
        key=lambda action: (
            action.action_type is NavigationActionType.STOP,
            _action_digest(action),
        ),
    )
    result: list[GraphReadAction] = []
    seen: set[tuple[object, ...]] = set()
    for action in ordered:
        key = (
            action.action_type,
            action.source_id,
            action.target_ids,
            action.relation,
        )
        if key not in seen:
            seen.add(key)
            result.append(action)
    return result


def _action_edge_ids(
    action: GraphReadAction,
    overlay: CausalTemporalOverlay,
) -> set[str]:
    if action.source_id is None:
        return set()
    return {
        edge.edge_id
        for edge in overlay.relations + overlay.l1_structural_relations
        if action.source_id in {edge.src, edge.dst}
        and any(target in {edge.src, edge.dst} for target in action.target_ids)
        and (action.relation is None or action.relation in edge.relation_probabilities)
    }


def _action_digest(action: GraphReadAction) -> str:
    value = "\x1f".join(
        (
            action.action_type.value,
            action.source_id or "",
            *action.target_ids,
            action.relation or "",
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _trajectory_id(transitions: tuple[PredictedTransition, ...]) -> str:
    value = "\x1e".join(_action_digest(item.action) for item in transitions)
    return f"trajectory:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def _max_fully_compared_trajectories(comparison_budget: int) -> int:
    count = 1
    while (count + 1) * count // 2 <= comparison_budget:
        count += 1
    return count


def _verification_targets(
    edge: object,
    overlay: CausalTemporalOverlay,
    acquired: set[str],
) -> tuple[str, ...]:
    known = {
        node.node_id for node in overlay.atomic_events + overlay.l1_observations
    }
    by_id = {
        node.node_id: node for node in overlay.atomic_events + overlay.l1_observations
    }
    endpoint = str(getattr(edge, "dst", ""))
    candidates = [
        str(value)
        for value in getattr(edge, "evidence_refs", ())
        if str(value) in known
    ]
    for endpoint in (getattr(edge, "src", ""), getattr(edge, "dst", "")):
        node = by_id.get(endpoint)
        if node is not None:
            candidates.extend(ref for ref in node.source_segments if ref in known)
    evidence_targets = tuple(
        value for value in dict.fromkeys(candidates) if value not in acquired
    )
    # A verifier reread may include provenance nodes, but the typed endpoint
    # must remain explicit so the executed action is unambiguously attributable
    # to one relation measurement.
    return tuple(dict.fromkeys(((endpoint,) if endpoint in known else ()) + evidence_targets))
