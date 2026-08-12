"""JSON-safe serialization for preference-navigation artifacts."""

from __future__ import annotations

from typing import Any

from memory_graph.navigation import GraphReadAction

from .contracts import (
    BeliefDeltaDescriptor,
    BeliefSnapshot,
    NavigationRun,
    PlanDecision,
    PredictedTransition,
    RelationState,
    ReasoningContext,
    TrajectoryPrediction,
    reasoning_context_audit_to_dict,
)


def action_to_dict(action: GraphReadAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "source_id": action.source_id,
        "target_ids": list(action.target_ids),
        "relation": action.relation,
        "rationale": action.rationale,
    }


def relation_state_to_dict(state: RelationState) -> dict[str, Any]:
    return {
        "edge_id": state.edge_id,
        "src": state.src,
        "dst": state.dst,
        "relation_probabilities": dict(state.relation_probabilities),
        "posterior_probabilities": dict(state.posterior_probabilities),
        "correlation_features": dict(state.correlation_features),
        "verified_relations": list(state.verified_relations),
        "factor_sources": list(state.factor_sources),
        "calibration_status": state.calibration_status,
        "grounding": state.grounding.value,
    }


def belief_to_dict(belief: BeliefSnapshot) -> dict[str, Any]:
    return {
        "belief_id": belief.belief_id,
        "backend_name": belief.backend_name,
        "backend_ref": belief.backend_ref,
        "question": belief.question,
        "acquired_evidence": list(belief.acquired_evidence),
        "frontier": list(belief.frontier),
        "missing_roles": list(belief.missing_roles),
        "contradictions": list(belief.contradictions),
        "relation_states": [
            relation_state_to_dict(state) for state in belief.relation_states
        ],
        "priority_edge_ids": list(belief.priority_edge_ids),
        "blocked_edge_ids": list(belief.blocked_edge_ids),
        "uncertainty": belief.uncertainty.value,
        "answerability": belief.answerability.value,
        "remaining_graph_reads": belief.remaining_graph_reads,
        "step": belief.step,
    }


def delta_to_dict(delta: BeliefDeltaDescriptor) -> dict[str, Any]:
    return {
        "resolved_roles": list(delta.resolved_roles),
        "relation_updates": list(delta.relation_updates),
        "contradiction_updates": list(delta.contradiction_updates),
        "hypothesis_updates": [
            {
                "edge_id": update.edge_id,
                "disposition": update.disposition.value,
            }
            for update in delta.hypothesis_updates
        ],
        "frontier_change": delta.frontier_change.value,
        "contradiction_change": delta.contradiction_change.value,
        "path_change": delta.path_change.value,
        "recovery_status": delta.recovery_status.value,
        "uncertainty_change": delta.uncertainty_change.value,
        "answerability_after": delta.answerability_after.value,
        "predicted_only": delta.predicted_only,
    }


def transition_to_dict(transition: PredictedTransition) -> dict[str, Any]:
    return {
        "action": action_to_dict(transition.action),
        "observation_descriptor": {
            "role": transition.observation.role.value,
            "target_ids": list(transition.observation.target_ids),
            "node_kind": transition.observation.node_kind,
            "predicted_only": transition.observation.predicted_only,
        },
        "belief_delta": delta_to_dict(transition.belief_delta),
    }


def trajectory_to_dict(trajectory: TrajectoryPrediction) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory.trajectory_id,
        "transitions": [
            transition_to_dict(transition)
            for transition in trajectory.transitions
        ],
    }


def plan_to_dict(plan: PlanDecision) -> dict[str, Any]:
    return {
        "selected_action": action_to_dict(plan.selected_action),
        "selected_reasoning_hop": plan.selected_hop.hop_type.value,
        "selected_trajectory_id": plan.selected_trajectory_id,
        "planning_status": plan.planning_status,
        "ambiguity_reason": plan.ambiguity_reason,
        "trajectories": [
            trajectory_to_dict(trajectory) for trajectory in plan.trajectories
        ],
        "comparisons": [
            {
                "left_id": comparison.left_id,
                "right_id": comparison.right_id,
                "label": comparison.label.value,
                "rationale": comparison.rationale,
            }
            for comparison in plan.comparisons
        ],
        "undominated_trajectory_ids": list(plan.undominated_trajectory_ids),
        "fallback_policy": plan.fallback_policy,
        "reasoning_context": (
            reasoning_context_to_dict(plan.reasoning_context)
            if plan.reasoning_context is not None
            else None
        ),
    }


def reasoning_context_to_dict(context: ReasoningContext) -> dict[str, Any]:
    return {
        "question": context.question,
        "answerability": context.answerability.value,
        "missing_roles": list(context.missing_roles),
        "accepted_hypotheses": list(context.accepted_hypotheses),
        "rejected_hypotheses": list(context.rejected_hypotheses),
        "unresolved_hypotheses": list(context.unresolved_hypotheses),
        "contradictions": list(context.contradictions),
        "acquired_evidence_refs": list(context.acquired_evidence_refs),
        "recent_hops": list(context.recent_hops),
        "local_node_ids": list(context.local_node_ids),
        "local_edge_ids": list(context.local_edge_ids),
        "candidate_hops": [
            {
                "hop_type": hop.hop_type.value,
                "action": action_to_dict(hop.action),
            }
            for hop in context.candidate_hops
        ],
        "audit": reasoning_context_audit_to_dict(context.audit),
    }


def navigation_run_to_dict(run: NavigationRun) -> dict[str, Any]:
    return {
        "schema_version": "steam-preference-navigation/v0.2",
        "initial_belief_id": run.initial_belief_id,
        "final_belief": belief_to_dict(run.final_belief),
        "belief_snapshots": [
            belief_to_dict(snapshot) for snapshot in run.belief_snapshots
        ],
        "steps": [
            {
                "belief_before_id": step.belief_before_id,
                "plan": plan_to_dict(step.decision),
                "real_observation_ids": list(step.observation_ids),
                "belief_after_id": step.belief_after_id,
                "realized_belief_delta": (
                    delta_to_dict(step.realized_belief_delta)
                    if step.realized_belief_delta is not None
                    else None
                ),
                "skill_invocation": step.skill_invocation,
            }
            for step in run.steps
        ],
        "output_contract": "ordinal_preference_only",
    }
