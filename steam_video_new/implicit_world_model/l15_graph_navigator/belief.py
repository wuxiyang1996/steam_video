"""A lightweight belief backend that can later be replaced by a factor graph."""

from __future__ import annotations

import hashlib
import re

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay, MemoryNode, RelationBelief

from .contracts import (
    Answerability,
    BeliefDeltaDescriptor,
    BeliefSnapshot,
    BeliefUpdateResult,
    RelationGrounding,
    RelationState,
    UncertaintyChange,
    UncertaintyLevel,
)


_ACTION_ROLE = {
    NavigationActionType.SEMANTIC: "semantic",
    NavigationActionType.TEMPORAL_BACK: "temporal",
    NavigationActionType.TEMPORAL_FORWARD: "temporal",
    NavigationActionType.TRACK_ENTITY: "identity",
    NavigationActionType.INSPECT_STATE_CHANGE: "state_transition",
    NavigationActionType.FOLLOW_DEPENDENCY: "dependency",
    NavigationActionType.CANDIDATE_CAUSE: "dependency",
    NavigationActionType.EFFECT: "dependency",
    NavigationActionType.FIND_BRIDGE: "bridge",
    NavigationActionType.SEARCH_COUNTEREVIDENCE: "counterevidence",
    NavigationActionType.VERIFY: "verification",
}


class FactorizedBeliefBackend:
    """Update independent relation states after real graph observations.

    This is intentionally not a full factor graph.  It implements the stable
    backend contract and exposes relation grounding explicitly, so a later
    message-passing backend can preserve the planner/world-model API.
    """

    name = "factorized_belief/v0.1"

    def initialize(
        self,
        question: str,
        overlay: CausalTemporalOverlay,
        *,
        seed_evidence: tuple[str, ...] = (),
        missing_roles: tuple[str, ...] | None = None,
        graph_read_budget: int = 8,
    ) -> BeliefSnapshot:
        known = {
            node.node_id
            for node in overlay.atomic_events + overlay.l1_observations
        }
        unknown = set(seed_evidence) - known
        if unknown:
            raise ValueError(f"seed_evidence references unknown nodes: {sorted(unknown)}")
        if graph_read_budget < 0:
            raise ValueError("graph_read_budget must be non-negative")

        acquired = tuple(dict.fromkeys(seed_evidence))
        roles = (
            _infer_missing_roles(question)
            if missing_roles is None
            else tuple(dict.fromkeys(missing_roles))
        )
        relation_states = tuple(
            _relation_state(edge, set(acquired))
            for edge in overlay.relations + overlay.l1_structural_relations
        )
        answerability = (
            Answerability.READY
            if acquired and not roles
            else Answerability.NOT_READY
        )
        return BeliefSnapshot(
            belief_id=f"{overlay.overlay_id}:belief:0",
            backend_name=self.name,
            question=question,
            acquired_evidence=acquired,
            frontier=acquired[-1:],
            missing_roles=roles,
            relation_states=relation_states,
            uncertainty=_uncertainty_level(roles, acquired),
            answerability=answerability,
            remaining_graph_reads=graph_read_budget,
            step=0,
        )

    def update(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        observations: list[MemoryNode],
        overlay: CausalTemporalOverlay,
    ) -> BeliefUpdateResult:
        known = {
            node.node_id
            for node in overlay.atomic_events + overlay.l1_observations
        }
        observed_ids = tuple(
            dict.fromkeys(node.node_id for node in observations if node.node_id in known)
        )
        acquired = tuple(dict.fromkeys(belief.acquired_evidence + observed_ids))
        acquired_set = set(acquired)

        resolved_roles: tuple[str, ...] = ()
        action_role = _ACTION_ROLE.get(action.action_type)
        if (
            observed_ids
            and action_role in belief.missing_roles
            and _action_can_resolve(action, overlay)
        ):
            resolved_roles = (action_role,)
        missing_roles = tuple(
            role for role in belief.missing_roles if role not in set(resolved_roles)
        )

        relation_states: list[RelationState] = []
        relation_updates: list[str] = []
        contradictions = list(belief.contradictions)
        edges = {
            edge.edge_id: edge
            for edge in overlay.relations + overlay.l1_structural_relations
        }
        for prior_state in belief.relation_states:
            edge = edges[prior_state.edge_id]
            updated = _relation_state(edge, acquired_set)
            if updated.grounding != prior_state.grounding:
                relation_updates.append(updated.edge_id)
            if (
                updated.grounding is RelationGrounding.CONTRADICTED
                and updated.edge_id not in contradictions
            ):
                contradictions.append(updated.edge_id)
            relation_states.append(updated)

        answerability = (
            Answerability.READY
            if acquired and not missing_roles and not contradictions
            else Answerability.NOT_READY
        )
        previous_uncertainty = belief.uncertainty
        uncertainty = _uncertainty_level(missing_roles, acquired)
        uncertainty_change = _compare_uncertainty(previous_uncertainty, uncertainty)
        next_step = belief.step + 1
        next_belief = BeliefSnapshot(
            belief_id=_next_belief_id(overlay.overlay_id, next_step, action),
            backend_name=self.name,
            question=belief.question,
            acquired_evidence=acquired,
            frontier=observed_ids or belief.frontier,
            missing_roles=missing_roles,
            contradictions=tuple(contradictions),
            relation_states=tuple(relation_states),
            uncertainty=uncertainty,
            answerability=answerability,
            remaining_graph_reads=max(
                0,
                belief.remaining_graph_reads
                - (0 if action.action_type is NavigationActionType.STOP else 1),
            ),
            step=next_step,
            backend_ref=belief.backend_ref,
        )
        delta = BeliefDeltaDescriptor(
            resolved_roles=resolved_roles,
            relation_updates=tuple(relation_updates),
            contradiction_updates=tuple(
                edge_id
                for edge_id in contradictions
                if edge_id not in set(belief.contradictions)
            ),
            uncertainty_change=uncertainty_change,
            answerability_after=answerability,
            predicted_only=False,
        )
        return BeliefUpdateResult(next_belief, delta)


def _relation_state(
    edge: RelationBelief,
    acquired: set[str],
) -> RelationState:
    observed_count = int(edge.src in acquired) + int(edge.dst in acquired)
    names = set(edge.relation_probabilities)
    if observed_count == 0:
        grounding = RelationGrounding.UNSEEN
    elif observed_count == 1:
        grounding = RelationGrounding.PARTIAL
    elif "contradicts" in names:
        grounding = RelationGrounding.CONTRADICTED
    elif _verified_relations(edge):
        grounding = RelationGrounding.VERIFIED
    else:
        grounding = RelationGrounding.ENDPOINTS_OBSERVED
    return RelationState(
        edge_id=edge.edge_id,
        src=edge.src,
        dst=edge.dst,
        relation_probabilities=tuple(sorted(edge.relation_probabilities.items())),
        posterior_probabilities=tuple(sorted(edge.relation_probabilities.items())),
        correlation_features=tuple(sorted(edge.features.items())),
        verified_relations=_verified_relations(edge),
        calibration_status=edge.status.value,
        grounding=grounding,
    )


def _infer_missing_roles(question: str) -> tuple[str, ...]:
    normalized = question.lower()
    roles: list[str] = []
    patterns = (
        ("temporal", r"\b(before|after|earlier|later|when|first|then)\b|之前|之后|何时"),
        ("identity", r"\b(who|whose|same person|same object|which person)\b|谁|同一"),
        ("state_transition", r"\b(change|changed|became|opened|closed|empty|full)\b|状态|变成"),
        ("dependency", r"\b(why|because|cause|caused|enable|result)\b|为什么|导致|原因"),
        ("bridge", r"\b(between|intermediate|connect)\b|中间|连接"),
    )
    for role, pattern in patterns:
        if re.search(pattern, normalized):
            roles.append(role)
    return tuple(roles or ["semantic"])


def _uncertainty_level(
    missing_roles: tuple[str, ...],
    acquired: tuple[str, ...],
) -> UncertaintyLevel:
    if not acquired or len(missing_roles) > 1:
        return UncertaintyLevel.HIGH
    if missing_roles:
        return UncertaintyLevel.MEDIUM
    return UncertaintyLevel.LOW


def _compare_uncertainty(
    before: UncertaintyLevel,
    after: UncertaintyLevel,
) -> UncertaintyChange:
    order = {
        UncertaintyLevel.LOW: 0,
        UncertaintyLevel.MEDIUM: 1,
        UncertaintyLevel.HIGH: 2,
    }
    if order[after] < order[before]:
        return UncertaintyChange.DECREASE
    if order[after] > order[before]:
        return UncertaintyChange.INCREASE
    return UncertaintyChange.UNCHANGED


def _action_can_resolve(
    action: GraphReadAction,
    overlay: CausalTemporalOverlay,
) -> bool:
    if action.action_type in {
        NavigationActionType.SEMANTIC,
        NavigationActionType.FIND_BRIDGE,
        NavigationActionType.SEARCH_COUNTEREVIDENCE,
    }:
        return True
    for edge in overlay.relations + overlay.l1_structural_relations:
        if action.source_id not in {edge.src, edge.dst}:
            continue
        if not any(target in {edge.src, edge.dst} for target in action.target_ids):
            continue
        if action.relation in _verified_relations(edge):
            return True
        if (
            action.action_type
            in {NavigationActionType.TEMPORAL_BACK, NavigationActionType.TEMPORAL_FORWARD}
            and edge.status.value == "deterministic"
        ):
            return True
    return False


def _verified_relations(edge: RelationBelief) -> tuple[str, ...]:
    verified: set[str] = set()
    provenance = edge.provenance or {}
    if provenance.get("accepted_identity_track"):
        verified.update(
            name
            for name in edge.relation_probabilities
            if name in {"same_entity", "same_object", "state_transition"}
        )
    hard_verifier = provenance.get("hard_verifier")
    if isinstance(hard_verifier, dict):
        verified.update(
            str(name)
            for name, result in hard_verifier.items()
            if isinstance(result, dict)
            and result.get("passed") is True
            and not any(
                "verification disabled" in str(reason).casefold()
                for reason in result.get("reasons") or []
            )
            and name in edge.relation_probabilities
        )
    return tuple(sorted(verified))


def _next_belief_id(
    overlay_id: str,
    step: int,
    action: GraphReadAction,
) -> str:
    raw = "|".join(
        (
            action.action_type.value,
            action.source_id or "",
            ",".join(action.target_ids),
            action.relation or "",
        )
    )
    branch = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{overlay_id}:belief:{step}:{branch}"
