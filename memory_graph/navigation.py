"""Belief-guided real graph reads over temporal and predictive dependencies."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

from .types import CausalTemporalOverlay, MemoryNode, RelationBelief


class NavigationActionType(str, Enum):
    SEMANTIC = "semantic"
    TEMPORAL_BACK = "temporal_back"
    TEMPORAL_FORWARD = "temporal_forward"
    TRACK_ENTITY = "track_entity"
    INSPECT_STATE_CHANGE = "inspect_state_change"
    FOLLOW_DEPENDENCY = "follow_dependency"
    CANDIDATE_CAUSE = "candidate_cause"
    EFFECT = "effect"
    FIND_BRIDGE = "find_bridge"
    SEARCH_COUNTEREVIDENCE = "search_counterevidence"
    VERIFY = "verify"
    STOP = "stop"


@dataclass(frozen=True)
class GraphReadAction:
    action_type: NavigationActionType
    source_id: str | None = None
    target_ids: tuple[str, ...] = ()
    relation: str | None = None
    rationale: str = ""


@dataclass(frozen=True)
class NavigationBeliefState:
    question: str
    acquired_evidence: tuple[str, ...] = ()
    frontier: tuple[str, ...] = ()
    missing_roles: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    residual_uncertainty: float = 1.0
    remaining_graph_reads: int = 8
    step: int = 0


@dataclass(frozen=True)
class PredictedReadTransition:
    """Legacy scalar baseline output.

    The preference-only implicit world model lives in
    ``steam_video_new.implicit_world_model.l15_graph_navigator`` and does not
    consume or emit this contract.
    """

    action: GraphReadAction
    expected_information_gain: float
    delayed_utility: float
    evidence_support: float
    cost: float
    score: float
    predicted_only: bool = True


class NavigationWorldModel(Protocol):
    def predict(
        self,
        belief: NavigationBeliefState,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> PredictedReadTransition: ...


@dataclass
class RuleBasedDependencyWorldModel:
    """Transparent baseline before fitting a learned transition model."""

    read_cost: float = 0.1

    def predict(
        self,
        belief: NavigationBeliefState,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> PredictedReadTransition:
        if action.action_type is NavigationActionType.STOP:
            information_gain = 0.0
            delayed = 0.0
            support = max(0.0, 1.0 - belief.residual_uncertainty)
            cost = 0.0
        else:
            unseen = [
                node_id
                for node_id in action.target_ids
                if node_id not in set(belief.acquired_evidence)
            ]
            information_gain = min(1.0, 0.35 * len(unseen))
            delayed = _delayed_bridge_value(action, overlay)
            support = _relation_support(action, overlay)
            cost = self.read_cost * max(1, len(action.target_ids))
        score = information_gain + delayed + support - cost
        return PredictedReadTransition(
            action=action,
            expected_information_gain=information_gain,
            delayed_utility=delayed,
            evidence_support=support,
            cost=cost,
            score=score,
        )


def propose_navigation_actions(
    belief: NavigationBeliefState,
    overlay: CausalTemporalOverlay,
    *,
    semantic_top_k: int = 3,
) -> list[GraphReadAction]:
    """Propose graph reads; relations guide retrieval but are not answer evidence."""

    event_by_id = {node.node_id: node for node in overlay.atomic_events}
    l1_by_id = {node.node_id: node for node in overlay.l1_observations}
    all_relations = overlay.relations + overlay.l1_structural_relations
    l1_to_events: dict[str, list[str]] = {}
    for event in overlay.atomic_events:
        for l1_id in event.source_segments:
            l1_to_events.setdefault(l1_id, []).append(event.node_id)
    acquired = set(belief.acquired_evidence)
    frontier = belief.frontier or belief.acquired_evidence[-1:]
    actions: list[GraphReadAction] = []

    semantic = sorted(
        (
            (_lexical_score(belief.question, node.text or ""), node)
            for node in overlay.atomic_events
            if node.node_id not in acquired
        ),
        key=lambda item: (-item[0], item[1].time_span.start_s, item[1].node_id),
    )
    semantic_targets = tuple(
        node.node_id for score, node in semantic[:semantic_top_k] if score > 0
    )
    if semantic_targets:
        actions.append(
            GraphReadAction(
                NavigationActionType.SEMANTIC,
                target_ids=semantic_targets,
                rationale="question-to-event lexical retrieval",
            )
        )

    for source_id in frontier:
        for edge in all_relations:
            if source_id not in {edge.src, edge.dst}:
                continue
            actions.extend(_edge_actions(source_id, edge, acquired))
        if source_id in event_by_id:
            grounding_targets = tuple(
                l1_id
                for l1_id in event_by_id[source_id].source_segments
                if l1_id in l1_by_id and l1_id not in acquired
            )
            if grounding_targets:
                actions.append(
                    GraphReadAction(
                        NavigationActionType.FOLLOW_DEPENDENCY,
                        source_id,
                        grounding_targets,
                        "grounded_by",
                        "enter the native L1 evidence graph",
                    )
                )
        if source_id in l1_by_id:
            projected_events = tuple(
                event_id
                for event_id in l1_to_events.get(source_id, [])
                if event_id not in acquired
            )
            if projected_events:
                actions.append(
                    GraphReadAction(
                        NavigationActionType.FOLLOW_DEPENDENCY,
                        source_id,
                        projected_events,
                        "event_projection",
                        "return from L1 evidence to its atomic-event projection",
                    )
                )

    bridge_targets = _bridge_targets(
        frontier=tuple(frontier),
        acquired=acquired,
        relations=all_relations,
    )
    if bridge_targets:
        actions.append(
            GraphReadAction(
                NavigationActionType.FIND_BRIDGE,
                source_id=frontier[0] if frontier else None,
                target_ids=bridge_targets,
                rationale="unseen intermediate connects the current frontier",
            )
        )
    actions.append(GraphReadAction(NavigationActionType.STOP, rationale="stop reading"))
    return _dedupe_actions(actions, known=set(event_by_id) | set(l1_by_id))


def plan_next_read(
    belief: NavigationBeliefState,
    overlay: CausalTemporalOverlay,
    world_model: NavigationWorldModel,
) -> tuple[GraphReadAction, list[PredictedReadTransition]]:
    """Rank imagined read outcomes without adding imagined evidence to belief."""

    actions = propose_navigation_actions(belief, overlay)
    predictions = [
        world_model.predict(belief, action, overlay) for action in actions
    ]
    predictions.sort(
        key=lambda value: (
            -value.score,
            value.action.action_type.value,
            value.action.target_ids,
        )
    )
    return predictions[0].action, predictions


def execute_real_graph_read(
    action: GraphReadAction,
    overlay: CausalTemporalOverlay,
) -> list[MemoryNode]:
    """Return only persisted nodes; predicted observations never enter answers."""

    by_id = {
        node.node_id: node
        for node in overlay.atomic_events + overlay.l1_observations
    }
    return [by_id[node_id] for node_id in action.target_ids if node_id in by_id]


def update_belief_after_read(
    belief: NavigationBeliefState,
    action: GraphReadAction,
    observations: list[MemoryNode],
) -> NavigationBeliefState:
    observed_ids = tuple(node.node_id for node in observations)
    acquired = tuple(dict.fromkeys(belief.acquired_evidence + observed_ids))
    uncertainty_drop = min(0.5, 0.12 * len(observed_ids))
    return replace(
        belief,
        acquired_evidence=acquired,
        frontier=observed_ids or belief.frontier,
        residual_uncertainty=max(0.0, belief.residual_uncertainty - uncertainty_drop),
        remaining_graph_reads=max(
            0,
            belief.remaining_graph_reads
            - (0 if action.action_type is NavigationActionType.STOP else 1),
        ),
        step=belief.step + 1,
    )


def _edge_actions(
    source_id: str,
    edge: RelationBelief,
    acquired: set[str],
) -> list[GraphReadAction]:
    other = edge.dst if edge.src == source_id else edge.src
    if other in acquired:
        return []
    names = set(edge.relation_probabilities)
    actions: list[GraphReadAction] = []
    if names & {"before", "temporal_next"}:
        action_type = (
            NavigationActionType.TEMPORAL_FORWARD
            if edge.src == source_id
            else NavigationActionType.TEMPORAL_BACK
        )
        relation_name = (
            "temporal_next" if "temporal_next" in names else "before"
        )
        actions.append(
            GraphReadAction(
                action_type,
                source_id,
                (other,),
                relation_name,
                "follow deterministic temporal structure",
            )
        )
    identity = names & {
        "same_entity",
        "same_object",
        "same_instance_candidate",
        "reappears_candidate",
    }
    for name in sorted(identity):
        actions.append(
            GraphReadAction(
                NavigationActionType.TRACK_ENTITY,
                source_id,
                (other,),
                name,
                "follow an identity or identity-candidate trajectory",
            )
        )
    if "state_transition" in names:
        actions.append(
            GraphReadAction(
                NavigationActionType.INSPECT_STATE_CHANGE,
                source_id,
                (other,),
                "state_transition",
                "inspect a visible state change",
            )
        )
    dependency = names & {
        "transition_support",
        "response_candidate",
        "observation_support",
    }
    for name in sorted(dependency):
        actions.append(
            GraphReadAction(
                NavigationActionType.FOLLOW_DEPENDENCY,
                source_id,
                (other,),
                name,
                "follow a predictive dependency without assuming causality",
            )
        )
    candidate_causal = names & {"explains", "enables"}
    for name in sorted(candidate_causal):
        if edge.dst == source_id:
            actions.append(
                GraphReadAction(
                    NavigationActionType.CANDIDATE_CAUSE,
                    source_id,
                    (edge.src,),
                    name,
                    "inspect an earlier candidate explanation without assuming causality",
                )
            )
        elif edge.src == source_id:
            actions.append(
                GraphReadAction(
                    NavigationActionType.EFFECT,
                    source_id,
                    (edge.dst,),
                    name,
                    "inspect a later candidate effect without assuming causality",
                )
            )
    if "contradicts" in names:
        actions.append(
            GraphReadAction(
                NavigationActionType.SEARCH_COUNTEREVIDENCE,
                source_id,
                (other,),
                "contradicts",
                "read counterevidence",
            )
        )
    return actions


def _bridge_targets(
    *,
    frontier: tuple[str, ...],
    acquired: set[str],
    relations: list[RelationBelief],
) -> tuple[str, ...]:
    neighbors: dict[str, set[str]] = {}
    for edge in relations:
        neighbors.setdefault(edge.src, set()).add(edge.dst)
        neighbors.setdefault(edge.dst, set()).add(edge.src)
    candidates: set[str] = set()
    for source in frontier:
        for middle in neighbors.get(source, set()):
            if middle in acquired:
                continue
            if any(
                target in acquired and target != source
                for target in neighbors.get(middle, set())
            ):
                candidates.add(middle)
    return tuple(sorted(candidates))


def _relation_support(
    action: GraphReadAction,
    overlay: CausalTemporalOverlay,
) -> float:
    if not action.relation:
        return 0.0
    values = [
        edge.relation_probabilities.get(action.relation, 0.0)
        for edge in overlay.relations + overlay.l1_structural_relations
        if action.source_id in {edge.src, edge.dst}
        and any(target in {edge.src, edge.dst} for target in action.target_ids)
    ]
    return max(values, default=0.0)


def _delayed_bridge_value(
    action: GraphReadAction,
    overlay: CausalTemporalOverlay,
) -> float:
    if action.action_type is NavigationActionType.FIND_BRIDGE:
        return min(1.0, 0.4 * len(action.target_ids))
    if action.action_type is NavigationActionType.FOLLOW_DEPENDENCY:
        return 0.25
    return 0.0


def _lexical_score(query: str, text: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _dedupe_actions(
    actions: list[GraphReadAction],
    *,
    known: set[str],
) -> list[GraphReadAction]:
    result: list[GraphReadAction] = []
    seen: set[tuple[str, str | None, tuple[str, ...], str | None]] = set()
    for action in actions:
        targets = tuple(target for target in action.target_ids if target in known)
        if action.action_type is not NavigationActionType.STOP and not targets:
            continue
        normalized = replace(action, target_ids=targets)
        key = (
            normalized.action_type.value,
            normalized.source_id,
            normalized.target_ids,
            normalized.relation,
        )
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
