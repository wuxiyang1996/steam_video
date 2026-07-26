"""Bounded local-subgraph context construction for multi-hop planning."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Mapping

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay

from .contracts import (
    BeliefSnapshot,
    ReasoningContext,
    ReasoningContextAudit,
    ReasoningContextBudget,
    RelationGrounding,
    reasoning_hop_from_action,
)


@dataclass(frozen=True)
class BuiltReasoningContext:
    context: ReasoningContext
    belief: BeliefSnapshot
    overlay: CausalTemporalOverlay
    actions: tuple[GraphReadAction, ...]


class ReasoningContextBuilder:
    """Retrieve and prune the model-visible graph without changing evidence.

    Optional ``embedding_scores`` must be computed outside the LLM (for
    example with Qwen/Qwen3-VL-Embedding-2B).  Only the ranking is used; raw
    vectors are never stored in ``ReasoningContext`` or sent to the model.
    """

    def __init__(self, budget: ReasoningContextBudget | None = None) -> None:
        self.budget = budget or ReasoningContextBudget()

    def build(
        self,
        belief: BeliefSnapshot,
        overlay: CausalTemporalOverlay,
        actions: list[GraphReadAction],
        *,
        recent_hops: tuple[str, ...] = (),
        embedding_scores: Mapping[str, float] | None = None,
    ) -> BuiltReasoningContext:
        ranked = self._rank_actions(actions, embedding_scores)
        retained, dropped_action_count = self._retain_actions(ranked)
        node_ids, _ = self._bounded_nodes(retained, overlay)
        all_node_ids = {
            node.node_id for node in overlay.atomic_events + overlay.l1_observations
        }
        dropped_node_ids = tuple(sorted(all_node_ids - node_ids))
        retained = tuple(self._trim_action(action, node_ids) for action in retained)
        retained = tuple(
            action
            for action in retained
            if action.action_type is NavigationActionType.STOP or action.target_ids
        )
        if not retained:
            retained = (
                GraphReadAction(
                    NavigationActionType.STOP,
                    rationale="context budget exhausted",
                ),
            )

        local_overlay, edge_ids, dropped_edge_ids = self._local_overlay(
            overlay,
            node_ids,
        )
        local_belief = self._local_belief(belief, node_ids, edge_ids)
        hypotheses = self._categorical_hypotheses(belief, set(edge_ids))
        embedding_models = tuple(
            sorted(
                {
                    node.embedding_ref.model
                    for node in overlay.atomic_events + overlay.l1_observations
                    if node.node_id in node_ids and node.embedding_ref is not None
                }
            )
        )
        recent = (
            recent_hops[-self.budget.recent_hop_window :]
            if self.budget.recent_hop_window
            else ()
        )
        prompt_tokens = _estimate_prompt_tokens(
            belief,
            local_overlay,
            retained,
            hypotheses,
            recent,
        )
        audit = ReasoningContextAudit(
            retrieval_mode=(
                "qwen_embedding_plus_structural"
                if embedding_scores
                else "structural_lexical"
            ),
            retrieved_node_ids=tuple(sorted(node_ids)),
            dropped_node_ids=dropped_node_ids,
            retrieved_edge_ids=edge_ids,
            dropped_edge_ids=dropped_edge_ids,
            retained_candidate_hops=len(retained),
            dropped_candidate_hops=dropped_action_count,
            comparison_budget=self.budget.max_comparisons,
            estimated_prompt_tokens=prompt_tokens,
            embedding_models_available=embedding_models,
        )
        context = ReasoningContext(
            question=belief.question,
            answerability=belief.answerability,
            missing_roles=belief.missing_roles,
            accepted_hypotheses=hypotheses[0],
            rejected_hypotheses=hypotheses[1],
            unresolved_hypotheses=hypotheses[2],
            contradictions=belief.contradictions,
            acquired_evidence_refs=tuple(
                value for value in belief.acquired_evidence if value in node_ids
            ),
            recent_hops=recent,
            local_node_ids=tuple(sorted(node_ids)),
            local_edge_ids=edge_ids,
            candidate_hops=tuple(reasoning_hop_from_action(action) for action in retained),
            audit=audit,
        )
        return BuiltReasoningContext(context, local_belief, local_overlay, retained)

    def _rank_actions(
        self,
        actions: list[GraphReadAction],
        embedding_scores: Mapping[str, float] | None,
    ) -> list[GraphReadAction]:
        if not embedding_scores:
            return sorted(
                actions,
                key=lambda action: (
                    action.action_type is NavigationActionType.STOP,
                    _stable_action_digest(action),
                ),
            )
        invalid = {
            node_id: value
            for node_id, value in embedding_scores.items()
            if not math.isfinite(float(value))
        }
        if invalid:
            raise ValueError("embedding_scores must contain only finite values")
        indexed = list(enumerate(actions))
        return [
            action
            for _, action in sorted(
                indexed,
                key=lambda item: (
                    item[1].action_type is NavigationActionType.STOP,
                    -max(
                        (
                            float(embedding_scores.get(target, 0.0))
                            for target in item[1].target_ids
                        ),
                        default=0.0,
                    ),
                    _stable_action_digest(item[1]),
                ),
            )
        ]

    def _retain_actions(
        self,
        actions: list[GraphReadAction],
    ) -> tuple[tuple[GraphReadAction, ...], int]:
        non_stop = [
            action
            for action in actions
            if action.action_type is not NavigationActionType.STOP
        ]
        stop = next(
            (
                action
                for action in actions
                if action.action_type is NavigationActionType.STOP
            ),
            None,
        )
        reserve_stop = 1 if stop is not None else 0
        retained = non_stop[: max(0, self.budget.max_candidate_hops - reserve_stop)]
        if stop is not None:
            retained.append(stop)
        return tuple(retained), max(0, len(actions) - len(retained))

    def _bounded_nodes(
        self,
        actions: tuple[GraphReadAction, ...],
        overlay: CausalTemporalOverlay,
    ) -> tuple[set[str], tuple[str, ...]]:
        known = {
            node.node_id: node
            for node in overlay.atomic_events + overlay.l1_observations
        }
        ordered: list[str] = []
        for action in actions:
            for node_id in (
                ((action.source_id,) if action.source_id else ())
                + action.target_ids
            ):
                if node_id in known and node_id not in ordered:
                    ordered.append(node_id)
        kept = ordered[: self.budget.max_nodes]
        return set(kept), tuple(ordered[self.budget.max_nodes :])

    @staticmethod
    def _trim_action(action: GraphReadAction, node_ids: set[str]) -> GraphReadAction:
        return replace(
            action,
            source_id=action.source_id if action.source_id in node_ids else None,
            target_ids=tuple(target for target in action.target_ids if target in node_ids),
        )

    def _local_overlay(
        self,
        overlay: CausalTemporalOverlay,
        node_ids: set[str],
    ) -> tuple[CausalTemporalOverlay, tuple[str, ...], tuple[str, ...]]:
        selected_l1 = [node for node in overlay.l1_observations if node.node_id in node_ids]
        selected_l1_ids = {node.node_id for node in selected_l1}
        selected_events = [
            replace(
                node,
                source_segments=[ref for ref in node.source_segments if ref in selected_l1_ids],
            )
            for node in overlay.atomic_events
            if node.node_id in node_ids
        ]
        candidate_relations = [
            edge
            for edge in overlay.relations + overlay.l1_structural_relations
            if edge.src in node_ids and edge.dst in node_ids
        ]
        selected_edges = candidate_relations[: self.budget.max_edges]
        edge_ids = tuple(edge.edge_id for edge in selected_edges)
        selected_set = set(edge_ids)
        all_edges = overlay.relations + overlay.l1_structural_relations
        dropped = tuple(
            edge.edge_id for edge in all_edges if edge.edge_id not in selected_set
        )
        return (
            CausalTemporalOverlay(
                overlay_id=f"{overlay.overlay_id}:reasoning-context",
                example_id=overlay.example_id,
                video_id=overlay.video_id,
                l1_observations=selected_l1,
                atomic_events=selected_events,
                relations=[
                    edge for edge in overlay.relations if edge.edge_id in selected_set
                ],
                l1_structural_relations=[
                    edge
                    for edge in overlay.l1_structural_relations
                    if edge.edge_id in selected_set
                ],
                metadata={"context_view_of": overlay.overlay_id},
            ),
            edge_ids,
            dropped,
        )

    @staticmethod
    def _local_belief(
        belief: BeliefSnapshot,
        node_ids: set[str],
        edge_ids: tuple[str, ...],
    ) -> BeliefSnapshot:
        edge_set = set(edge_ids)
        return replace(
            belief,
            acquired_evidence=tuple(
                value for value in belief.acquired_evidence if value in node_ids
            ),
            frontier=tuple(value for value in belief.frontier if value in node_ids),
            relation_states=tuple(
                state for state in belief.relation_states if state.edge_id in edge_set
            ),
            priority_edge_ids=tuple(
                value for value in belief.priority_edge_ids if value in edge_set
            ),
            blocked_edge_ids=tuple(
                value for value in belief.blocked_edge_ids if value in edge_set
            ),
        )

    @staticmethod
    def _categorical_hypotheses(
        belief: BeliefSnapshot,
        edge_ids: set[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        accepted: list[str] = []
        rejected: list[str] = []
        unresolved: list[str] = []
        for state in belief.relation_states:
            if state.edge_id not in edge_ids:
                continue
            if state.grounding is RelationGrounding.VERIFIED:
                accepted.append(state.edge_id)
            elif state.grounding is RelationGrounding.CONTRADICTED:
                rejected.append(state.edge_id)
            else:
                unresolved.append(state.edge_id)
        return tuple(accepted), tuple(rejected), tuple(unresolved)


def _estimate_prompt_tokens(
    belief: BeliefSnapshot,
    overlay: CausalTemporalOverlay,
    actions: tuple[GraphReadAction, ...],
    hypotheses: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    recent_hops: tuple[str, ...],
) -> int:
    """Cheap audit estimate; it is never an LLM-produced value."""

    text = " ".join(
        [belief.question, *belief.missing_roles, *belief.contradictions, *recent_hops]
        + [node.text or "" for node in overlay.atomic_events + overlay.l1_observations]
        + [action.rationale for action in actions]
        + [value for group in hypotheses for value in group]
    )
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _stable_action_digest(action: GraphReadAction) -> str:
    """Permutation-invariant budget key; never an action-value heuristic."""

    payload = {
        "type": action.action_type.value,
        "source": action.source_id,
        "targets": list(action.target_ids),
        "relation": action.relation,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
