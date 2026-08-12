"""Categorical relation verification after an executed graph read."""

from __future__ import annotations

from dataclasses import replace

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay, RelationBelief
from memory_graph.verifiers import verify_relation
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    GraphReadExecution,
)

from .measurement import MeasurementOutcome, VerifierDecision


_INCONCLUSIVE_MARKERS = (
    "lacks ",
    "has no structured",
    "requires one accepted",
    "unknown ",
    "unsupported ",
    "not grounded",
    "must include",
    "must occur",
    "must cite",
    "non-empty warrant",
    "no resolvable",
)


class PostReadCategoricalVerifier:
    """Run deterministic verifiers without consulting persisted decisions."""

    name = "post_read_categorical_relation_verifier"
    version = "memory_graph.verify_relation/post-read-v1"

    def verify(
        self,
        *,
        action: GraphReadAction,
        execution: GraphReadExecution,
        overlay: CausalTemporalOverlay,
        edge_id: str,
        acquired_observation_ids: tuple[str, ...],
    ) -> VerifierDecision:
        if action.action_type is NavigationActionType.STOP:
            raise ValueError("stop action cannot be verified")
        if execution.skill_invocation.get("status") != "executed":
            raise ValueError("post-read verification requires an executed skill")
        relation = str(action.relation or "")
        if not relation:
            raise ValueError("post-read relation verification requires action.relation")
        edges = {
            edge.edge_id: edge
            for edge in overlay.relations + overlay.l1_structural_relations
        }
        edge = edges.get(edge_id)
        if edge is None or relation not in edge.relation_probabilities:
            raise ValueError("post-read verifier received an unknown edge relation")
        action_refs = set(action.target_ids)
        if action.source_id:
            action_refs.add(action.source_id)
        if not {edge.src, edge.dst} <= action_refs:
            raise ValueError("executed action does not cover verifier endpoints")
        observed = set(acquired_observation_ids)
        observed.update(node.node_id for node in execution.observations)
        if not {edge.src, edge.dst} <= observed:
            raise ValueError("both verifier endpoints must be acquired observations")

        by_id = {
            node.node_id: node
            for node in overlay.atomic_events + overlay.l1_observations
        }
        src, dst = by_id[edge.src], by_id[edge.dst]
        clean = _without_persisted_decisions(edge, relation)
        result = verify_relation(
            clean,
            src,
            dst,
            relation,
            l1_by_id={node.node_id: node for node in overlay.l1_observations},
        )
        outcome = (
            MeasurementOutcome.SUPPORTS
            if result.passed
            else _failed_outcome(result.reasons)
        )
        evidence_refs = tuple(
            dict.fromkeys(src.source_segments + dst.source_segments)
        )
        if not evidence_refs:
            raise ValueError("post-read verifier endpoints lack grounded L1 evidence")
        return VerifierDecision(
            edge_id=edge.edge_id,
            relation=relation,
            outcome=outcome,
            verifier_name=self.name,
            verifier_version=self.version,
            evidence_refs=evidence_refs,
            reasons=result.reasons,
        )


def _without_persisted_decisions(
    edge: RelationBelief,
    relation: str,
) -> RelationBelief:
    provenance = {
        key: value
        for key, value in edge.provenance.items()
        if key not in {"hard_verifier", "visual_verification"}
    }
    return replace(
        edge,
        relation_probabilities={relation: edge.relation_probabilities[relation]},
        provenance=provenance,
    )


def _failed_outcome(reasons: tuple[str, ...]) -> MeasurementOutcome:
    normalized = " ".join(reason.casefold() for reason in reasons)
    if any(marker in normalized for marker in _INCONCLUSIVE_MARKERS):
        return MeasurementOutcome.INCONCLUSIVE
    return MeasurementOutcome.REJECTS
