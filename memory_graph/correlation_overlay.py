"""Categorical, question-independent L1.5 correlations over retained L1 nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from itertools import combinations
from typing import Any, Protocol, Sequence

from .types import CausalTemporalOverlay, MemoryNode, RelationBelief


class CorrelationType(str, Enum):
    SEMANTIC_RECURRENCE = "semantic_recurrence"
    ENTITY_RECURRENCE = "entity_recurrence"
    STATE_CONTINUITY = "state_continuity"
    STATE_TRANSITION_CANDIDATE = "state_transition_candidate"
    TRANSITION_SUPPORT = "transition_support"
    RESPONSE_CANDIDATE = "response_candidate"
    MISSING_BRIDGE_CANDIDATE = "missing_bridge_candidate"
    CONTRADICTION_CANDIDATE = "contradiction_candidate"
    EXPLAINS = "explains"
    ENABLES = "enables"


class CorrelationStatus(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class CorrelationPair:
    """One pair presented to a categorical relation evaluator.

    Graph construction is question-independent, so this record contains no
    question, answer key, belief, planner trace, or reward-like field.
    """

    src: MemoryNode
    dst: MemoryNode
    native_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoricalCorrelationJudgment:
    src: str
    dst: str
    relation: CorrelationType
    status: CorrelationStatus
    evidence_refs: tuple[str, ...]
    candidate_sources: tuple[str, ...]
    alignment: dict[str, Any] = field(default_factory=dict)
    verifier_result: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.src or not self.dst or self.src == self.dst:
            raise ValueError("correlation judgment endpoints are invalid")
        if not self.evidence_refs:
            raise ValueError("correlation judgment requires evidence references")
        if not self.candidate_sources:
            raise ValueError("correlation judgment requires a candidate source")


class CategoricalCorrelationEvaluator(Protocol):
    evaluator_name: str

    def evaluate(
        self,
        pairs: Sequence[CorrelationPair],
    ) -> Sequence[CategoricalCorrelationJudgment]: ...


@dataclass(frozen=True)
class CorrelationEdge:
    edge_id: str
    src: str
    dst: str
    relation: CorrelationType
    status: CorrelationStatus
    evidence_refs: tuple[str, ...]
    candidate_sources: tuple[str, ...]
    alignment: dict[str, Any] = field(default_factory=dict)
    verifier_result: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.edge_id or not self.src or not self.dst or self.src == self.dst:
            raise ValueError("correlation edge identifiers are invalid")
        if not self.evidence_refs:
            raise ValueError("correlation edge requires evidence references")
        if not self.candidate_sources:
            raise ValueError("correlation edge requires candidate provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation.value,
            "status": self.status.value,
            "evidence_refs": list(self.evidence_refs),
            "candidate_sources": list(self.candidate_sources),
            "alignment": self.alignment,
            "verifier_result": self.verifier_result,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CategoricalCorrelationOverlay:
    overlay_id: str
    node_ids: tuple[str, ...]
    edges: tuple[CorrelationEdge, ...]
    evaluated_pair_count: int
    evaluator_name: str | None
    schema_version: str = "steam-categorical-l1.5-correlation/v0.1"

    def __post_init__(self) -> None:
        known = set(self.node_ids)
        if len(known) != len(self.node_ids):
            raise ValueError("categorical overlay contains duplicate nodes")
        if any(edge.src not in known or edge.dst not in known for edge in self.edges):
            raise ValueError("categorical overlay contains an orphan edge")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "overlay_id": self.overlay_id,
            "node_ids": list(self.node_ids),
            "edges": [edge.to_dict() for edge in self.edges],
            "build_audit": {
                "pair_generation": "all_retained_node_pairs_no_top_k",
                "evaluated_pair_count": self.evaluated_pair_count,
                "emitted_edge_count": len(self.edges),
                "evaluator": self.evaluator_name,
                "question_independent": True,
                "numeric_relation_output": False,
            },
        }


def build_categorical_correlation_overlay(
    nodes: Sequence[MemoryNode],
    *,
    evaluator: CategoricalCorrelationEvaluator | None = None,
    seed_edges: Sequence[CorrelationEdge] = (),
    overlay_id: str | None = None,
) -> CategoricalCorrelationOverlay:
    """Evaluate every retained pair without embedding Top-K pruning."""

    ordered = tuple(
        sorted(
            nodes,
            key=lambda node: (
                node.time_span.start_s,
                node.time_span.end_s,
                node.node_id,
            ),
        )
    )
    by_id = {node.node_id: node for node in ordered}
    if len(by_id) != len(ordered):
        raise ValueError("retained L1 nodes must have unique IDs")
    known = set(by_id)
    if any(edge.src not in known or edge.dst not in known for edge in seed_edges):
        raise ValueError("seed correlation edge references a non-retained node")

    hints: dict[tuple[str, str], set[str]] = {}
    for edge in seed_edges:
        key = _ordered_pair(edge.src, edge.dst, by_id)
        hints.setdefault(key, set()).add(edge.relation.value)
    pairs = tuple(
        CorrelationPair(
            src=left,
            dst=right,
            native_hints=tuple(sorted(hints.get((left.node_id, right.node_id), set()))),
        )
        for left, right in combinations(ordered, 2)
    )

    judgments = list(_structural_candidate_judgments(pairs))
    if evaluator is not None:
        evaluated = list(evaluator.evaluate(pairs))
        for judgment in evaluated:
            if judgment.src not in known or judgment.dst not in known:
                raise ValueError("evaluator returned a non-retained endpoint")
        judgments.extend(evaluated)

    generated = [_edge_from_judgment(judgment, evaluator) for judgment in judgments]
    edges = _deduplicate_edges([*seed_edges, *generated], by_id)
    video_id = ordered[0].video_id if ordered else "empty"
    return CategoricalCorrelationOverlay(
        overlay_id=overlay_id or f"categorical-l1.5:{video_id}",
        node_ids=tuple(node.node_id for node in ordered),
        edges=tuple(edges),
        evaluated_pair_count=len(pairs),
        evaluator_name=(
            getattr(evaluator, "evaluator_name", None) if evaluator else None
        ),
    )


def project_legacy_overlay_correlations(
    overlay: CausalTemporalOverlay,
    *,
    retained_node_ids: Sequence[str],
    id_map: dict[str, str | None] | None = None,
) -> tuple[CorrelationEdge, ...]:
    """Project legacy event/L1 relations onto retained L1 IDs categorically."""

    retained = set(retained_node_ids)
    mapping = id_map or {node_id: node_id for node_id in retained}
    event_sources = {
        node.node_id: tuple(
            dict.fromkeys(
                mapped
                for source in node.source_segments
                if (mapped := mapping.get(source)) is not None and mapped in retained
            )
        )
        for node in overlay.atomic_events
    }
    edges: list[CorrelationEdge] = []
    for relation in overlay.l1_structural_relations:
        src = mapping.get(relation.src)
        dst = mapping.get(relation.dst)
        if (
            src is None
            or dst is None
            or src == dst
            or src not in retained
            or dst not in retained
        ):
            continue
        edges.extend(_legacy_relation_edges(relation, ((src, dst),)))
    for relation in overlay.relations:
        endpoint_pairs = tuple(
            (src, dst)
            for src in event_sources.get(relation.src, ())
            for dst in event_sources.get(relation.dst, ())
            if src != dst
        )
        edges.extend(_legacy_relation_edges(relation, endpoint_pairs))
    by_id = {
        node.node_id: node
        for node in overlay.l1_observations
        if node.node_id in retained
    }
    # Consolidated IDs are not present in the legacy overlay, so orientation is
    # already inherited from the source relation for those endpoints.
    return tuple(_deduplicate_edges(edges, by_id, tolerate_unknown_order=True))


def _structural_candidate_judgments(
    pairs: Sequence[CorrelationPair],
) -> Sequence[CategoricalCorrelationJudgment]:
    judgments: list[CategoricalCorrelationJudgment] = []
    for pair in pairs:
        left_entities = _stable_entity_keys(pair.src)
        right_entities = _stable_entity_keys(pair.dst)
        shared_entities = tuple(sorted(left_entities & right_entities))
        if shared_entities:
            judgments.append(
                CategoricalCorrelationJudgment(
                    src=pair.src.node_id,
                    dst=pair.dst.node_id,
                    relation=CorrelationType.ENTITY_RECURRENCE,
                    status=CorrelationStatus.CANDIDATE,
                    evidence_refs=(pair.src.node_id, pair.dst.node_id),
                    candidate_sources=("shared_stable_entity_key",),
                    alignment={"stable_entity_keys": list(shared_entities)},
                    rationale="shared stable entity key proposes an identity inspection",
                )
            )
        for subject, attribute, left_value, right_value in _aligned_states(
            pair.src, pair.dst
        ):
            relation = (
                CorrelationType.STATE_CONTINUITY
                if left_value == right_value
                else CorrelationType.STATE_TRANSITION_CANDIDATE
            )
            judgments.append(
                CategoricalCorrelationJudgment(
                    src=pair.src.node_id,
                    dst=pair.dst.node_id,
                    relation=relation,
                    status=CorrelationStatus.CANDIDATE,
                    evidence_refs=(pair.src.node_id, pair.dst.node_id),
                    candidate_sources=("same_subject_same_attribute",),
                    alignment={
                        "subject": subject,
                        "attribute": attribute,
                        "before": left_value,
                        "after": right_value,
                    },
                    rationale="same subject and attribute require an explicit state gate",
                )
            )
    return judgments


def _stable_entity_keys(node: MemoryNode) -> set[str]:
    keys: set[str] = set()
    for participant in node.metadata.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        for name in ("track_id", "entity_id", "visual_signature"):
            value = str(participant.get(name) or "").strip()
            if value:
                keys.add(f"{name}:{value}")
    for name in ("track_id", "entity_id", "visual_signature"):
        value = str(node.metadata.get(name) or "").strip()
        if value:
            keys.add(f"{name}:{value}")
    return keys


def _aligned_states(
    left: MemoryNode,
    right: MemoryNode,
) -> tuple[tuple[str, str, str, str], ...]:
    left_states = _state_map(left)
    right_states = _state_map(right)
    aligned: list[tuple[str, str, str, str]] = []
    for key in sorted(set(left_states) & set(right_states)):
        aligned.append((key[0], key[1], left_states[key], right_states[key]))
    return tuple(aligned)


def _state_map(node: MemoryNode) -> dict[tuple[str, str], str]:
    mention_to_stable: dict[str, str] = {}
    for participant in node.metadata.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        mention = str(participant.get("mention_id") or "")
        stable = next(
            (
                str(participant.get(name))
                for name in ("track_id", "entity_id", "visual_signature")
                if participant.get(name)
            ),
            mention,
        )
        if mention and stable:
            mention_to_stable[mention] = stable
    result: dict[tuple[str, str], str] = {}
    for state in node.metadata.get("states") or []:
        if not isinstance(state, dict):
            continue
        mention = str(state.get("mention_id") or state.get("subject_id") or "")
        subject = mention_to_stable.get(mention, mention)
        attribute = str(state.get("attribute") or "")
        value = str(state.get("value") or "")
        if subject and attribute and value:
            result[(subject, attribute)] = value
    return result


def _legacy_relation_edges(
    relation: RelationBelief,
    endpoint_pairs: Sequence[tuple[str, str]],
) -> list[CorrelationEdge]:
    mapping = {
        "same_entity": CorrelationType.ENTITY_RECURRENCE,
        "same_object": CorrelationType.ENTITY_RECURRENCE,
        "same_instance_candidate": CorrelationType.ENTITY_RECURRENCE,
        "reappears_candidate": CorrelationType.ENTITY_RECURRENCE,
        "state_transition": CorrelationType.STATE_TRANSITION_CANDIDATE,
        "transition_support": CorrelationType.TRANSITION_SUPPORT,
        "observation_support": CorrelationType.TRANSITION_SUPPORT,
        "response_candidate": CorrelationType.RESPONSE_CANDIDATE,
        "contradicts": CorrelationType.CONTRADICTION_CANDIDATE,
        "explains": CorrelationType.EXPLAINS,
        "enables": CorrelationType.ENABLES,
    }
    result: list[CorrelationEdge] = []
    for relation_name in sorted(relation.relation_probabilities):
        mapped = mapping.get(relation_name)
        if mapped is None:
            continue
        for src, dst in endpoint_pairs:
            status = _legacy_categorical_status(relation, relation_name)
            digest = hashlib.sha256(
                f"{relation.edge_id}\x1f{src}\x1f{dst}\x1f{mapped.value}".encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            result.append(
                CorrelationEdge(
                    edge_id=f"correlation:legacy:{digest}",
                    src=src,
                    dst=dst,
                    relation=mapped,
                    status=status,
                    evidence_refs=tuple(
                        dict.fromkeys((src, dst, *relation.evidence_refs))
                    ),
                    candidate_sources=("legacy_l1_l1.5_projection",),
                    alignment=_legacy_alignment(relation),
                    verifier_result=_legacy_verifier_record(relation),
                    provenance={
                        "source_edge_id": relation.edge_id,
                        "source_status": relation.status.value,
                        "source_relation": relation_name,
                        "numeric_prior_ignored_for_admission": True,
                    },
                )
            )
    return result


def _legacy_categorical_status(
    relation: RelationBelief,
    relation_name: str,
) -> CorrelationStatus:
    explicit = str(
        relation.provenance.get("admission_status")
        or relation.provenance.get("verification_status")
        or ""
    ).lower()
    if explicit in {"verified", "accepted", "passed"}:
        return CorrelationStatus.VERIFIED
    if explicit in {"rejected", "failed"}:
        return CorrelationStatus.REJECTED
    if explicit in {"inconclusive", "pending"}:
        return CorrelationStatus.INCONCLUSIVE
    hard = relation.provenance.get("hard_verifier")
    hard_record = hard.get(relation_name) if isinstance(hard, dict) else None
    visual = relation.provenance.get("visual_verification")
    hard_passed = isinstance(hard_record, dict) and hard_record.get("passed") is True
    visual_passed = isinstance(visual, dict) and visual.get("status") == "passed"
    if relation_name in {"explains", "enables"}:
        return (
            CorrelationStatus.VERIFIED
            if hard_passed and visual_passed
            else CorrelationStatus.CANDIDATE
        )
    if hard_passed:
        return CorrelationStatus.VERIFIED
    return CorrelationStatus.CANDIDATE


def _legacy_verifier_record(relation: RelationBelief) -> dict[str, Any]:
    return {
        key: relation.provenance[key]
        for key in (
            "hard_verifier",
            "visual_verification",
            "identity_gate",
            "state_gate",
        )
        if key in relation.provenance
    }


def _legacy_alignment(relation: RelationBelief) -> dict[str, Any]:
    value = relation.provenance.get("participant_alignment")
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return {"records": value}
    return {}


def _edge_from_judgment(
    judgment: CategoricalCorrelationJudgment,
    evaluator: CategoricalCorrelationEvaluator | None,
) -> CorrelationEdge:
    digest = hashlib.sha256(
        "\x1f".join(
            (
                judgment.src,
                judgment.dst,
                judgment.relation.value,
                *judgment.candidate_sources,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    return CorrelationEdge(
        edge_id=f"correlation:categorical:{digest}",
        src=judgment.src,
        dst=judgment.dst,
        relation=judgment.relation,
        status=judgment.status,
        evidence_refs=judgment.evidence_refs,
        candidate_sources=judgment.candidate_sources,
        alignment=judgment.alignment,
        verifier_result=judgment.verifier_result,
        provenance={
            "producer": getattr(evaluator, "evaluator_name", None)
            or "memory_graph.correlation_overlay.structural_candidates",
            "question_independent": True,
            "categorical_output_only": True,
            "rationale": judgment.rationale,
        },
    )


def _ordered_pair(
    left_id: str,
    right_id: str,
    by_id: dict[str, MemoryNode],
) -> tuple[str, str]:
    left = by_id[left_id]
    right = by_id[right_id]
    if (left.time_span.start_s, left.node_id) <= (
        right.time_span.start_s,
        right.node_id,
    ):
        return left_id, right_id
    return right_id, left_id


def _deduplicate_edges(
    edges: Sequence[CorrelationEdge],
    by_id: dict[str, MemoryNode],
    *,
    tolerate_unknown_order: bool = False,
) -> list[CorrelationEdge]:
    rank = {
        CorrelationStatus.CANDIDATE: 0,
        CorrelationStatus.INCONCLUSIVE: 1,
        CorrelationStatus.REJECTED: 2,
        CorrelationStatus.VERIFIED: 3,
    }
    selected: dict[tuple[str, str, CorrelationType], CorrelationEdge] = {}
    for edge in edges:
        if edge.src in by_id and edge.dst in by_id:
            src, dst = _ordered_pair(edge.src, edge.dst, by_id)
        elif tolerate_unknown_order:
            src, dst = edge.src, edge.dst
        else:
            raise ValueError("correlation edge order cannot be resolved")
        normalized = (
            edge
            if (src, dst) == (edge.src, edge.dst)
            else CorrelationEdge(
                edge_id=edge.edge_id,
                src=src,
                dst=dst,
                relation=edge.relation,
                status=edge.status,
                evidence_refs=edge.evidence_refs,
                candidate_sources=edge.candidate_sources,
                alignment=edge.alignment,
                verifier_result=edge.verifier_result,
                provenance=edge.provenance,
            )
        )
        key = (src, dst, normalized.relation)
        existing = selected.get(key)
        if existing is None or rank[normalized.status] > rank[existing.status]:
            selected[key] = normalized
    return sorted(
        selected.values(),
        key=lambda edge: (edge.src, edge.dst, edge.relation.value, edge.edge_id),
    )
