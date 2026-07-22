"""Question-independent soft navigation correlations over semantic L1 nodes.

The values in this module are produced by a learned embedding representation,
not by an LLM judgment.  Cosine similarity is retained as a similarity feature;
it is never presented as a calibrated probability or a verified semantic fact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from statistics import fmean
from typing import Mapping, Sequence

from .types import MemoryNode


SOFT_CORRELATION_SCHEMA = "steam-soft-l1.5-navigation-correlation/v0.2"
SOFT_CORRELATION_PAIR_AUDIT_SCHEMA = "steam-soft-l1.5-pair-audit/v0.1"
SOFT_CORRELATION_CALIBRATION_SCHEMA = "steam-soft-l1.5-calibration/v0.1"
SEMANTIC_EQUIVALENCE_COSINE = 0.999


@dataclass(frozen=True)
class SoftCorrelationAdmissionPolicy:
    """Frozen, question-independent admission contract for soft L1.5 edges.

    This is a global calibration policy, never a per-node Top-K rule.  It may
    filter an edge produced by the structural sparsemax/recurrence builder, but
    it cannot manufacture a relation or turn similarity into a verified fact.
    """

    policy_id: str = "positive-standardized-sparsemax/v1"
    minimum_semantic_similarity: float = 0.0
    minimum_directional_affinity: float = 0.0
    allowed_channels: tuple[str, ...] = ("semantic", "semantic_recurrence")
    calibration_source: str = "structural_default_no_labeled_negatives"
    calibration_split: str | None = None
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id or not self.calibration_source:
            raise ValueError("soft-correlation policy requires identifiers")
        if not -1.0 <= self.minimum_semantic_similarity <= 1.0:
            raise ValueError("minimum semantic similarity must be in [-1, 1]")
        if not 0.0 <= self.minimum_directional_affinity <= 1.0:
            raise ValueError("minimum directional affinity must be in [0, 1]")
        supported = {"semantic", "semantic_recurrence"}
        if not self.allowed_channels or not set(self.allowed_channels) <= supported:
            raise ValueError("soft-correlation policy has unsupported channels")
        if not self.frozen:
            raise ValueError("runtime correlation admission policy must be frozen")

    def rejection_reasons(
        self, edge: "SoftNavigationCorrelation"
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if edge.channel not in self.allowed_channels:
            reasons.append("channel_not_admitted")
        if edge.semantic_similarity < self.minimum_semantic_similarity:
            reasons.append("below_global_similarity_threshold")
        if edge.navigation_affinity < self.minimum_directional_affinity:
            reasons.append("below_global_affinity_threshold")
        return tuple(reasons)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "minimum_semantic_similarity": self.minimum_semantic_similarity,
            "minimum_directional_affinity": self.minimum_directional_affinity,
            "allowed_channels": list(self.allowed_channels),
            "calibration_source": self.calibration_source,
            "calibration_split": self.calibration_split,
            "frozen": self.frozen,
            "question_independent": True,
            "top_k_applied": False,
            "similarity_is_verified_fact": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SoftCorrelationAdmissionPolicy":
        return cls(
            policy_id=str(payload.get("policy_id") or ""),
            minimum_semantic_similarity=float(
                payload.get("minimum_semantic_similarity", 0.0)
            ),
            minimum_directional_affinity=float(
                payload.get("minimum_directional_affinity", 0.0)
            ),
            allowed_channels=tuple(
                str(value)
                for value in payload.get(
                    "allowed_channels", ("semantic", "semantic_recurrence")
                )
            ),
            calibration_source=str(payload.get("calibration_source") or ""),
            calibration_split=(
                str(payload["calibration_split"])
                if payload.get("calibration_split") is not None
                else None
            ),
            frozen=payload.get("frozen") is True,
        )


@dataclass(frozen=True)
class SoftCorrelationPairAudit:
    """Question-independent features and admission outcome for one L1 pair."""

    src: str
    dst: str
    semantic_similarity: float
    temporal_gap_s: float
    src_node_type: str
    dst_node_type: str
    src_modality: str
    dst_modality: str
    same_node_type: bool
    same_modality: bool
    excluded_temporal_pair: bool
    same_semantic_class: bool
    candidate_channel: str
    admission: str
    admission_reasons: tuple[str, ...]
    src_to_dst_affinity: float = 0.0
    dst_to_src_affinity: float = 0.0
    emitted_edge_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "src": self.src,
            "dst": self.dst,
            "features": {
                "semantic_similarity": self.semantic_similarity,
                "temporal_gap_s": self.temporal_gap_s,
                "src_node_type": self.src_node_type,
                "dst_node_type": self.dst_node_type,
                "src_modality": self.src_modality,
                "dst_modality": self.dst_modality,
                "same_node_type": self.same_node_type,
                "same_modality": self.same_modality,
                "same_semantic_class": self.same_semantic_class,
            },
            "excluded_temporal_pair": self.excluded_temporal_pair,
            "candidate_channel": self.candidate_channel,
            "admission": self.admission,
            "admission_reasons": list(self.admission_reasons),
            "src_to_dst_affinity": self.src_to_dst_affinity,
            "dst_to_src_affinity": self.dst_to_src_affinity,
            "direction_semantics": (
                "bidirectional_symmetric_semantic_signal"
                if self.emitted_edge_id is not None
                else "not_emitted"
            ),
            "emitted_edge_id": self.emitted_edge_id,
        }


@dataclass(frozen=True)
class SoftNavigationCorrelation:
    """One sparse, embedding-derived L1.5 navigation edge."""

    edge_id: str
    src: str
    dst: str
    semantic_similarity: float
    src_to_dst_affinity: float
    dst_to_src_affinity: float
    score_source: str
    evidence_refs: tuple[str, ...]
    provenance: dict[str, object]
    channel: str = "semantic"

    def __post_init__(self) -> None:
        if not self.edge_id or not self.src or not self.dst or self.src == self.dst:
            raise ValueError("soft correlation edge identifiers are invalid")
        if self.channel not in {"semantic", "semantic_recurrence"}:
            raise ValueError(f"unsupported soft correlation channel: {self.channel}")
        if not -1.0 <= self.semantic_similarity <= 1.0:
            raise ValueError("semantic similarity must be in [-1, 1]")
        if not 0.0 <= self.src_to_dst_affinity <= 1.0:
            raise ValueError("src_to_dst_affinity must be in [0, 1]")
        if not 0.0 <= self.dst_to_src_affinity <= 1.0:
            raise ValueError("dst_to_src_affinity must be in [0, 1]")
        if max(self.src_to_dst_affinity, self.dst_to_src_affinity) <= 0.0:
            raise ValueError("soft correlation edge must have positive affinity")
        if not self.score_source or not self.evidence_refs:
            raise ValueError("soft correlation requires score source and evidence refs")

    @property
    def navigation_affinity(self) -> float:
        return max(self.src_to_dst_affinity, self.dst_to_src_affinity)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SOFT_CORRELATION_SCHEMA,
            "edge_id": self.edge_id,
            "src": self.src,
            "dst": self.dst,
            "channel": self.channel,
            "semantic_similarity": self.semantic_similarity,
            "src_to_dst_affinity": self.src_to_dst_affinity,
            "dst_to_src_affinity": self.dst_to_src_affinity,
            "navigation_affinity": self.navigation_affinity,
            "score_source": self.score_source,
            "score_semantics": "embedding_similarity_and_sparse_navigation_affinity_not_probability",
            "evidence_refs": list(self.evidence_refs),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SoftCorrelationBuildResult:
    edges: tuple[SoftNavigationCorrelation, ...]
    evaluated_pair_count: int
    excluded_temporal_pair_count: int
    score_source: str
    node_count: int
    semantic_class_count: int
    recurrence_edge_count: int
    pair_audits: tuple[SoftCorrelationPairAudit, ...]
    admission_policy: SoftCorrelationAdmissionPolicy
    l1_fingerprint: str

    def audit_dict(self) -> dict[str, object]:
        possible = self.node_count * (self.node_count - 1) // 2
        navigable_possible = possible - self.excluded_temporal_pair_count
        degrees: dict[str, int] = {}
        for edge in self.edges:
            degrees[edge.src] = degrees.get(edge.src, 0) + 1
            degrees[edge.dst] = degrees.get(edge.dst, 0) + 1
        admission_counts: dict[str, int] = {}
        reason_counts: dict[str, int] = {}
        for row in self.pair_audits:
            admission_counts[row.admission] = admission_counts.get(row.admission, 0) + 1
            for reason in row.admission_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return {
            "schema_version": SOFT_CORRELATION_SCHEMA,
            "node_count": self.node_count,
            "evaluated_pair_count": self.evaluated_pair_count,
            "excluded_temporal_pair_count": self.excluded_temporal_pair_count,
            "emitted_edge_count": len(self.edges),
            "possible_pair_count": possible,
            "navigable_possible_pair_count": navigable_possible,
            "edge_density": (
                len(self.edges) / navigable_possible if navigable_possible else 0.0
            ),
            "maximum_degree": max(degrees.values(), default=0),
            "mean_nonzero_degree": fmean(degrees.values()) if degrees else 0.0,
            "score_source": self.score_source,
            "pair_score": "cosine_similarity",
            "semantic_equivalence_cosine": SEMANTIC_EQUIVALENCE_COSINE,
            "semantic_equivalence_class_count": self.semantic_class_count,
            "recurrence_edge_count": self.recurrence_edge_count,
            "pair_audit_schema": SOFT_CORRELATION_PAIR_AUDIT_SCHEMA,
            "pair_audit_row_count": len(self.pair_audits),
            "pair_admission_counts": admission_counts,
            "pair_admission_reason_counts": reason_counts,
            "admission_policy": self.admission_policy.to_dict(),
            "admission_policy_fingerprint": self.admission_policy.fingerprint,
            "l1_fingerprint": self.l1_fingerprint,
            "sparsifier": (
                "near_identity_equivalence_recurrence_chain_plus_"
                "per_class_positive_standardized_sparsemax"
            ),
            "top_k_applied": False,
            "question_independent": True,
            "numeric_values_produced_by_llm": False,
            "similarity_is_probability": False,
        }

    def pair_audit_dict(self) -> dict[str, object]:
        return {
            "schema_version": SOFT_CORRELATION_PAIR_AUDIT_SCHEMA,
            "l1_fingerprint": self.l1_fingerprint,
            "score_source": self.score_source,
            "question_independent": True,
            "contains_question_or_answer": False,
            "model_numeric_reward_present": False,
            "admission_policy": self.admission_policy.to_dict(),
            "admission_policy_fingerprint": self.admission_policy.fingerprint,
            "summary": self.audit_dict(),
            "pairs": [row.to_dict() for row in self.pair_audits],
        }


def build_soft_semantic_correlations(
    nodes: Sequence[MemoryNode],
    embeddings: Mapping[str, Sequence[float]],
    *,
    score_source: str,
    excluded_pairs: Sequence[tuple[str, str]] = (),
    admission_policy: SoftCorrelationAdmissionPolicy | None = None,
) -> SoftCorrelationBuildResult:
    """Build a sparse soft graph from all embedding pairs without Top-K.

    All non-temporal pairs are scored. Near-identical embeddings are treated as
    semantic equivalence classes and linked by recurrence chains. Positive
    class-to-class cosine scores are standardized and projected onto the
    simplex with sparsemax. The resulting values are directional navigation
    affinities, not confidence values.
    """

    policy = admission_policy or SoftCorrelationAdmissionPolicy()
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
    node_ids = {node.node_id for node in ordered}
    if len(node_ids) != len(ordered):
        raise ValueError("soft correlation nodes must have unique IDs")
    missing = sorted(node_ids - set(embeddings))
    if missing:
        raise ValueError("missing node embeddings: " + ", ".join(missing[:5]))
    vectors = {node_id: _normalized(embeddings[node_id]) for node_id in node_ids}
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) > 1:
        raise ValueError("soft correlation embedding dimensions must match")
    embedding_dimension = next(iter(dimensions), 0)
    excluded = {frozenset(pair) for pair in excluded_pairs if pair[0] != pair[1]}
    similarities = {
        _ordered_pair(left.node_id, right.node_id): _cosine(
            vectors[left.node_id], vectors[right.node_id]
        )
        for left_index, left in enumerate(ordered)
        for right in ordered[left_index + 1 :]
    }
    excluded_count = sum(1 for pair in excluded if pair <= node_ids)
    semantic_classes = _semantic_equivalence_classes(ordered, similarities)
    class_by_node = {
        node_id: class_index
        for class_index, members in enumerate(semantic_classes)
        for node_id in members
    }
    class_vectors = {
        class_index: _normalized(
            tuple(
                fmean(vectors[node_id][dimension] for node_id in members)
                for dimension in range(embedding_dimension)
            )
        )
        for class_index, members in enumerate(semantic_classes)
    }
    class_pair_candidates: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            pair = frozenset((left.node_id, right.node_id))
            if pair in excluded:
                continue
            left_class = class_by_node[left.node_id]
            right_class = class_by_node[right.node_id]
            if left_class == right_class:
                continue
            class_pair = (
                (left_class, right_class)
                if left_class < right_class
                else (right_class, left_class)
            )
            class_pair_candidates.setdefault(class_pair, []).append(
                (left.node_id, right.node_id)
            )

    class_similarities = {
        pair: _cosine(class_vectors[pair[0]], class_vectors[pair[1]])
        for pair in class_pair_candidates
    }
    directional: dict[tuple[int, int], float] = {}
    for source_class in range(len(semantic_classes)):
        targets: list[int] = []
        scores: list[float] = []
        for target_class in range(len(semantic_classes)):
            if source_class == target_class:
                continue
            pair = (
                (source_class, target_class)
                if source_class < target_class
                else (target_class, source_class)
            )
            similarity = class_similarities.get(pair)
            if similarity is not None and similarity > 0.0:
                targets.append(target_class)
                scores.append(similarity)
        for target_class, weight in zip(
            targets, _positive_standardized_sparsemax(scores)
        ):
            if weight > 1e-12:
                directional[(source_class, target_class)] = weight

    raw_edges = _recurrence_edges(
        ordered,
        semantic_classes,
        similarities,
        excluded,
        score_source=score_source,
    )
    node_by_id = {node.node_id: node for node in ordered}
    representative_by_class_pair: dict[tuple[int, int], tuple[str, str]] = {}
    for class_pair, candidates in sorted(class_pair_candidates.items()):
        forward = directional.get(class_pair, 0.0)
        backward = directional.get((class_pair[1], class_pair[0]), 0.0)
        if max(forward, backward) <= 1e-12:
            continue
        left_id, right_id = min(
            candidates,
            key=lambda pair: (
                -similarities[_ordered_pair(*pair)],
                abs(_center(node_by_id[pair[0]]) - _center(node_by_id[pair[1]])),
                pair,
            ),
        )
        representative_by_class_pair[class_pair] = (left_id, right_id)
        left_class = class_by_node[left_id]
        right_class = class_by_node[right_id]
        # Embedding cosine is symmetric.  Sparsemax decides whether a class
        # pair is admitted, but it must not fabricate a one-way semantic fact.
        # Both legal directions therefore receive the same navigation affinity.
        symmetric_affinity = max(
            directional.get((left_class, right_class), 0.0),
            directional.get((right_class, left_class), 0.0),
        )
        raw_edges.append(
            _edge(
                left_id,
                right_id,
                channel="semantic",
                similarity=similarities[_ordered_pair(left_id, right_id)],
                forward=symmetric_affinity,
                backward=symmetric_affinity,
                score_source=score_source,
                provenance={
                    "pair_enumeration": "all_non_temporal_pairs",
                    "sparsifier": "per_semantic_class_positive_standardized_sparsemax",
                    "source_semantic_class": left_class,
                    "target_semantic_class": right_class,
                    "class_similarity": class_similarities[class_pair],
                    "representative_policy": (
                        "highest_endpoint_similarity_then_nearest_time"
                    ),
                    "direction_policy": "bidirectional_for_symmetric_semantic_signal",
                },
            )
        )
    emitted_edges: list[SoftNavigationCorrelation] = []
    policy_rejections: dict[tuple[str, str], tuple[str, ...]] = {}
    for edge in raw_edges:
        reasons = policy.rejection_reasons(edge)
        pair = _ordered_pair(edge.src, edge.dst)
        if reasons:
            policy_rejections[pair] = reasons
        else:
            emitted_edges.append(edge)
    emitted_by_pair = {
        _ordered_pair(edge.src, edge.dst): edge for edge in emitted_edges
    }
    raw_by_pair = {_ordered_pair(edge.src, edge.dst): edge for edge in raw_edges}
    pair_audits = _build_pair_audits(
        ordered,
        similarities=similarities,
        excluded=excluded,
        class_by_node=class_by_node,
        class_similarities=class_similarities,
        directional=directional,
        representative_by_class_pair=representative_by_class_pair,
        raw_by_pair=raw_by_pair,
        emitted_by_pair=emitted_by_pair,
        policy_rejections=policy_rejections,
    )
    return SoftCorrelationBuildResult(
        edges=tuple(
            sorted(emitted_edges, key=lambda edge: (edge.src, edge.dst, edge.channel))
        ),
        evaluated_pair_count=len(similarities) - excluded_count,
        excluded_temporal_pair_count=excluded_count,
        score_source=score_source,
        node_count=len(ordered),
        semantic_class_count=len(semantic_classes),
        recurrence_edge_count=sum(
            edge.channel == "semantic_recurrence" for edge in emitted_edges
        ),
        pair_audits=pair_audits,
        admission_policy=policy,
        l1_fingerprint=_l1_fingerprint(ordered),
    )


def calibrate_soft_correlation_admission(
    positive_similarities: Sequence[float],
    *,
    trusted_negative_similarities: Sequence[float] = (),
    target_positive_coverage: float = 0.9,
    minimum_precision: float | None = None,
    calibration_source: str,
    calibration_split: str = "train",
) -> tuple[SoftCorrelationAdmissionPolicy, dict[str, object]]:
    """Fit one global similarity threshold without inventing negative labels.

    When trusted negatives are absent, this routine calibrates positive bridge
    coverage only and reports precision as unavailable.  Unmatched graph pairs
    must never be silently converted into negatives.
    """

    positives = tuple(float(value) for value in positive_similarities)
    negatives = tuple(float(value) for value in trusted_negative_similarities)
    if not positives:
        raise ValueError("correlation calibration requires positive examples")
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in positives):
        raise ValueError("positive calibration similarities must be finite cosines")
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in negatives):
        raise ValueError("negative calibration similarities must be finite cosines")
    if not 0.0 < target_positive_coverage <= 1.0:
        raise ValueError("target positive coverage must be in (0, 1]")
    if minimum_precision is not None and not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum precision must be in (0, 1]")
    if minimum_precision is not None and not negatives:
        raise ValueError("minimum precision requires trusted negative examples")

    thresholds = sorted(
        {0.0, *(max(0.0, value) for value in positives + negatives)},
        reverse=True,
    )
    selected: tuple[float, float, float | None, int, int] | None = None
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        true_positive = sum(value >= threshold for value in positives)
        false_positive = sum(value >= threshold for value in negatives)
        coverage = true_positive / len(positives)
        precision = (
            true_positive / (true_positive + false_positive)
            if negatives and true_positive + false_positive
            else (1.0 if negatives else None)
        )
        satisfies = coverage >= target_positive_coverage and (
            minimum_precision is None
            or (precision is not None and precision >= minimum_precision)
        )
        rows.append(
            {
                "threshold": threshold,
                "positive_coverage": coverage,
                "precision": precision,
                "true_positive_count": true_positive,
                "false_positive_count": false_positive,
                "satisfies_constraints": satisfies,
            }
        )
        if selected is None and satisfies:
            selected = (
                threshold,
                coverage,
                precision,
                true_positive,
                false_positive,
            )
    if selected is None:
        selected = (0.0, 1.0, None if not negatives else 0.0, len(positives), len(negatives))
    threshold, coverage, precision, true_positive, false_positive = selected
    policy = SoftCorrelationAdmissionPolicy(
        policy_id="calibrated-global-similarity-plus-sparsemax/v1",
        minimum_semantic_similarity=threshold,
        calibration_source=calibration_source,
        calibration_split=calibration_split,
    )
    report: dict[str, object] = {
        "schema_version": SOFT_CORRELATION_CALIBRATION_SCHEMA,
        "calibration_source": calibration_source,
        "calibration_split": calibration_split,
        "positive_example_count": len(positives),
        "trusted_negative_example_count": len(negatives),
        "unmatched_pairs_treated_as_negative": False,
        "target_positive_coverage": target_positive_coverage,
        "minimum_precision": minimum_precision,
        "selected_threshold": threshold,
        "selected_positive_coverage": coverage,
        "selected_precision": precision,
        "selected_true_positive_count": true_positive,
        "selected_false_positive_count": false_positive,
        "precision_status": (
            "available_from_trusted_negatives"
            if negatives
            else "unavailable_no_trusted_negative_labels"
        ),
        "policy": policy.to_dict(),
        "policy_fingerprint": policy.fingerprint,
        "candidate_thresholds": rows,
        "top_k_applied": False,
        "training_performed": False,
    }
    return policy, report


def _build_pair_audits(
    nodes: Sequence[MemoryNode],
    *,
    similarities: Mapping[tuple[str, str], float],
    excluded: set[frozenset[str]],
    class_by_node: Mapping[str, int],
    class_similarities: Mapping[tuple[int, int], float],
    directional: Mapping[tuple[int, int], float],
    representative_by_class_pair: Mapping[tuple[int, int], tuple[str, str]],
    raw_by_pair: Mapping[tuple[str, str], SoftNavigationCorrelation],
    emitted_by_pair: Mapping[tuple[str, str], SoftNavigationCorrelation],
    policy_rejections: Mapping[tuple[str, str], tuple[str, ...]],
) -> tuple[SoftCorrelationPairAudit, ...]:
    rows: list[SoftCorrelationPairAudit] = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            pair = _ordered_pair(left.node_id, right.node_id)
            frozen_pair = frozenset(pair)
            left_class = class_by_node[left.node_id]
            right_class = class_by_node[right.node_id]
            same_class = left_class == right_class
            class_pair = (
                (left_class, right_class)
                if left_class < right_class
                else (right_class, left_class)
            )
            raw_edge = raw_by_pair.get(pair)
            emitted = emitted_by_pair.get(pair)
            if frozen_pair in excluded:
                admission = "excluded"
                reasons = ("covered_by_l1_temporal_backbone",)
                channel = "temporal_excluded"
            elif emitted is not None:
                admission = "admitted"
                reasons = (
                    "semantic_recurrence_chain"
                    if emitted.channel == "semantic_recurrence"
                    else "sparse_class_representative"
                ,)
                channel = emitted.channel
            elif pair in policy_rejections:
                admission = "not_admitted"
                reasons = policy_rejections[pair]
                channel = raw_edge.channel if raw_edge is not None else "semantic"
            elif same_class:
                admission = "not_admitted"
                reasons = ("recurrence_chain_omits_transitive_clique_edge",)
                channel = "semantic_recurrence"
            else:
                representative = representative_by_class_pair.get(class_pair)
                class_similarity = class_similarities.get(class_pair)
                if class_similarity is None or class_similarity <= 0.0:
                    reasons = ("nonpositive_class_similarity",)
                elif max(
                    directional.get((left_class, right_class), 0.0),
                    directional.get((right_class, left_class), 0.0),
                ) <= 1e-12:
                    reasons = ("zero_after_sparsemax",)
                elif representative is not None and _ordered_pair(*representative) != pair:
                    reasons = ("semantic_class_representative_only",)
                else:
                    reasons = ("not_selected_by_sparse_builder",)
                admission = "not_admitted"
                channel = "semantic"
            edge_for_values = emitted or raw_edge
            rows.append(
                SoftCorrelationPairAudit(
                    src=left.node_id,
                    dst=right.node_id,
                    semantic_similarity=similarities[pair],
                    temporal_gap_s=max(
                        0.0, right.time_span.start_s - left.time_span.end_s
                    ),
                    src_node_type=left.node_type,
                    dst_node_type=right.node_type,
                    src_modality=_node_modality(left),
                    dst_modality=_node_modality(right),
                    same_node_type=left.node_type == right.node_type,
                    same_modality=_node_modality(left) == _node_modality(right),
                    excluded_temporal_pair=frozen_pair in excluded,
                    same_semantic_class=same_class,
                    candidate_channel=channel,
                    admission=admission,
                    admission_reasons=reasons,
                    src_to_dst_affinity=(
                        edge_for_values.src_to_dst_affinity
                        if edge_for_values is not None
                        else 0.0
                    ),
                    dst_to_src_affinity=(
                        edge_for_values.dst_to_src_affinity
                        if edge_for_values is not None
                        else 0.0
                    ),
                    emitted_edge_id=emitted.edge_id if emitted is not None else None,
                )
            )
    return tuple(rows)


def _positive_standardized_sparsemax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    if len(values) == 1:
        # A single positive cosine has no within-video contrastive baseline.
        return (0.0,)
    mean = fmean(values)
    variance = fmean((value - mean) ** 2 for value in values)
    if variance <= 1e-18:
        return tuple(0.0 for _ in values)
    scale = math.sqrt(variance)
    standardized = tuple((value - mean) / scale for value in values)
    return _sparsemax(standardized)


def _sparsemax(values: Sequence[float]) -> tuple[float, ...]:
    """Euclidean projection onto the simplex (Martins & Astudillo, 2016)."""

    ordered = sorted((float(value) for value in values), reverse=True)
    cumulative = 0.0
    support = 0
    support_sum = 0.0
    for index, value in enumerate(ordered, start=1):
        cumulative += value
        if 1.0 + index * value > cumulative:
            support = index
            support_sum = cumulative
    if support == 0:
        return tuple(0.0 for _ in values)
    threshold = (support_sum - 1.0) / support
    projected = tuple(max(float(value) - threshold, 0.0) for value in values)
    total = sum(projected)
    if total <= 1e-12:
        return tuple(0.0 for _ in values)
    return tuple(value / total for value in projected)


def _normalized(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("embedding vector must be finite and non-empty")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("embedding vector must have non-zero norm")
    return tuple(value / norm for value in values)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _semantic_equivalence_classes(
    nodes: Sequence[MemoryNode],
    similarities: Mapping[tuple[str, str], float],
) -> tuple[tuple[str, ...], ...]:
    parent = {node.node_id: node.node_id for node in nodes}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for (left, right), similarity in similarities.items():
        if similarity >= SEMANTIC_EQUIVALENCE_COSINE:
            union(left, right)
    groups: dict[str, list[str]] = {}
    for node in nodes:
        groups.setdefault(find(node.node_id), []).append(node.node_id)
    order = {node.node_id: index for index, node in enumerate(nodes)}
    return tuple(
        tuple(sorted(members, key=order.__getitem__))
        for members in sorted(groups.values(), key=lambda values: order[values[0]])
    )


def _recurrence_edges(
    nodes: Sequence[MemoryNode],
    semantic_classes: Sequence[Sequence[str]],
    similarities: Mapping[tuple[str, str], float],
    excluded: set[frozenset[str]],
    *,
    score_source: str,
) -> list[SoftNavigationCorrelation]:
    order = {node.node_id: index for index, node in enumerate(nodes)}
    edges: list[SoftNavigationCorrelation] = []
    for class_index, members in enumerate(semantic_classes):
        ordered_members = sorted(members, key=order.__getitem__)
        for left, right in zip(ordered_members, ordered_members[1:]):
            if frozenset((left, right)) in excluded:
                continue
            edges.append(
                _edge(
                    left,
                    right,
                    channel="semantic_recurrence",
                    similarity=similarities[_ordered_pair(left, right)],
                    forward=1.0,
                    backward=1.0,
                    score_source=score_source,
                    provenance={
                        "pair_enumeration": "all_non_temporal_pairs",
                        "sparsifier": "temporal_chain_within_semantic_equivalence_class",
                        "semantic_class": class_index,
                    },
                )
            )
    return edges


def _edge(
    src: str,
    dst: str,
    *,
    channel: str,
    similarity: float,
    forward: float,
    backward: float,
    score_source: str,
    provenance: dict[str, object],
) -> SoftNavigationCorrelation:
    digest = hashlib.sha256(f"{src}\x1f{dst}\x1f{channel}".encode("utf-8")).hexdigest()[
        :20
    ]
    return SoftNavigationCorrelation(
        edge_id=f"soft-correlation:{digest}",
        src=src,
        dst=dst,
        channel=channel,
        semantic_similarity=similarity,
        src_to_dst_affinity=forward,
        dst_to_src_affinity=backward,
        score_source=score_source,
        evidence_refs=(src, dst),
        provenance={
            "producer": "memory_graph.soft_correlation.build_soft_semantic_correlations",
            "question_independent": True,
            **provenance,
        },
    )


def _center(node: MemoryNode) -> float:
    return (node.time_span.start_s + node.time_span.end_s) / 2.0


def _node_modality(node: MemoryNode) -> str:
    value = node.metadata.get("modality")
    if value is None:
        value = node.provenance.get("modality")
    return str(value or "unspecified")


def _l1_fingerprint(nodes: Sequence[MemoryNode]) -> str:
    payload = [
        node.to_dict()
        for node in sorted(
            nodes,
            key=lambda value: (
                value.time_span.start_s,
                value.time_span.end_s,
                value.node_id,
            ),
        )
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
