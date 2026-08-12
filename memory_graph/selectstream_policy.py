"""Minimal SelectStream-style bounded-memory policy with causal protection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .causal_witness import witness_from_provenance
from .types import MemoryNode, RelationBelief


@dataclass(frozen=True)
class MemoryUtilityWeights:
    write_surprise: float = 0.75
    semantic_relevance: float = 1.0
    temporal_bridge: float = 0.5
    state_change: float = 1.0
    predictive_dependency: float = 1.25
    causal_witness: float = 2.0
    unresolved_hypothesis: float = 0.75
    redundancy: float = 1.0


@dataclass(frozen=True)
class MemoryPolicyDecision:
    keep: tuple[str, ...]
    merge_groups: tuple[tuple[str, ...], ...]
    evict: tuple[str, ...]
    protected_witness_sets: tuple[tuple[str, ...], ...]
    utility: dict[str, float]
    utility_components: dict[str, dict[str, float]]
    capacity: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep": list(self.keep),
            "merge_groups": [list(group) for group in self.merge_groups],
            "evict": list(self.evict),
            "protected_witness_sets": [
                list(group) for group in self.protected_witness_sets
            ],
            "utility": self.utility,
            "utility_components": self.utility_components,
            "capacity": self.capacity,
            "metadata": self.metadata,
        }


def plan_bounded_memory(
    nodes: list[MemoryNode],
    relations: list[RelationBelief],
    *,
    capacity: int,
    semantic_relevance: dict[str, float] | None = None,
    redundancy: dict[str, float] | None = None,
    pairwise_redundancy: Mapping[tuple[str, str], float] | None = None,
    weights: MemoryUtilityWeights = MemoryUtilityWeights(),
    merge_redundancy_threshold: float = 0.8,
) -> MemoryPolicyDecision:
    """Plan keep/merge/evict without breaking a verified causal witness.

    This function is question-independent when ``semantic_relevance`` is not
    supplied, as in the full-IWM main path.  Call ``materialize_bounded_memory``
    to execute the returned decision while preserving lineage and graph validity.
    """

    if capacity < 1:
        raise ValueError("memory capacity must be positive")
    node_by_id = {node.node_id: node for node in nodes}
    semantic_relevance = semantic_relevance or {}
    redundancy = redundancy or {}
    pairwise_redundancy = pairwise_redundancy or {}
    witness_sets = _protected_witness_sets(relations, known=set(node_by_id))
    protected = {node_id for group in witness_sets for node_id in group}
    if len(protected) > capacity:
        raise ValueError(
            "memory capacity cannot retain all verified causal witness members"
        )

    components: dict[str, dict[str, float]] = {}
    utility: dict[str, float] = {}
    for node in nodes:
        values = {
            "write_surprise": _write_surprise_value(node),
            "semantic_relevance": _unit(semantic_relevance.get(node.node_id, 0.0)),
            "temporal_bridge": _temporal_bridge_value(node.node_id, relations),
            "state_change": _state_change_value(node, relations),
            "predictive_dependency": _predictive_dependency_value(
                node.node_id,
                relations,
            ),
            "causal_witness": _causal_witness_value(node.node_id, relations),
            "unresolved_hypothesis": _unresolved_value(node.node_id, relations),
            "redundancy": _unit(redundancy.get(node.node_id, 0.0)),
        }
        components[node.node_id] = values
        utility[node.node_id] = (
            weights.write_surprise * values["write_surprise"]
            + weights.semantic_relevance * values["semantic_relevance"]
            + weights.temporal_bridge * values["temporal_bridge"]
            + weights.state_change * values["state_change"]
            + weights.predictive_dependency * values["predictive_dependency"]
            + weights.causal_witness * values["causal_witness"]
            + weights.unresolved_hypothesis * values["unresolved_hypothesis"]
            - weights.redundancy * values["redundancy"]
        )

    candidate_merge_groups = _safe_merge_groups(
        nodes,
        protected=protected,
        redundancy=redundancy,
        pairwise_redundancy=pairwise_redundancy,
        threshold=merge_redundancy_threshold,
    )
    merged_members = {node_id for group in candidate_merge_groups for node_id in group}
    units = [*candidate_merge_groups]
    units.extend(
        (node.node_id,) for node in nodes if node.node_id not in merged_members
    )
    units.sort(
        key=lambda group: (
            not bool(set(group) & protected),
            -max(utility[node_id] for node_id in group),
            min(node_by_id[node_id].time_span.start_s for node_id in group),
            group,
        )
    )
    retained_units = tuple(units[:capacity])
    keep_ids = {node_id for group in retained_units for node_id in group}
    evict_ids = {node.node_id for node in nodes} - keep_ids
    merge_groups = tuple(group for group in retained_units if len(group) > 1)
    ordered_keep = tuple(node.node_id for node in nodes if node.node_id in keep_ids)
    ordered_evict = tuple(node.node_id for node in nodes if node.node_id in evict_ids)
    return MemoryPolicyDecision(
        keep=ordered_keep,
        merge_groups=merge_groups,
        evict=ordered_evict,
        protected_witness_sets=witness_sets,
        utility=utility,
        utility_components=components,
        capacity=capacity,
        metadata={
            "policy": "selectstream_causal_witness_value/v0.2",
            "weights": asdict(weights),
            "merge_is_plan_only": True,
            "capacity_unit": "retained_node_after_merge",
            "pairwise_redundancy_available": bool(pairwise_redundancy),
        },
    )


def merge_preserves_causal_witness(
    node_ids: set[str],
    *,
    relations: list[RelationBelief],
) -> bool:
    """Return false when a merge would absorb only part of a witness set."""

    for witness_set in _protected_witness_sets(relations, known=None):
        overlap = node_ids & set(witness_set)
        if overlap and overlap != set(witness_set):
            return False
    return True


def _protected_witness_sets(
    relations: list[RelationBelief],
    *,
    known: set[str] | None,
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for relation in relations:
        witness = witness_from_provenance(relation.provenance)
        if witness is None:
            continue
        hard = relation.provenance.get("hard_verifier")
        if isinstance(hard, dict):
            record = hard.get(witness.relation)
            if isinstance(record, dict) and record.get("passed") is not True:
                continue
        group = tuple(dict.fromkeys(witness.minimal_support_set))
        if known is not None:
            group = tuple(node_id for node_id in group if node_id in known)
        if len(group) >= 2 and group not in seen:
            seen.add(group)
            groups.append(group)
    return tuple(groups)


def _temporal_bridge_value(node_id: str, relations: list[RelationBelief]) -> float:
    temporal = {"before", "temporal_next", "overlaps", "during"}
    degree = sum(
        1
        for relation in relations
        if node_id in {relation.src, relation.dst}
        and any(name in temporal for name in relation.relation_probabilities)
    )
    return min(1.0, degree / 2.0)


def _write_surprise_value(node: MemoryNode) -> float:
    localization = node.metadata.get("localization")
    coarse_window = (
        localization.get("coarse_window") if isinstance(localization, dict) else None
    )
    reason = (
        str(coarse_window.get("boundary_reason") or "")
        if isinstance(coarse_window, dict)
        else ""
    )
    if reason == "representation_surprise":
        return 1.0
    if node.metadata.get("state_change") or node.metadata.get("states"):
        return 1.0
    return 0.0


def _state_change_value(node: MemoryNode, relations: list[RelationBelief]) -> float:
    if node.metadata.get("states"):
        return 1.0
    return float(
        any(
            node.node_id in {relation.src, relation.dst}
            and relation.relation_probabilities.get("state_transition", 0.0) >= 0.5
            for relation in relations
        )
    )


def _causal_witness_value(node_id: str, relations: list[RelationBelief]) -> float:
    values: list[float] = []
    for relation in relations:
        witness = witness_from_provenance(relation.provenance)
        if witness is None or node_id not in witness.minimal_support_set:
            continue
        confidence_values = [
            value
            for value in asdict(witness.confidence_components).values()
            if value is not None
        ]
        if confidence_values:
            values.append(sum(confidence_values) / len(confidence_values))
        else:
            values.append(
                max(
                    relation.relation_probabilities.get("explains", 0.0),
                    relation.relation_probabilities.get("enables", 0.0),
                )
            )
    return max(values, default=0.0)


def _predictive_dependency_value(
    node_id: str,
    relations: list[RelationBelief],
) -> float:
    dependency_types = {
        "transition_support",
        "response_candidate",
        "observation_support",
    }
    values = [
        probability
        for relation in relations
        if node_id in {relation.src, relation.dst}
        for name, probability in relation.relation_probabilities.items()
        if name in dependency_types
    ]
    return max(values, default=0.0)


def _unresolved_value(node_id: str, relations: list[RelationBelief]) -> float:
    for relation in relations:
        witness = witness_from_provenance(relation.provenance)
        if witness is None or node_id not in witness.minimal_support_set:
            continue
        visual = relation.provenance.get("visual_verification")
        if not isinstance(visual, dict) or visual.get("status") in {
            "pending",
            "inconclusive",
            "not_requested",
            None,
        }:
            return 1.0
    return 0.0


def _safe_merge_groups(
    nodes: list[MemoryNode],
    *,
    protected: set[str],
    redundancy: dict[str, float],
    pairwise_redundancy: Mapping[tuple[str, str], float],
    threshold: float,
) -> tuple[tuple[str, ...], ...]:
    ordered = sorted(
        nodes,
        key=lambda node: (node.time_span.start_s, node.time_span.end_s, node.node_id),
    )
    groups: list[tuple[str, ...]] = []
    run: list[MemoryNode] = []

    def flush() -> None:
        if len(run) > 1:
            groups.append(tuple(node.node_id for node in run))

    for node in ordered:
        if not run:
            run.append(node)
            continue
        left = run[-1]
        pair_score = _pair_value(pairwise_redundancy, left.node_id, node.node_id)
        pairwise_merge = pair_score is not None and pair_score >= threshold
        legacy_scalar_merge = (
            pair_score is None
            and min(
                _unit(redundancy.get(left.node_id, 0.0)),
                _unit(redundancy.get(node.node_id, 0.0)),
            )
            >= threshold
            and set(left.source_segments) == set(node.source_segments)
        )
        if (
            left.node_id not in protected
            and node.node_id not in protected
            and (pairwise_merge or legacy_scalar_merge)
        ):
            run.append(node)
            continue
        flush()
        run = [node]
    flush()
    return tuple(groups)


def _pair_value(
    values: Mapping[tuple[str, str], float], left: str, right: str
) -> float | None:
    value = values.get((left, right))
    if value is None:
        value = values.get((right, left))
    return _unit(value) if value is not None else None


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
