"""Separate substrate, navigation, transition, and planning gates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..evidence.contracts import EvidenceMemory
from ..navigation.contracts import NavigationGraph


@dataclass(frozen=True)
class FrozenReasoningCase:
    case_id: str
    entry_node_ids: tuple[str, ...]
    clue_node_groups: tuple[tuple[str, ...], ...]
    read_budget: int

    def __post_init__(self) -> None:
        if not self.case_id or not self.entry_node_ids or not self.clue_node_groups:
            raise ValueError("frozen reasoning case is incomplete")
        if self.read_budget < 1:
            raise ValueError("reasoning case read budget must be positive")


@dataclass(frozen=True)
class CaseReachability:
    case_id: str
    clue_retained: bool
    all_clues_reachable: bool
    shortest_clue_hops: tuple[int | None, ...]
    delayed_candidate: bool


@dataclass(frozen=True)
class CohortGateReport:
    case_count: int
    retained_case_count: int
    reachable_case_count: int
    delayed_candidate_count: int
    cases: tuple[CaseReachability, ...]
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.blockers


def evaluate_frozen_cohort(
    memory: EvidenceMemory,
    navigation: NavigationGraph,
    cases: Sequence[FrozenReasoningCase],
    *,
    minimum_case_count: int = 10,
    minimum_delayed_cases: int = 2,
) -> CohortGateReport:
    """Evaluator-only join; cases never influence graph construction."""

    known = set(memory.record_by_id)
    rows: list[CaseReachability] = []
    for case in cases:
        clue_retained = all(bool(set(group) & known) for group in case.clue_node_groups)
        # The entry itself is a grounded read.  Runtime therefore has only
        # read_budget - 1 graph traversals left after acquiring it.
        distances = _distances(
            navigation,
            case.entry_node_ids,
            max(0, case.read_budget - 1),
        )
        shortest = tuple(
            min(
                (distances[node_id] for node_id in group if node_id in distances),
                default=None,
            )
            for group in case.clue_node_groups
        )
        reachable = clue_retained and all(value is not None for value in shortest)
        # A structural delayed candidate must require a second graph hop; mere
        # multi-clue membership or a second read is not enough.
        delayed = reachable and any(
            value is not None and value >= 2 for value in shortest
        )
        rows.append(
            CaseReachability(
                case.case_id,
                clue_retained,
                reachable,
                shortest,
                delayed,
            )
        )
    retained = sum(row.clue_retained for row in rows)
    reachable = sum(row.all_clues_reachable for row in rows)
    delayed = sum(row.delayed_candidate for row in rows)
    blockers: list[str] = []
    if len(rows) < minimum_case_count:
        blockers.append("insufficient_fixed_cases")
    if retained != len(rows):
        blockers.append("clue_retention_incomplete")
    if reachable != len(rows):
        blockers.append("legal_navigation_reachability_incomplete")
    if delayed < minimum_delayed_cases:
        blockers.append("delayed_reasoning_cases_missing")
    return CohortGateReport(
        case_count=len(rows),
        retained_case_count=retained,
        reachable_case_count=reachable,
        delayed_candidate_count=delayed,
        cases=tuple(rows),
        blockers=tuple(blockers),
    )


def _distances(
    graph: NavigationGraph,
    starts: tuple[str, ...],
    maximum_hops: int,
) -> Mapping[str, int]:
    distances = {node_id: 0 for node_id in starts if node_id in set(graph.node_ids)}
    queue = deque(distances)
    while queue:
        node_id = queue.popleft()
        distance = distances[node_id]
        if distance >= maximum_hops:
            continue
        for proposal in graph.neighbors(node_id):
            target = proposal.dst if proposal.src == node_id else proposal.src
            if target not in distances:
                distances[target] = distance + 1
                queue.append(target)
    return distances


@dataclass(frozen=True)
class DecomposedArmMetrics:
    arm: str
    case_count: int
    answer_accuracy: float
    clue_recall: float
    delayed_success: float


@dataclass(frozen=True)
class OracleDecompositionReport:
    arms: tuple[DecomposedArmMetrics, ...]
    blockers: tuple[str, ...]

    @property
    def ready_for_model_training(self) -> bool:
        return not self.blockers


def evaluate_oracle_decomposition(
    arms: Sequence[DecomposedArmMetrics],
) -> OracleDecompositionReport:
    required = {
        "oracle_observation_oracle_effect",
        "predicted_observation_oracle_effect",
        "oracle_observation_predicted_effect",
        "predicted_observation_predicted_effect",
        "no_world_model",
        "shuffled_world_model",
    }
    by_name = {row.arm: row for row in arms}
    blockers: list[str] = []
    if not required.issubset(by_name):
        blockers.append("matched_oracle_arms_incomplete")
    oracle = by_name.get("oracle_observation_oracle_effect")
    if oracle is not None and oracle.answer_accuracy <= 0.5:
        blockers.append("oracle_headroom_not_established")
    return OracleDecompositionReport(tuple(arms), tuple(blockers))
