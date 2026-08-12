"""Complete-graph entry protocols and evaluator-only path targets.

Both protocols retain the same question-independent L1/L1.5 substrate.  They
only differ in how the initial entry frontier is obtained.  Hidden clue IDs are
used by the oracle protocol and metrics only; they are never included in a
model payload.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol

from ..evidence import EvidenceMemory
from ..navigation import NavigationGraph


class EntryProtocol(str, Enum):
    ORACLE = "oracle_entry"
    LEARNED = "learned_entry"


class EntryLocalizer(Protocol):
    audits: list[dict[str, Any]]

    def localize(
        self,
        *,
        question: str,
        missing_roles: tuple[str, ...],
        memory: EvidenceMemory,
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class EntryFrontier:
    protocol: EntryProtocol
    node_ids: tuple[str, ...]
    audit: dict[str, Any]


def build_entry_frontiers(
    *,
    question: str,
    missing_roles: tuple[str, ...],
    memory: EvidenceMemory,
    expected_first_clue_ids: tuple[str, ...],
    localizer: EntryLocalizer,
) -> tuple[EntryFrontier, EntryFrontier]:
    """Build matched complete-graph protocols without exposing hidden clues."""

    known = set(memory.record_by_id)
    oracle_ids = tuple(dict.fromkeys(expected_first_clue_ids))
    if not oracle_ids:
        raise ValueError("oracle-entry evaluation requires a first clue")
    if not set(oracle_ids).issubset(known):
        raise ValueError("oracle-entry clue is absent from the complete graph")
    learned_ids = localizer.localize(
        question=question,
        missing_roles=missing_roles,
        memory=memory,
    )
    learned_audit = dict(localizer.audits[-1])
    return (
        EntryFrontier(
            EntryProtocol.ORACLE,
            oracle_ids,
            {
                "status": "located",
                "source": "hidden_evaluator_only",
                "selected_node_ids": list(oracle_ids),
                "selected_anchor_count": len(oracle_ids),
                "candidate_address_count": len(memory.records),
                "all_addresses_available": True,
                "hidden_clue_exposed_to_model": False,
                "top_k_applied": False,
                "numeric_score_used": False,
            },
        ),
        EntryFrontier(EntryProtocol.LEARNED, learned_ids, learned_audit),
    )


def with_entry_frontier(
    graph: NavigationGraph,
    frontier: EntryFrontier,
) -> NavigationGraph:
    """Change only entry IDs; preserve every complete-graph node and proposal."""

    known = set(graph.node_ids)
    if not set(frontier.node_ids).issubset(known):
        raise ValueError("entry frontier contains a node outside the graph")
    updated = replace(
        graph,
        entry_node_ids=frontier.node_ids,
        metadata={
            **graph.metadata,
            "entry_protocol": frontier.protocol.value,
            "complete_graph_preserved": True,
        },
    )
    if updated.node_ids != graph.node_ids or updated.proposals != graph.proposals:
        raise AssertionError("entry protocol mutated the complete graph")
    return updated


def shortest_path_next_hops(
    graph: NavigationGraph,
    *,
    source_ids: tuple[str, ...],
    goal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return evaluator-only first hops on any shortest route to a later clue."""

    known = set(graph.node_ids)
    sources = tuple(node_id for node_id in source_ids if node_id in known)
    goals = {node_id for node_id in goal_ids if node_id in known}
    if not sources or not goals:
        return ()
    adjacency = {node_id: set() for node_id in graph.node_ids}
    for proposal in graph.proposals:
        adjacency[proposal.src].add(proposal.dst)
        if proposal.bidirectional:
            adjacency[proposal.dst].add(proposal.src)

    distance_to_goal: dict[str, int] = {}
    queue = deque((goal, 0) for goal in sorted(goals))
    while queue:
        node_id, distance = queue.popleft()
        if node_id in distance_to_goal:
            continue
        distance_to_goal[node_id] = distance
        queue.extend(
            (neighbor, distance + 1)
            for neighbor in sorted(adjacency[node_id])
            if neighbor not in distance_to_goal
        )
    candidates: set[str] = set()
    for source in sources:
        source_distance = distance_to_goal.get(source)
        if source_distance is None:
            continue
        if source_distance == 0:
            candidates.add(source)
            continue
        candidates.update(
            neighbor
            for neighbor in adjacency[source]
            if distance_to_goal.get(neighbor) == source_distance - 1
        )
    return tuple(sorted(candidates))


def shortest_path_distance(
    graph: NavigationGraph,
    *,
    source_ids: tuple[str, ...],
    goal_ids: tuple[str, ...],
) -> int | None:
    """Return the exact unweighted proposal distance used by route metrics."""

    known = set(graph.node_ids)
    sources = {node_id for node_id in source_ids if node_id in known}
    goals = {node_id for node_id in goal_ids if node_id in known}
    if not sources or not goals:
        return None
    if sources & goals:
        return 0
    adjacency = {node_id: set() for node_id in graph.node_ids}
    for proposal in graph.proposals:
        adjacency[proposal.src].add(proposal.dst)
        if proposal.bidirectional:
            adjacency[proposal.dst].add(proposal.src)
    visited = set(sources)
    frontier = sources
    distance = 0
    while frontier:
        distance += 1
        next_frontier = {
            neighbor
            for node_id in frontier
            for neighbor in adjacency[node_id]
            if neighbor not in visited
        }
        if next_frontier & goals:
            return distance
        visited.update(next_frontier)
        frontier = next_frontier
    return None


def audit_clue_grounded_values(
    memory: EvidenceMemory,
    clue_groups: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    """Audit value richness with hidden IDs, without serializing clue content."""

    by_id = memory.record_by_id
    groups = []
    for index, node_ids in enumerate(clue_groups):
        rows = []
        for node_id in node_ids:
            record = by_id.get(node_id)
            if record is None:
                continue
            provenance = record.value.provenance
            subtitle = provenance.get("subtitle_enrichment")
            rows.append(
                {
                    "node_id": node_id,
                    "descriptor_char_count": len(record.value.descriptor),
                    "descriptor_differs_from_predicate": (
                        record.value.descriptor.strip()
                        != record.value.predicate.strip()
                    ),
                    "time_aligned_subtitle_present": isinstance(subtitle, dict),
                    "embedding_present": record.address.embedding_ref is not None,
                }
            )
        groups.append(
            {
                "clue_index": index,
                "retained_node_count": len(rows),
                "grounded_value_enriched": any(
                    row["descriptor_differs_from_predicate"]
                    and row["time_aligned_subtitle_present"]
                    for row in rows
                ),
                "nodes": rows,
            }
        )
    return {
        "hidden_evaluator_only": True,
        "clue_group_count": len(groups),
        "all_clue_groups_have_enriched_value": bool(groups)
        and all(row["grounded_value_enriched"] for row in groups),
        "groups": groups,
    }
