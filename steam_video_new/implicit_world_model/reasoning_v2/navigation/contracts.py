"""Typed L1.5 navigation proposals.

Proposals make a node locally reachable.  They are neither factual relations
nor calibrated probabilities, and model-facing views never expose affinities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ProposalKind(str, Enum):
    TEMPORAL = "temporal"
    SEMANTIC_RECURRENCE = "semantic_recurrence"
    SEMANTIC_NEIGHBOR = "semantic_neighbor"
    ENTITY_RECURRENCE_CANDIDATE = "entity_recurrence_candidate"
    EVENT_CONTINUATION_CANDIDATE = "event_continuation_candidate"


class ProposalCalibration(str, Enum):
    STRUCTURAL = "structural"
    UNCALIBRATED = "uncalibrated"
    CALIBRATED_WITH_HARD_NEGATIVES = "calibrated_with_hard_negatives"


@dataclass(frozen=True)
class NavigationProposal:
    proposal_id: str
    src: str
    dst: str
    kind: ProposalKind
    bidirectional: bool
    evidence_refs: tuple[str, ...]
    descriptor: tuple[str, ...] = ()
    calibration: ProposalCalibration = ProposalCalibration.UNCALIBRATED
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.src or not self.dst or self.src == self.dst:
            raise ValueError("navigation proposal is invalid")
        if not self.evidence_refs:
            raise ValueError("navigation proposal requires provenance references")
        if self.src not in self.evidence_refs or self.dst not in self.evidence_refs:
            raise ValueError("proposal evidence must cite both endpoints")

    def permits(self, source: str, target: str) -> bool:
        if source == self.src and target == self.dst:
            return True
        return self.bidirectional and source == self.dst and target == self.src

    def model_view(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind.value,
            "bidirectional": self.bidirectional,
            "descriptor": list(self.descriptor),
            "calibration": self.calibration.value,
        }


@dataclass(frozen=True)
class NavigationGraph:
    graph_id: str
    node_ids: tuple[str, ...]
    proposals: tuple[NavigationProposal, ...]
    entry_node_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        known = set(self.node_ids)
        if len(known) != len(self.node_ids):
            raise ValueError("navigation graph contains duplicate nodes")
        if not set(self.entry_node_ids).issubset(known):
            raise ValueError("navigation entry is absent from evidence memory")
        if any(row.src not in known or row.dst not in known for row in self.proposals):
            raise ValueError("navigation graph contains an orphan proposal")

    def neighbors(self, node_id: str) -> tuple[NavigationProposal, ...]:
        if node_id not in set(self.node_ids):
            raise ValueError("navigation cursor is absent from graph")
        return tuple(
            row
            for row in self.proposals
            if row.src == node_id or (row.bidirectional and row.dst == node_id)
        )


class NavigationActionKind(str, Enum):
    START = "start"
    FOLLOW = "follow"
    STOP = "stop"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class NavigationAction:
    action_id: str
    kind: NavigationActionKind
    target_id: str | None = None
    source_id: str | None = None
    proposal_id: str | None = None
    proposal_kind: ProposalKind | None = None
    reads_evidence: bool = False

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("navigation action requires an ID")
        if self.kind in {NavigationActionKind.START, NavigationActionKind.FOLLOW}:
            if not self.target_id or not self.reads_evidence:
                raise ValueError("read action requires an evidence target")
        if self.kind is NavigationActionKind.FOLLOW and not self.proposal_id:
            raise ValueError("follow action requires a proposal")

    @property
    def shared_key(self) -> tuple[str, ...]:
        if self.reads_evidence:
            return ("read", str(self.target_id))
        return (self.kind.value,)
