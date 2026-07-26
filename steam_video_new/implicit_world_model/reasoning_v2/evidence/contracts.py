"""Question-independent evidence-memory contracts for reasoning v2.

The central boundary is intentionally explicit: an address is safe to expose
before a read, while an evidence value is grounded content revealed only by an
executed read.  Legacy L1 nodes mixed both concepts in one ``MemoryNode``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class EvidenceAddress:
    node_id: str
    video_id: str
    start_s: float
    end_s: float
    event_family: str
    semantic_key: str = ""
    structural_tags: tuple[str, ...] = ()
    source_segments: tuple[str, ...] = ()
    embedding_ref: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.node_id or not self.video_id:
            raise ValueError("evidence address identifiers must be non-empty")
        if self.start_s < 0 or self.end_s < self.start_s:
            raise ValueError("evidence address has an invalid time span")
        if not self.event_family.strip():
            raise ValueError("evidence address requires a coarse event family")
        if len(self.semantic_key) > 480:
            raise ValueError("evidence address semantic key exceeds its bound")
        if len(self.structural_tags) != len(set(self.structural_tags)):
            raise ValueError("evidence address structural tags must be unique")


@dataclass(frozen=True)
class EntityMention:
    mention_id: str
    role: str
    entity_type: str
    surface: str
    visual_signature: str = ""
    track_status: str = "event_local_only"

    def __post_init__(self) -> None:
        if not self.mention_id or not self.entity_type:
            raise ValueError("entity mention requires an ID and entity type")


@dataclass(frozen=True)
class GroundedState:
    mention_id: str
    attribute: str
    value: str
    polarity: str = "positive"

    def __post_init__(self) -> None:
        if not self.mention_id or not self.attribute or not self.value:
            raise ValueError("grounded state is incomplete")
        if self.polarity not in {"positive", "negative"}:
            raise ValueError("grounded state polarity is invalid")


@dataclass(frozen=True)
class GroundedStateDelta:
    mention_id: str
    attribute: str
    before: str
    after: str

    def __post_init__(self) -> None:
        if not all((self.mention_id, self.attribute, self.before, self.after)):
            raise ValueError("grounded state delta is incomplete")
        if self.before == self.after:
            raise ValueError("grounded state delta must change its value")


@dataclass(frozen=True)
class EvidenceValue:
    descriptor: str
    predicate: str
    entities: tuple[EntityMention, ...] = ()
    states: tuple[GroundedState, ...] = ()
    state_delta: GroundedStateDelta | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.descriptor.strip() or not self.predicate.strip():
            raise ValueError("evidence value requires grounded text")
        mention_ids = {entity.mention_id for entity in self.entities}
        if any(state.mention_id not in mention_ids for state in self.states):
            raise ValueError("grounded state references an unknown mention")
        if (
            self.state_delta is not None
            and self.state_delta.mention_id not in mention_ids
        ):
            raise ValueError("state delta references an unknown mention")


@dataclass(frozen=True)
class EvidenceRecord:
    address: EvidenceAddress
    value: EvidenceValue


@dataclass(frozen=True)
class TemporalLink:
    edge_id: str
    src: str
    dst: str
    relation: str

    def __post_init__(self) -> None:
        if not self.edge_id or not self.src or not self.dst or self.src == self.dst:
            raise ValueError("temporal link is invalid")
        if self.relation not in {"temporal_next", "before", "overlaps", "during"}:
            raise ValueError("unsupported temporal relation")


@dataclass(frozen=True)
class EvidenceMemory:
    memory_id: str
    records: tuple[EvidenceRecord, ...]
    temporal_links: tuple[TemporalLink, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_id or not self.records:
            raise ValueError("evidence memory must be non-empty")
        ids = [record.address.node_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence memory contains duplicate addresses")
        known = set(ids)
        if any(
            link.src not in known or link.dst not in known
            for link in self.temporal_links
        ):
            raise ValueError("evidence memory contains an orphan temporal link")

    @property
    def record_by_id(self) -> dict[str, EvidenceRecord]:
        return {record.address.node_id: record for record in self.records}

    def address_view(self, acquired_ids: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        """Return model views without leaking unread grounded values."""

        acquired = set(acquired_ids)
        unknown = acquired - set(self.record_by_id)
        if unknown:
            raise ValueError(
                f"acquired evidence is absent from memory: {sorted(unknown)}"
            )
        return tuple(
            {
                "address": record.address,
                "acquired": record.address.node_id in acquired,
                "value": record.value if record.address.node_id in acquired else None,
            }
            for record in self.records
        )

    def read(self, node_id: str) -> EvidenceRecord:
        try:
            return self.record_by_id[node_id]
        except KeyError as exc:
            raise ValueError(f"unknown evidence address: {node_id}") from exc
