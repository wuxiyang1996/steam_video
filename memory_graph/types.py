"""Typed contracts for the first memory-graph prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any


SCHEMA_VERSION = "steam-causal-graph/v0.1"
OVERLAY_SCHEMA_VERSION = "steam-causal-overlay/v0.2"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_EMBEDDING_DIM = 2048


class RelationType(str, Enum):
    TEMPORAL_NEXT = "temporal_next"
    BEFORE = "before"
    OVERLAPS = "overlaps"
    DURING = "during"
    SAME_ENTITY = "same_entity"
    SAME_OBJECT = "same_object"
    SAME_INSTANCE_CANDIDATE = "same_instance_candidate"
    REAPPEARS_CANDIDATE = "reappears_candidate"
    STATE_TRANSITION = "state_transition"
    OBSERVATION_SUPPORT = "observation_support"
    TRANSITION_SUPPORT = "transition_support"
    RESPONSE_CANDIDATE = "response_candidate"
    EXPLAINS = "explains"
    ENABLES = "enables"
    CONTRADICTS = "contradicts"


class RelationStatus(str, Enum):
    DETERMINISTIC = "deterministic"
    UNCALIBRATED_PRIOR = "uncalibrated_prior"
    CALIBRATED_POSTERIOR = "calibrated_posterior"


class MechanismKind(str, Enum):
    NONE = "none"
    STATE_BRIDGE = "state_bridge"
    OBSERVABLE_PRECONDITION = "observable_precondition"
    RULE_RESPONSE = "rule_response"
    CONTACT_TRANSFER = "contact_transfer"


@dataclass(frozen=True)
class ConfidenceComponents:
    temporal_grounding: float | None = None
    entity_continuity: float | None = None
    state_delta_grounding: float | None = None
    mechanism_visibility: float | None = None
    effect_grounding: float | None = None
    alternative_cause_penalty: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1]")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConfidenceComponents:
        names = cls.__dataclass_fields__
        return cls(
            **{
                name: float(payload[name]) if payload.get(name) is not None else None
                for name in names
            }
        )

    def to_dict(self) -> dict[str, float]:
        return _drop_none(asdict(self))


@dataclass(frozen=True)
class CausalWitness:
    """Structured, revisable support for one candidate-causal relation."""

    witness_id: str
    relation: str
    cause_event_id: str
    effect_event_id: str
    mechanism: MechanismKind
    mechanism_detail: str
    evidence_refs: tuple[str, ...]
    minimal_support_set: tuple[str, ...]
    evidence_quotes: dict[str, str]
    mechanism_event_id: str | None = None
    affected_entity: dict[str, Any] | None = None
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    precondition: str | None = None
    state_bridge: str | None = None
    confidence_components: ConfidenceComponents = field(
        default_factory=ConfidenceComponents
    )
    alternative_explanations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.witness_id or not self.cause_event_id or not self.effect_event_id:
            raise ValueError("causal witness IDs must not be empty")
        if self.cause_event_id == self.effect_event_id:
            raise ValueError("causal witness endpoints must be different")
        if self.relation not in {
            RelationType.EXPLAINS.value,
            RelationType.ENABLES.value,
        }:
            raise ValueError("causal witness relation must be explains or enables")
        if self.mechanism is MechanismKind.NONE:
            raise ValueError("causal witness requires an observable mechanism")
        if len(self.mechanism_detail.strip().split()) < 3:
            raise ValueError("causal witness mechanism_detail is not substantive")
        if not self.evidence_refs:
            raise ValueError("causal witness must cite evidence")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("causal witness evidence_refs must be unique")
        support = set(self.minimal_support_set)
        required = {self.cause_event_id, self.effect_event_id}
        if not required.issubset(support):
            raise ValueError("minimal_support_set must include both causal endpoints")
        if self.mechanism_event_id and self.mechanism_event_id not in support:
            raise ValueError("minimal_support_set must include mechanism_event_id")
        if not self.evidence_quotes.get("src") or not self.evidence_quotes.get("dst"):
            raise ValueError("causal witness requires source and destination quotes")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CausalWitness:
        mechanism = payload.get("mechanism", MechanismKind.NONE.value)
        return cls(
            witness_id=str(payload.get("witness_id") or ""),
            relation=str(payload.get("relation") or ""),
            cause_event_id=str(payload.get("cause_event_id") or ""),
            effect_event_id=str(payload.get("effect_event_id") or ""),
            mechanism=MechanismKind(str(mechanism)),
            mechanism_detail=str(payload.get("mechanism_detail") or ""),
            evidence_refs=tuple(str(value) for value in payload.get("evidence_refs") or []),
            minimal_support_set=tuple(
                str(value) for value in payload.get("minimal_support_set") or []
            ),
            evidence_quotes={
                str(key): str(value)
                for key, value in (payload.get("evidence_quotes") or {}).items()
                if value is not None
            },
            mechanism_event_id=(
                str(payload["mechanism_event_id"])
                if payload.get("mechanism_event_id")
                else None
            ),
            affected_entity=payload.get("affected_entity"),
            before_state=payload.get("before_state"),
            after_state=payload.get("after_state"),
            precondition=(
                str(payload["precondition"]) if payload.get("precondition") else None
            ),
            state_bridge=(
                str(payload["state_bridge"]) if payload.get("state_bridge") else None
            ),
            confidence_components=ConfidenceComponents.from_dict(
                payload.get("confidence_components") or {}
            ),
            alternative_explanations=tuple(
                str(value) for value in payload.get("alternative_explanations") or []
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mechanism"] = self.mechanism.value
        payload["evidence_refs"] = list(self.evidence_refs)
        payload["minimal_support_set"] = list(self.minimal_support_set)
        payload["alternative_explanations"] = list(self.alternative_explanations)
        payload["confidence_components"] = self.confidence_components.to_dict()
        return _drop_none(payload)


@dataclass(frozen=True)
class VisualVerification:
    status: str
    model: str | None = None
    protocol_version: str = "causal-visual-reread/v0.1"
    video_windows: tuple[dict[str, Any], ...] = ()
    frame_records: tuple[dict[str, Any], ...] = ()
    checks: dict[str, bool | None] = field(default_factory=dict)
    evidence: dict[str, tuple[int, ...]] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {"not_requested", "pending", "passed", "failed", "inconclusive"}
        if self.status not in allowed:
            raise ValueError(f"unknown visual verification status: {self.status}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VisualVerification:
        return cls(
            status=str(payload.get("status") or ""),
            model=str(payload["model"]) if payload.get("model") else None,
            protocol_version=str(
                payload.get("protocol_version") or "causal-visual-reread/v0.1"
            ),
            video_windows=tuple(payload.get("video_windows") or []),
            frame_records=tuple(payload.get("frame_records") or []),
            checks=dict(payload.get("checks") or {}),
            evidence={
                str(key): tuple(int(index) for index in value)
                for key, value in (payload.get("evidence") or {}).items()
                if isinstance(value, (list, tuple))
            },
            reasons=tuple(str(value) for value in payload.get("reasons") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "status": self.status,
                "model": self.model,
                "protocol_version": self.protocol_version,
                "video_windows": list(self.video_windows),
                "frame_records": list(self.frame_records),
                "checks": self.checks,
                "evidence": {
                    key: list(indices) for key, indices in self.evidence.items()
                },
                "reasons": list(self.reasons),
            }
        )


@dataclass(frozen=True)
class TimeSpan:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise ValueError("time_span.start_s must be non-negative")
        if self.end_s < self.start_s:
            raise ValueError("time_span.end_s must be >= start_s")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TimeSpan:
        return cls(start_s=float(payload["start_s"]), end_s=float(payload["end_s"]))

    def overlaps(self, other: TimeSpan, *, tolerance_s: float = 1e-6) -> bool:
        return self.start_s < other.end_s - tolerance_s and other.start_s < self.end_s - tolerance_s

    def contains(self, other: TimeSpan, *, tolerance_s: float = 1e-6) -> bool:
        return self.start_s <= other.start_s + tolerance_s and self.end_s >= other.end_s - tolerance_s


@dataclass(frozen=True)
class EmbeddingRef:
    path: str
    model: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = DEFAULT_EMBEDDING_DIM
    dtype: str = "float32"
    normalized: bool = True
    row_index: int | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("embedding path must not be empty")
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be positive")


@dataclass
class MemoryNode:
    node_id: str
    video_id: str
    time_span: TimeSpan
    provenance: dict[str, Any]
    node_type: str = "event"
    text: str | None = None
    source_node_id: str | None = None
    source_segments: list[str] = field(default_factory=list)
    embedding_ref: EmbeddingRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must not be empty")
        if not self.video_id:
            raise ValueError("video_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return _drop_none(payload)


@dataclass
class RelationBelief:
    edge_id: str
    src: str
    dst: str
    relation_probabilities: dict[str, float]
    status: RelationStatus
    direction_confidence: float
    evidence_refs: list[str] = field(default_factory=list)
    warrant: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.edge_id or not self.src or not self.dst:
            raise ValueError("edge_id, src, and dst must not be empty")
        if self.src == self.dst:
            raise ValueError("self-relations are not allowed")
        if not math.isfinite(self.direction_confidence) or not 0.0 <= self.direction_confidence <= 1.0:
            raise ValueError("direction_confidence must be in [0, 1]")
        allowed = {relation.value for relation in RelationType}
        unknown = set(self.relation_probabilities) - allowed
        if unknown:
            raise ValueError(f"unknown relation types: {sorted(unknown)}")
        for relation, probability in self.relation_probabilities.items():
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{relation} probability must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return _drop_none(payload)


@dataclass
class MemoryGraph:
    graph_id: str
    example_id: str
    video_id: str
    nodes: list[MemoryNode]
    relations: list[RelationBelief]
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("memory graph contains duplicate node_id values")
        known = set(node_ids)
        for relation in self.relations:
            if relation.src not in known or relation.dst not in known:
                raise ValueError(f"relation {relation.edge_id} references an unknown node")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "example_id": self.example_id,
            "video_id": self.video_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "relations": [relation.to_dict() for relation in self.relations],
            "metadata": self.metadata,
        }


@dataclass
class CausalTemporalOverlay:
    """L1 navigation graph plus an L1.5 event-relation overlay.

    Candidate-causal relations remain event-only. Native L1 structural
    relations are stored separately and may guide reads without becoming
    answer evidence or bypassing event verifiers.
    """

    overlay_id: str
    example_id: str
    video_id: str
    l1_observations: list[MemoryNode]
    atomic_events: list[MemoryNode]
    relations: list[RelationBelief]
    l1_structural_relations: list[RelationBelief] = field(default_factory=list)
    schema_version: str = OVERLAY_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        l1_ids = [node.node_id for node in self.l1_observations]
        event_ids = [node.node_id for node in self.atomic_events]
        if len(l1_ids) != len(set(l1_ids)):
            raise ValueError("overlay contains duplicate L1 observation IDs")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("overlay contains duplicate atomic event IDs")
        overlap = set(l1_ids) & set(event_ids)
        if overlap:
            raise ValueError(f"L1 and L1.5 IDs must be disjoint: {sorted(overlap)}")
        known_events = set(event_ids)
        known_l1 = set(l1_ids)
        for event in self.atomic_events:
            if event.node_type != "atomic_event":
                raise ValueError(f"overlay event {event.node_id} is not atomic_event")
            unknown_evidence = set(event.source_segments) - set(l1_ids)
            if unknown_evidence:
                raise ValueError(
                    f"event {event.node_id} references unknown L1 evidence: "
                    f"{sorted(unknown_evidence)}"
                )
        for relation in self.relations:
            if relation.src not in known_events or relation.dst not in known_events:
                raise ValueError(
                    f"overlay relation {relation.edge_id} must reference atomic events"
                )
        for relation in self.l1_structural_relations:
            if relation.src not in known_l1 or relation.dst not in known_l1:
                raise ValueError(
                    f"L1 structural relation {relation.edge_id} must reference L1 observations"
                )
            forbidden = {"explains", "enables"} & set(
                relation.relation_probabilities
            )
            if forbidden:
                raise ValueError(
                    "candidate-causal relations are forbidden on the L1 navigation "
                    f"track: {sorted(forbidden)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "overlay_id": self.overlay_id,
            "example_id": self.example_id,
            "video_id": self.video_id,
            "l1_observations": [node.to_dict() for node in self.l1_observations],
            "atomic_events": [node.to_dict() for node in self.atomic_events],
            "relations": [relation.to_dict() for relation in self.relations],
            "l1_structural_relations": [
                relation.to_dict() for relation in self.l1_structural_relations
            ],
            "metadata": self.metadata,
        }


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
