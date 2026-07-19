"""Contracts for grounded atomic-event hypotheses and L1 human audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .types import TimeSpan


@dataclass(frozen=True)
class EntityMention:
    """An entity mention local to one atomic event, not a global identity claim."""

    mention_id: str
    role: str
    entity_type: str
    surface: str
    confidence: float
    grounding_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.mention_id or not self.role or not self.entity_type or not self.surface:
            raise ValueError("entity mention identifiers, role, type, and surface are required")
        _validate_probability(self.confidence, "entity mention confidence")


@dataclass(frozen=True)
class StateAssertion:
    """A visible state assertion; cross-event transitions are inferred separately."""

    mention_id: str
    attribute: str
    value: str
    confidence: float
    polarity: str = "positive"

    def __post_init__(self) -> None:
        if not self.mention_id or not self.attribute or not self.value:
            raise ValueError("state assertion mention_id, attribute, and value are required")
        if self.polarity not in {"positive", "negative"}:
            raise ValueError("state assertion polarity must be positive or negative")
        _validate_probability(self.confidence, "state assertion confidence")


@dataclass(frozen=True)
class AtomicEvent:
    """One revisable L1.5 event hypothesis grounded in immutable L1 evidence."""

    event_id: str
    video_id: str
    time_span: TimeSpan
    predicate: str
    evidence_refs: tuple[str, ...]
    confidence: float
    participants: tuple[EntityMention, ...] = ()
    states: tuple[StateAssertion, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.video_id or not self.predicate.strip():
            raise ValueError("atomic event id, video id, and predicate are required")
        if not self.evidence_refs:
            raise ValueError("atomic events must cite at least one L1 evidence node")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("atomic event evidence_refs must be unique")
        _validate_probability(self.confidence, "atomic event confidence")
        mention_ids = {participant.mention_id for participant in self.participants}
        for state in self.states:
            if state.mention_id not in mention_ids:
                raise ValueError(
                    f"state assertion references unknown local mention {state.mention_id!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_refs"] = list(self.evidence_refs)
        payload["participants"] = [asdict(participant) for participant in self.participants]
        payload["states"] = [asdict(state) for state in self.states]
        return payload


@dataclass(frozen=True)
class EntityLinkJudgment:
    """Independent judgment for one proposed cross-node entity link."""

    src: str
    dst: str
    correct: bool

    def __post_init__(self) -> None:
        if not self.src or not self.dst or self.src == self.dst:
            raise ValueError("entity link judgment requires two different node IDs")
        _require_bool(self.correct, "entity link judgment correct")


@dataclass(frozen=True)
class L1HumanAudit:
    """Independent labels required for semantic L1 reliability metrics."""

    grounded_event_correct: dict[str, bool]
    compound_event: dict[str, bool]
    gold_key_event_ids: tuple[str, ...]
    covered_key_event_ids: tuple[str, ...]
    entity_link_judgments: tuple[EntityLinkJudgment, ...]
    annotator: str
    protocol_version: str

    def __post_init__(self) -> None:
        if not self.annotator or not self.protocol_version:
            raise ValueError("human audit annotator and protocol_version are required")
        unknown_covered = set(self.covered_key_event_ids) - set(self.gold_key_event_ids)
        if unknown_covered:
            raise ValueError(
                f"covered key events are absent from gold set: {sorted(unknown_covered)}"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> L1HumanAudit:
        return cls(
            grounded_event_correct={
                str(node_id): _require_bool(value, f"grounded_event_correct.{node_id}")
                for node_id, value in (payload.get("grounded_event_correct") or {}).items()
            },
            compound_event={
                str(node_id): _require_bool(value, f"compound_event.{node_id}")
                for node_id, value in (payload.get("compound_event") or {}).items()
            },
            gold_key_event_ids=tuple(str(value) for value in payload.get("gold_key_event_ids") or []),
            covered_key_event_ids=tuple(
                str(value) for value in payload.get("covered_key_event_ids") or []
            ),
            entity_link_judgments=tuple(
                EntityLinkJudgment(
                    src=str(value.get("src") or ""),
                    dst=str(value.get("dst") or ""),
                    correct=_require_bool(value.get("correct"), "entity_link_judgments.correct"),
                )
                for value in _require_dict_list(
                    payload.get("entity_link_judgments") or [],
                    "entity_link_judgments",
                )
            ),
            annotator=str(payload.get("annotator") or ""),
            protocol_version=str(payload.get("protocol_version") or ""),
        )


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _require_dict_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of objects")
    return value
