"""Evaluator-only evidence substrate diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from .contracts import EvidenceMemory


@dataclass(frozen=True)
class EvidenceQualityReport:
    node_count: int
    mean_descriptor_words: float
    entity_coverage: float
    state_coverage: float
    state_delta_coverage: float
    candidate_track_rate: float
    learned_windowing: bool
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.blockers


def audit_evidence_memory(
    memory: EvidenceMemory,
    *,
    minimum_descriptor_words: float = 3.0,
    minimum_entity_coverage: float = 0.5,
) -> EvidenceQualityReport:
    values = [record.value for record in memory.records]
    count = len(values)
    words = [len(value.descriptor.split()) for value in values]
    entity_coverage = _ratio(sum(bool(value.entities) for value in values), count)
    state_coverage = _ratio(sum(bool(value.states) for value in values), count)
    delta_coverage = _ratio(
        sum(value.state_delta is not None for value in values), count
    )
    mentions = [entity for value in values for entity in value.entities]
    candidate_rate = _ratio(
        sum(entity.track_status != "accepted_identity" for entity in mentions),
        len(mentions),
    )
    learned = bool(memory.metadata.get("formal_learned_representation"))
    blockers: list[str] = []
    if fmean(words) < minimum_descriptor_words:
        blockers.append("descriptors_too_compressed")
    if entity_coverage < minimum_entity_coverage:
        blockers.append("insufficient_entity_grounding")
    if not learned:
        blockers.append("windowing_uses_nonlearned_fallback")
    return EvidenceQualityReport(
        node_count=count,
        mean_descriptor_words=fmean(words),
        entity_coverage=entity_coverage,
        state_coverage=state_coverage,
        state_delta_coverage=delta_coverage,
        candidate_track_rate=candidate_rate,
        learned_windowing=learned,
        blockers=tuple(blockers),
    )


def clue_retention(
    memory: EvidenceMemory,
    clue_intervals: Iterable[tuple[float, float]],
) -> float:
    """Join hidden clue intervals after memory freeze; never a construction input."""

    clues = tuple(clue_intervals)
    if not clues:
        return 1.0
    retained = 0
    for start, end in clues:
        if any(
            record.address.start_s < end and start < record.address.end_s
            for record in memory.records
        ):
            retained += 1
    return retained / len(clues)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
