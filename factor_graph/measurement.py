"""Strict executed-read to relation-measurement boundary for GTSAM belief."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Iterable

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    GraphReadExecution,
)


class MeasurementOutcome(str, Enum):
    SUPPORTS = "supports"
    REJECTS = "rejects"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class VerifierDecision:
    """Categorical verifier output; deliberately contains no confidence field."""

    edge_id: str
    relation: str
    outcome: MeasurementOutcome
    verifier_name: str
    verifier_version: str
    evidence_refs: tuple[str, ...]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.edge_id, self.relation, self.verifier_name, self.verifier_version)):
            raise ValueError("verifier decision identity fields must not be empty")
        if not self.evidence_refs:
            raise ValueError("verifier decision requires grounded evidence refs")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("verifier evidence refs must be unique")


@dataclass(frozen=True)
class RelationMeasurement:
    measurement_id: str
    action_id: str
    overlay_id: str
    edge_id: str
    relation: str
    variable_id: str
    outcome: MeasurementOutcome
    observation_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    verifier_name: str
    verifier_version: str
    calibration_version: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = (
            self.measurement_id,
            self.action_id,
            self.overlay_id,
            self.edge_id,
            self.relation,
            self.variable_id,
            self.verifier_name,
            self.verifier_version,
            self.calibration_version,
        )
        if not all(required):
            raise ValueError("relation measurement identity fields must not be empty")
        if not self.observation_ids or not self.evidence_refs:
            raise ValueError("relation measurement requires observations and evidence")

    def to_dict(self) -> dict[str, object]:
        """Audit representation; intentionally contains no numeric belief values."""

        return {
            "measurement_id": self.measurement_id,
            "action_id": self.action_id,
            "overlay_id": self.overlay_id,
            "edge_id": self.edge_id,
            "relation": self.relation,
            "variable_id": self.variable_id,
            "outcome": self.outcome.value,
            "observation_ids": list(self.observation_ids),
            "evidence_refs": list(self.evidence_refs),
            "verifier_name": self.verifier_name,
            "verifier_version": self.verifier_version,
            "calibration_version": self.calibration_version,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CalibrationEntry:
    likelihood_false: float
    likelihood_true: float

    def __post_init__(self) -> None:
        values = (self.likelihood_false, self.likelihood_true)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("measurement likelihoods must be finite and positive")


class MeasurementCalibrationRegistry:
    """Internal numeric calibration; never populated from LLM text."""

    def __init__(
        self,
        *,
        version: str,
        entries: dict[MeasurementOutcome, CalibrationEntry],
    ) -> None:
        if not version:
            raise ValueError("calibration version must not be empty")
        required = {MeasurementOutcome.SUPPORTS, MeasurementOutcome.REJECTS}
        if not required <= set(entries):
            raise ValueError("calibration requires supports and rejects entries")
        self.version = version
        self._entries = dict(entries)

    def lookup(self, outcome: MeasurementOutcome) -> CalibrationEntry | None:
        if outcome is MeasurementOutcome.INCONCLUSIVE:
            return None
        return self._entries[outcome]

    @classmethod
    def pilot(cls) -> MeasurementCalibrationRegistry:
        """Explicit non-production constants for mechanism/integration tests."""

        return cls(
            version="pilot_relation_measurement/v0.1-not-calibrated",
            entries={
                MeasurementOutcome.SUPPORTS: CalibrationEntry(0.01, 0.99),
                MeasurementOutcome.REJECTS: CalibrationEntry(0.99, 0.01),
            },
        )


class MeasurementJournal:
    """Append-only, idempotent journal that retains conflicting evidence."""

    def __init__(self) -> None:
        self._measurements: dict[str, RelationMeasurement] = {}

    @property
    def measurements(self) -> tuple[RelationMeasurement, ...]:
        return tuple(self._measurements.values())

    def append(self, measurement: RelationMeasurement) -> bool:
        existing = self._measurements.get(measurement.measurement_id)
        if existing is None:
            self._measurements[measurement.measurement_id] = measurement
            return True
        if existing != measurement:
            raise ValueError(
                f"measurement id collision with different content: "
                f"{measurement.measurement_id}"
            )
        return False

    def to_records(self) -> list[dict[str, object]]:
        return [measurement.to_dict() for measurement in self.measurements]


def measurement_from_execution(
    *,
    action: GraphReadAction,
    execution: GraphReadExecution,
    overlay: CausalTemporalOverlay,
    decision: VerifierDecision,
    calibration_version: str,
) -> RelationMeasurement:
    """Create a measurement only from executed, grounded, independently verified data."""

    invocation = execution.skill_invocation
    action_id = str(invocation.get("node_id") or "")
    if not action_id:
        raise ValueError("executed skill invocation requires a stable node_id")
    if invocation.get("status") != "executed":
        raise ValueError("only a successfully executed skill can create a measurement")
    if action.action_type is NavigationActionType.STOP:
        raise ValueError("stop action cannot create a relation measurement")
    observation_ids = tuple(node.node_id for node in execution.observations)
    if not observation_ids:
        raise ValueError("measurement requires at least one real observation")
    returned = {
        str(value)
        for value in (
            (invocation.get("outputs") or {}).get("real_observation_ids") or []
        )
    }
    if not set(observation_ids) <= returned:
        raise ValueError("execution observations are not recorded in skill outputs")

    edges = {
        edge.edge_id: edge
        for edge in overlay.relations + overlay.l1_structural_relations
    }
    edge = edges.get(decision.edge_id)
    if edge is None or decision.relation not in edge.relation_probabilities:
        raise ValueError("verifier decision does not reference a known edge relation")
    action_refs = set(action.target_ids)
    if action.source_id:
        action_refs.add(action.source_id)
    if not {edge.src, edge.dst} <= action_refs:
        raise ValueError("executed action does not cover both relation endpoints")
    if action.relation and action.relation != decision.relation:
        raise ValueError("verifier relation does not match the executed graph read")

    grounded = set(observation_ids)
    for node in execution.observations:
        grounded.update(node.source_segments)
    invocation_evidence = {
        str(value) for value in invocation.get("evidence_refs") or []
    }
    grounded.update(invocation_evidence)
    if not set(decision.evidence_refs) <= grounded:
        raise ValueError("verifier cites evidence not grounded by the execution")

    variable_id = f"relation::{edge.edge_id}::{decision.relation}"
    digest = _measurement_digest(
        action_id,
        overlay.overlay_id,
        decision.edge_id,
        decision.relation,
        decision.outcome.value,
        decision.verifier_name,
        decision.verifier_version,
        decision.evidence_refs,
    )
    return RelationMeasurement(
        measurement_id=f"measurement:{digest}",
        action_id=action_id,
        overlay_id=overlay.overlay_id,
        edge_id=decision.edge_id,
        relation=decision.relation,
        variable_id=variable_id,
        outcome=decision.outcome,
        observation_ids=observation_ids,
        evidence_refs=decision.evidence_refs,
        verifier_name=decision.verifier_name,
        verifier_version=decision.verifier_version,
        calibration_version=calibration_version,
        reasons=decision.reasons,
    )


def _measurement_digest(*parts: str | Iterable[str]) -> str:
    normalized = ["\x1f".join(part) if not isinstance(part, str) else part for part in parts]
    return hashlib.sha256("\x1e".join(normalized).encode("utf-8")).hexdigest()[:24]
