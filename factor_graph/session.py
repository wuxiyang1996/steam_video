"""Stateful append-only GTSAM belief session for executed graph reads."""

from __future__ import annotations

from dataclasses import dataclass

from memory_graph.navigation import GraphReadAction
from memory_graph.types import CausalTemporalOverlay
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    GraphReadExecution,
)

from .measurement import (
    MeasurementCalibrationRegistry,
    MeasurementJournal,
    RelationMeasurement,
    VerifierDecision,
    measurement_from_execution,
)
from .overlay_adapter import (
    GTSAMOverlayAdapter,
    OverlayInferenceResult,
    changed_variables,
)


@dataclass(frozen=True)
class SessionUpdateResult:
    measurement: RelationMeasurement
    appended: bool
    belief_before: OverlayInferenceResult
    belief_after: OverlayInferenceResult
    changed_variables: tuple[str, ...]
    direct_changed_variables: tuple[str, ...]
    propagated_changed_variables: tuple[str, ...]


class GTSAMBeliefSession:
    """Own the numeric belief while retaining every categorical measurement."""

    name = "gtsam_executed_measurement_session/v0.1"

    def __init__(
        self,
        overlay: CausalTemporalOverlay,
        *,
        calibration: MeasurementCalibrationRegistry,
        activate_persisted_verified_measurements: bool = True,
        adapter: GTSAMOverlayAdapter | None = None,
    ) -> None:
        self.overlay = overlay
        self.calibration = calibration
        self.activate_persisted_verified_measurements = (
            activate_persisted_verified_measurements
        )
        self.adapter = adapter or GTSAMOverlayAdapter()
        self.journal = MeasurementJournal()

    def snapshot(self, *, include_pairwise_factors: bool = True) -> OverlayInferenceResult:
        return self.adapter.infer(
            self.overlay,
            activate_verified_measurements=self.activate_persisted_verified_measurements,
            include_pairwise_factors=include_pairwise_factors,
            measurement_journal=self.journal,
            calibration=self.calibration,
        )

    def update(
        self,
        *,
        action: GraphReadAction,
        execution: GraphReadExecution,
        decision: VerifierDecision,
        previously_grounded_evidence_refs: tuple[str, ...] = (),
    ) -> SessionUpdateResult:
        before = self.snapshot()
        measurement = measurement_from_execution(
            action=action,
            execution=execution,
            overlay=self.overlay,
            decision=decision,
            calibration_version=self.calibration.version,
            previously_grounded_evidence_refs=previously_grounded_evidence_refs,
        )
        appended = self.journal.append(measurement)
        after = self.snapshot()
        changed = changed_variables(before, after)
        direct = tuple(name for name in changed if name == measurement.variable_id)
        propagated = tuple(name for name in changed if name != measurement.variable_id)
        return SessionUpdateResult(
            measurement=measurement,
            appended=appended,
            belief_before=before,
            belief_after=after,
            changed_variables=changed,
            direct_changed_variables=direct,
            propagated_changed_variables=propagated,
        )
